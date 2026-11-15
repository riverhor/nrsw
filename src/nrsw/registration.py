"""Reciprocal SIFT registration and single-pass field rectification."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np


MAX_EDGE = 1600

MAX_FEATURES = 4000

MAX_PIXELS = 32_000_000

MAX_ENCODED = 64 * 1024 * 1024

RATIO = .75

DEDUP = 4.

GRID_GUARD = 8.

MIN_MATCHES = 12

RANSAC_THRESHOLD = 3.

RANSAC_ITERATIONS = 2000

RANSAC_CONFIDENCE = .999

ERROR_MEDIAN = 2.5

ERROR_P90 = 5.

ERROR_FRACTION = .80

FIT_FRACTION = .40

FIT_HULL_FRACTION = .05

RESERVED_HULL_FRACTION = .03

POLE_RATIO = 1e-6

CORNER_MARGIN = .02

MIN_EDGE = 32.

MIN_QUAD_AREA = 1024.

ROI_DISTANCE = .15

def _matrix(np, value):
    return value.tolist() if value is not None and np.isfinite(value).all() else None

def _project(np, matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    valid = np.isfinite(homogeneous).all(axis=1) & (homogeneous[:, 2] != 0)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:]
    return result, homogeneous[:, 2]

def _summary(np, errors):
    errors = np.asarray(errors, dtype=np.float64)
    finite = np.isfinite(errors)
    complete = bool(len(errors) and finite.all())
    return dict(count=len(errors), finite_count=int(finite.sum()),
                median=float(np.median(errors)) if complete else None,
                p90=float(np.quantile(errors, .9, method='linear')) if complete else None,
                maximum=float(errors.max()) if complete else None,
                fraction_at_most_5=float((finite & (errors <= 5.)).sum() / len(errors)) if len(errors) else 0.)

def _area(points):
    return sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1])) / 2 if len(points) >= 3 else 0.

def _hull(points):
    points = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(points) < 3:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def _inside(hull, point):
    if len(hull) < 3:
        return False
    return all((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]) >= 0
               for a, b in zip(hull, hull[1:]+hull[:1]))

def _spread(points, width, height):
    hull = _hull(points)
    quadrants = [0]*4
    cells = [0]*16
    for x, y in points:
        quadrants[int(y >= height/2)*2+int(x >= width/2)] += 1
        cells[min(3, int(4*y/height))*4+min(3, int(4*x/width))] += 1
    return dict(count=len(points), hull=hull,
                hull_area_fraction=abs(_area(hull))/(width*height),
                occupied_quadrants=sum(v > 0 for v in quadrants), quadrants=quadrants, cells=cells)

def _resize_matrix(np, width, height, out_width, out_height):
    sx, sy = out_width/width, out_height/height
    return np.array([[sx, 0., (sx-1.)/2.], [0., sy, (sy-1.)/2.], [0., 0., 1.]], dtype=np.float64)

def _work_image(np, cv2, gray, work, guard):
    guard()
    height, width = gray.shape
    scale = min(1., MAX_EDGE/max(width, height))
    out_width, out_height = max(1, round(width*scale)), max(1, round(height*scale))
    if (out_width, out_height) != (width, height):
        work['resize_calls'] += 1
        gray = cv2.resize(gray, (out_width, out_height), interpolation=cv2.INTER_AREA)
    S = _resize_matrix(np, width, height, out_width, out_height)
    guard()
    return gray, S

def _features(np, cv2, gray, work, guard):
    guard()
    detector = cv2.SIFT_create(MAX_FEATURES, 3, .04, 10., 1.6)
    work['SIFT_calls'] += 1
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    # SIFT may retain tied responses beyond nfeatures. Freeze an explicit cap.
    order = sorted(range(len(keypoints)), key=lambda i: (-keypoints[i].response,
                   keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].size,
                   keypoints[i].angle, i))[:MAX_FEATURES]
    points = np.array([keypoints[i].pt for i in order], dtype=np.float64).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.ascontiguousarray(descriptors[order], dtype=np.float32)
    assert len(points) == len(descriptors) <= MAX_FEATURES
    assert np.isfinite(points).all() and np.isfinite(descriptors).all()
    work['retained_keypoints'] += len(points)
    work['descriptor_bytes'] += descriptors.nbytes
    guard()
    return points, descriptors

def _matches(np, cv2, pt, dt, ps, ds, work, guard):
    if min(len(dt), len(ds)) < 2:
        return [], dict(forward_ratio=0, reverse_ratio=0, reciprocal=0, deduplicated=0)
    guard()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    work['directed_BF_knn_calls'] += 1
    forward = matcher.knnMatch(dt, ds, k=2)
    guard()
    work['directed_BF_knn_calls'] += 1
    reverse = matcher.knnMatch(ds, dt, k=2)
    f = {pair[0].queryIdx: pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < RATIO*pair[1].distance}
    b = {pair[0].queryIdx: pair[0] for pair in reverse if len(pair) == 2 and pair[0].distance < RATIO*pair[1].distance}
    reciprocal = [m for i, m in f.items() if m.trainIdx in b and b[m.trainIdx].trainIdx == i]
    reciprocal.sort(key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
    gt, gs, selected = {}, {}, []
    def near(grid, p):
        bx, by = math.floor(p[0]/DEDUP), math.floor(p[1]/DEDUP)
        return any((p[0]-v[0])**2+(p[1]-v[1])**2 <= DEDUP**2
                   for x in range(bx-1, bx+2) for y in range(by-1, by+2) for v in grid.get((x, y), ()))
    def insert(grid, p):
        key = math.floor(p[0]/DEDUP), math.floor(p[1]/DEDUP)
        grid.setdefault(key, []).append(p)
    for match in reciprocal:
        a, bpoint = pt[match.queryIdx], ps[match.trainIdx]
        if near(gt, a) or near(gs, bpoint):
            continue
        insert(gt, a); insert(gs, bpoint)
        selected.append((match.queryIdx, match.trainIdx))
    guard()
    return selected, dict(forward_ratio=len(f), reverse_ratio=len(b), reciprocal=len(reciprocal), deduplicated=len(selected))

def _partition(np, matches, pt, ps, width, height):
    fit, reserved, excluded = [], [], 0
    for i, j in matches:
        x, y = pt[i]
        assert 0 <= x < width and 0 <= y < height
        if any(abs(x-width*k/4) <= GRID_GUARD or abs(y-height*k/4) <= GRID_GUARD for k in (1, 2, 3)):
            excluded += 1
            continue
        cellx, celly = min(3, int(4*x/width)), min(3, int(4*y/height))
        (fit if (cellx+celly) % 2 == 0 else reserved).append((i, j))
    def operands(indices):
        return (np.array([pt[i] for i, j in indices], dtype=np.float64).reshape(-1, 2),
                np.array([ps[j] for i, j in indices], dtype=np.float64).reshape(-1, 2))
    seal = hashlib.sha256(json.dumps(dict(fit=fit, reserved=reserved), separators=(',', ':')).encode()).hexdigest()
    return operands(fit), operands(reserved), dict(fit_count=len(fit), reserved_count=len(reserved), guard_excluded=excluded, partition_index_sha256=seal)

def _analyze(np, H, fit, reserved, inliers, raw_t_size, raw_s_size, work_t_size, work_s_size,
             St, Ss, boxes, frame_verified):
    failures = []
    result = dict(registration_qualified=False, registration_only=True, failure_reasons=failures,
                  candidate_H_work=_matrix(np, H), candidate_H_raw=None, qualified_H_raw=None,
                  operation_support_qualified=False, ROI_status='NOT_SUPPLIED', ROI=[])
    ft, fs = fit
    rt, rs = reserved
    if H is None or not np.isfinite(H).all() or H.shape != (3, 3) or H[2, 2] == 0:
        failures.append('INVALID_HOMOGRAPHY')
        return result
    H = np.asarray(H, dtype=np.float64)/H[2, 2]
    try:
        inverse = np.linalg.inv(H)
        Hr = np.linalg.inv(Ss) @ H @ St
        assert Hr[2, 2] != 0
        Hr = Hr/Hr[2, 2]
    except (np.linalg.LinAlgError, AssertionError):
        failures.append('SINGULAR_HOMOGRAPHY')
        return result
    if not np.isfinite(inverse).all() or not np.isfinite(Hr).all():
        failures.append('NONFINITE_HOMOGRAPHY')
        return result
    result['candidate_H_work'], result['candidate_H_raw'] = H.tolist(), Hr.tolist()
    fraction = float(inliers.sum()/len(ft)) if len(ft) else 0.
    result['fit_inliers'] = int(inliers.sum())
    result['fit_inlier_fraction'] = fraction
    if len(ft) < MIN_MATCHES or len(rt) < MIN_MATCHES:
        failures.append('INSUFFICIENT_PARTITION_MATCHES')
    if inliers.sum() < MIN_MATCHES or fraction < FIT_FRACTION:
        failures.append('INSUFFICIENT_FIT_INLIERS')
    forward_points, _ = _project(np, H, rt)
    inverse_points, _ = _project(np, inverse, rs)
    forward_error = np.linalg.norm(forward_points-rs, axis=1)
    inverse_error = np.linalg.norm(inverse_points-rt, axis=1)
    error = np.maximum(forward_error, inverse_error)
    result['reserved_forward_scan_work'] = _summary(np, forward_error)
    result['reserved_inverse_template_work'] = _summary(np, inverse_error)
    combined = result['reserved_max_bidirectional_work'] = _summary(np, error)
    if combined['finite_count'] != len(rt) or combined['median'] is None or combined['median'] > ERROR_MEDIAN or combined['p90'] > ERROR_P90 or combined['fraction_at_most_5'] < ERROR_FRACTION:
        failures.append('HELDOUT_RESIDUAL')
    good = np.isfinite(error) & (error <= ERROR_P90)
    tw, th = work_t_size
    fit_spread = _spread(ft[inliers], tw, th)
    reserved_spread = _spread(rt[good], tw, th)
    result['fit_source_spread'] = fit_spread
    result['reserved_GOOD_source_spread'] = reserved_spread
    if fit_spread['occupied_quadrants'] < 3 or fit_spread['hull_area_fraction'] < FIT_HULL_FRACTION:
        failures.append('FIT_SPREAD')
    if reserved_spread['occupied_quadrants'] < 3 or reserved_spread['hull_area_fraction'] < RESERVED_HULL_FRACTION:
        failures.append('HELDOUT_SPREAD')
    for prefix, spread in [('fit', fit_spread), ('reserved_GOOD', reserved_spread)]:
        raw_hull, _ = _project(np, np.linalg.inv(St), spread['hull'])
        result[prefix+'_source_hull_raw'] = _matrix(np, raw_hull)
    width, height = raw_t_size
    corners = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]])
    projected, denominators = _project(np, Hr, corners)
    grid = np.array([(x, y) for y in np.linspace(0., height-1., 5) for x in np.linspace(0., width-1., 5)])
    _, grid_den = _project(np, Hr, grid)
    pole_ratio = float(np.min(np.abs(denominators))/np.max(np.abs(denominators))) if np.isfinite(denominators).all() and np.max(np.abs(denominators)) > 0 else 0.
    same_sign = bool(np.all(denominators > 0) or np.all(denominators < 0))
    if not same_sign or pole_ratio < POLE_RATIO or not np.isfinite(grid_den).all() or (grid_den == 0).any():
        failures.append('PROJECTIVE_POLE')
    result['corner_denominator_ratio'] = pole_ratio
    result['projected_center_corners_raw'] = _matrix(np, projected)
    corner_work, _ = _project(np, Ss, projected)
    result['projected_center_corners_scan_work'] = _matrix(np, corner_work)
    if not np.isfinite(corner_work).all():
        failures.append('NONFINITE_CORNERS')
    else:
        points = corner_work.tolist()
        turns = []
        for i in range(4):
            a, b, c = corner_work[i], corner_work[(i+1)%4], corner_work[(i+2)%4]
            turns.append(float((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])))
        area = _area(points)
        edges = np.linalg.norm(np.roll(corner_work, -1, axis=0)-corner_work, axis=1)
        sw, sh = work_s_size
        margin = CORNER_MARGIN*math.hypot(sw, sh)
        result['projected_quad_signed_area_scan_work'] = float(area)
        result['projected_edge_lengths_scan_work'] = edges.tolist()
        if area <= 0 or not all(v > 0 for v in turns):
            failures.append('ORIENTATION_OR_NONCONVEX_QUAD')
        if area < MIN_QUAD_AREA or (edges < MIN_EDGE).any():
            failures.append('TINY_OR_DEGENERATE_QUAD')
        if not (np.all(corner_work[:, 0] >= -margin) and np.all(corner_work[:, 0] <= sw-1+margin) and np.all(corner_work[:, 1] >= -margin) and np.all(corner_work[:, 1] <= sh-1+margin)):
            failures.append('CORNER_BOUNDS')
    result['registration_qualified'] = not failures
    if not failures:
        result['qualified_H_raw'] = Hr.tolist()
    if boxes:
        result['ROI_status'] = 'FRAME_UNQUALIFIED' if not frame_verified else 'REGISTRATION_UNQUALIFIED'
        if frame_verified and not failures:
            hull = reserved_spread['hull']
            good_points = rt[good]
            for bbox in boxes:
                x0, y0, x1, y1 = bbox
                probes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0+x1)/2, (y0+y1)/2]], dtype=np.float64)
                probes_work, _ = _project(np, St, probes)
                inside = [_inside(hull, p) for p in probes_work]
                distances = [float(np.linalg.norm(good_points-p, axis=1).min()) for p in probes_work] if len(good_points) else [None]*5
                supported = all(inside) and all(v is not None and v <= ROI_DISTANCE*math.hypot(tw, th) for v in distances)
                transported, _ = _project(np, Hr, probes[:4])
                result['ROI'].append(dict(bbox_raw=list(bbox), reserved_hull_membership=inside,
                    nearest_reserved_GOOD_work_distances=distances, supported=supported,
                    transported_quadrilateral_raw=_matrix(np, transported)))
            result['operation_support_qualified'] = all(v['supported'] for v in result['ROI'])
            result['ROI_status'] = 'GEOMETRIC_SUPPORT_ONLY' if result['operation_support_qualified'] else 'ROI_EXTRAPOLATION'
            result['registration_only'] = not result['operation_support_qualified']
    return result


@dataclass
class RegistrationResult:
    """A first-qualified registration, even if some fields are unobservable."""

    local_valid: bool
    reference_index: int | None
    homography_edges: np.ndarray | None
    quads: dict[str, np.ndarray]
    diagnostics: dict
    sampling_matrices: dict[str, np.ndarray] | None = None


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Image extraction needs OpenCV: install nrsw-document-forgery[vision].") from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build lacks SIFT; install opencv-python>=4.8.")
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    return cv2


def center_to_edge_homography(matrix):
    """Conjugate pixel-center registration to Pillow's pixel-edge frame."""
    h = np.asarray(matrix, dtype=np.float64)
    plus = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    minus = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])
    result = plus @ h @ minus
    if not np.isfinite(result).all() or result[2, 2] == 0:
        raise ValueError("Invalid center-coordinate homography")
    return result / result[2, 2]


def project_quad(matrix, quad):
    points, denominators = _project(np, np.asarray(matrix, np.float64), quad)
    if not np.isfinite(points).all() or np.any(denominators == 0):
        raise ValueError("Quadrilateral crosses a projective pole")
    return points


def validate_quad(quad, width, height):
    q = np.asarray(quad, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("Each field quad must contain four finite [x,y] points")
    if not (np.all(q >= 0) and np.all(q[:, 0] <= width) and np.all(q[:, 1] <= height)):
        raise ValueError("Field quad is outside its EXIF-corrected reference image")
    edges = np.roll(q, -1, axis=0) - q
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not np.all(turns > 0):
        raise ValueError("Field quad must be convex, ordered TL,TR,BR,BL without reflection")
    return q


def quad_sampling_matrix(quad, size=(224, 224)):
    """Solve a destination-edge to source-edge map in FP64 (no raster warp)."""
    width, height = size
    destination = np.array([[0., 0.], [width, 0.], [width, height], [0., height]])
    source = np.asarray(quad, np.float64)
    if source.shape != (4, 2) or not np.isfinite(source).all():
        raise ValueError("Invalid quadrilateral")
    left, top = source[0]
    right, bottom = source[2]
    if np.array_equal(source, [[left, top], [right, top], [right, bottom], [left, bottom]]):
        if right <= left or bottom <= top:
            raise ValueError("Degenerate rectangle")
        return np.array([[(right-left)/width, 0., left], [0., (bottom-top)/height, top], [0., 0., 1.]], np.float64)
    lhs, rhs = [], []
    for (x, y), (u, v) in zip(destination, source):
        lhs.extend(([x, y, 1., 0., 0., 0., -u*x, -u*y], [0., 0., 0., x, y, 1., -v*x, -v*y]))
        rhs.extend((u, v))
    return np.append(np.linalg.solve(lhs, rhs), 1.).reshape(3, 3)


def rectify_quad(image, quad, size=(224, 224)):
    """One Pillow bilinear perspective resample with white border."""
    return rectify_map(image, quad_sampling_matrix(quad, size), size)


def rectify_map(image, matrix, size=(224, 224)):
    """Sample the composed native edge map without solving it a second time."""
    from PIL import Image
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[2, 2] == 0:
        raise ValueError("Invalid perspective sampling matrix")
    matrix = matrix/matrix[2, 2]
    return np.asarray(image.transform(size, Image.Transform.PERSPECTIVE,
                      tuple(matrix.flat)[:8], resample=Image.Resampling.BILINEAR,
                      fillcolor=(255, 255, 255)), dtype=np.uint8)


class LayoutRegistrar:
    """Ordered genuine references in original EXIF-corrected RGB coordinates."""

    def __init__(self, layout, base_directory="."):
        if not isinstance(layout, dict):
            raise ValueError("layout JSON must be an object")
        from PIL import Image, ImageOps
        self.cv2 = _opencv()
        self.layout_id = layout.get("layout_id")
        self.field_order = layout.get("field_order", [])
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a nonempty string")
        if (not isinstance(self.field_order, list) or len(self.field_order) < 2
                or not all(isinstance(f, str) and f for f in self.field_order)
                or len(set(self.field_order)) != len(self.field_order)):
            raise ValueError("field_order must list at least two unique semantic fields")
        references = layout.get("references", [])
        if not isinstance(references, list) or not references:
            raise ValueError("Layout needs at least one annotated genuine reference image")
        self.references = []
        for reference in references:
            path = Path(base_directory) / reference["image"]
            payload = path.read_bytes()
            if len(payload) > MAX_ENCODED:
                raise ValueError(f"Reference exceeds {MAX_ENCODED} encoded bytes: {path}")
            with Image.open(path) as original:
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError(f"Reference exceeds {MAX_PIXELS} pixels: {path}")
                rgb = ImageOps.exif_transpose(original).convert("RGB")
            quads = reference.get("quads", {})
            if set(quads) != set(self.field_order):
                raise ValueError("Every reference must define exactly the layout field_order")
            quads = {f: validate_quad(quads[f], *rgb.size) for f in self.field_order}
            self.references.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                                        gray=np.asarray(rgb.convert("L")), quads=quads))

    def register(self, image, image_sha256=None):
        """Only image pixels and optional encoded-image SHA reach this method."""
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        attempts = []
        for index, reference in enumerate(self.references):
            if image_sha256 == reference["sha256"]:
                attempts.append(dict(reference_index=index, status="same_encoded_image_excluded"))
                continue
            result = _register(np, self.cv2, reference["gray"], gray, [], False, Counter(), lambda: None)
            attempts.append(dict(reference_index=index, status="qualified" if result["registration_qualified"] else "failed",
                                 reasons=result["failure_reasons"], matching=result["matching"], partition=result["partition"]))
            if not result["registration_qualified"]:
                continue
            # Selection precedes observability; never try another reference to rescue fields.
            h = center_to_edge_homography(result["qualified_H_raw"])
            quads, failures, sampling = {}, [], {}
            for field in self.field_order:
                try:
                    quad = project_quad(h, reference["quads"][field])
                    quads[field] = validate_quad(quad, *image.size)
                    sampling[field] = h @ quad_sampling_matrix(reference["quads"][field])
                except ValueError:
                    failures.append(field)
            return RegistrationResult(not failures, index, h, quads,
                                      dict(attempts=attempts, unavailable_fields=failures,
                                           status="local" if not failures else "global_fallback"), sampling)
        return RegistrationResult(False, None, None, {}, dict(attempts=attempts, status="global_fallback"))


def registration_contract():
    return dict(algorithm="native_reciprocal_sift_reserved_checkerboard_ransac_v1",
                reference_policy="first_globally_qualified_before_field_observability",
                exact_encoded_reference_exclusion=True, exif="transpose_before_RGB_and_geometry",
                max_edge=MAX_EDGE, max_features=MAX_FEATURES, ratio=RATIO, dedup=DEDUP,
                checkerboard=[4, 4], checkerboard_guard=GRID_GUARD, min_matches=MIN_MATCHES,
                ransac_threshold=RANSAC_THRESHOLD, ransac_iterations=RANSAC_ITERATIONS,
                ransac_confidence=RANSAC_CONFIDENCE, ransac_seed=0,
                reserved_error_median=ERROR_MEDIAN, reserved_error_p90=ERROR_P90,
                reserved_error_fraction=ERROR_FRACTION, min_inlier_fraction=FIT_FRACTION,
                fit_hull_fraction=FIT_HULL_FRACTION, reserved_hull_fraction=RESERVED_HULL_FRACTION,
                corner_margin=CORNER_MARGIN, min_edge=MIN_EDGE, min_quad_area=MIN_QUAD_AREA,
                projective_pole_ratio=POLE_RATIO, local_policy="all_fields_or_global_fallback",
                coordinate_frame="EXIF_corrected_RGB_pixel_edges", center_edge_conjugation=True,
                crop=[224, 224], resample="Pillow_PERSPECTIVE_BILINEAR_white_border")

def _register(np, cv2, template_gray, scan_gray, boxes, frame_verified, work, guard):
    raw_t_size = template_gray.shape[1], template_gray.shape[0]
    raw_s_size = scan_gray.shape[1], scan_gray.shape[0]
    template_work, St = _work_image(np, cv2, template_gray, work, guard)
    scan_work, Ss = _work_image(np, cv2, scan_gray, work, guard)
    # The function retains only one pair. No global image/descriptor cache exists.
    pt, dt = _features(np, cv2, template_work, work, guard)
    ps, ds = _features(np, cv2, scan_work, work, guard)
    matches, matching = _matches(np, cv2, pt, dt, ps, ds, work, guard)
    fit, reserved, partition = _partition(np, matches, pt, ps, *template_work.shape[::-1])
    H, inliers = None, np.zeros(len(fit[0]), dtype=bool)
    if len(fit[0]) >= MIN_MATCHES and len(reserved[0]) >= MIN_MATCHES:
        guard()
        cv2.setRNGSeed(0)
        work['homography_fit_calls'] += 1
        H, mask = cv2.findHomography(fit[0], fit[1], cv2.RANSAC, RANSAC_THRESHOLD,
                                   maxIters=RANSAC_ITERATIONS, confidence=RANSAC_CONFIDENCE)
        if mask is not None:
            inliers = mask.reshape(-1).astype(bool)
        guard()
    result = _analyze(np, H, fit, reserved, inliers, raw_t_size, raw_s_size,
                      template_work.shape[::-1], scan_work.shape[::-1], St, Ss, boxes, frame_verified)
    result.update(template_raw_size=list(raw_t_size), scan_raw_size=list(raw_s_size),
                  template_work_size=list(template_work.shape[::-1]), scan_work_size=list(scan_work.shape[::-1]),
                  S_template=St.tolist(), S_scan=Ss.tolist(), matching=matching, partition=partition,
                  template_keypoints=len(pt), scan_keypoints=len(ps), H_direction='template_raw_centers_to_scan_raw_centers',
                  pristine_pair_qualified=False, pixel_tamper_GT_qualified=False)
    if H is None:
        result['failure_reasons'].insert(0, 'INSUFFICIENT_MATCHES' if min(len(fit[0]), len(reserved[0])) < MIN_MATCHES else 'HOMOGRAPHY_FIT_FAILED')
    guard()
    return result
