"use strict";

const byId = (id) => document.getElementById(id);
const sourceSelect = byId("source-select");
const documentSelect = byId("example-select");
const thresholdSelect = byId("threshold-select");
const traceFile = byId("trace-file");
const restoreButton = byId("restore");
const uploadForm = byId("upload-form");
const uploadFile = byId("upload-file");
const uploadSubmit = byId("upload-submit");
const numberFormat = new Intl.NumberFormat(locale, { maximumSignificantDigits: 8 });
const integerFormat = new Intl.NumberFormat(locale);
const percentFormat = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 4 });
const weightFormat = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 2 });

const fieldLabels = { name: "Name", number: "Permit number", valid_until: "Valid until" };
const referenceValues = { name: "Avery Rowan", number: "NB 094 381", valid_until: "18 NOV 2028" };
const editedValues = { name: "Avery Robin", number: "NB 094 881", valid_until: "18 NOV 2030" };
const sourceNames = { specimen: "Document images", synthetic: "Simulated features", custom: "Local model" };
const slovakLayouts = new Set(["idspace-svk", "IDSpace_SVK_R29"]);
const builtIn = new Map();

let defaultSource;
let trace;
let previewUrl;
let pendingUpload;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function fieldLabel(field) {
  if (trace.dataset === "synthetic" && Object.hasOwn(fieldLabels, field)) return t(fieldLabels[field]);
  if (locale === "zh-CN" && trace.field_labels_zh && Object.hasOwn(trace.field_labels_zh, field)) return trace.field_labels_zh[field];
  if (trace.field_labels && Object.hasOwn(trace.field_labels, field)) return trace.field_labels[field];
  const specimen = builtIn.get("specimen");
  if (specimen && slovakLayouts.has(trace.layout)) {
    const label = (locale === "zh-CN" ? specimen.field_labels_zh : specimen.field_labels)?.[field];
    if (label) return label;
  }
  return field.replaceAll("_", " ");
}

function localized(item, name) {
  const translated = item[`${name}_zh`];
  if (locale === "zh-CN" && typeof translated === "string" && translated.trim()) return translated;
  const value = item[name];
  return typeof value === "string" && value.trim() ? t(value) : undefined;
}

function option(value, label) {
  const node = element("option", label);
  node.value = value;
  return node;
}

function verify(condition, message) {
  if (!condition) throw new Error(t(message));
}

function validate(data) {
  verify(data && typeof data === "object", "Open a prediction file exported by NRSW.");
  verify(Object.hasOwn(sourceNames, data.dataset), "The prediction file has an unknown data source.");
  verify(
    Array.isArray(data.fields) && data.fields.length > 0 &&
      data.fields.every((name) => typeof name === "string" && name.trim()) &&
      new Set(data.fields).size === data.fields.length,
    "Field names in the prediction file must be unique."
  );
  for (const labels of [data.field_labels, data.field_labels_zh]) {
    if (labels === undefined) continue;
    verify(
      labels && typeof labels === "object" &&
      Object.entries(labels).every(([field, label]) => data.fields.includes(field) && typeof label === "string" && label.trim()),
      "A field label must belong to this layout."
    );
  }
  verify(Array.isArray(data.examples) && data.examples.length > 0, "The file contains no document predictions.");
  const ids = new Set();
  for (const sample of data.examples) {
    verify(sample && typeof sample.id === "string" && sample.id.trim() && !ids.has(sample.id), "Each prediction needs a unique document ID.");
    ids.add(sample.id);
    verify(Number.isFinite(sample.score), "Each prediction needs a finite score.");
    for (const key of ["weights", "energies"]) {
      const values = sample[key];
      if (values === null) continue;
      verify(Array.isArray(values) && values.length === data.fields.length, "Each field needs one weight and one energy value.");
      verify(
        values.every((value) => value === null ||
          (Number.isFinite(value) && value >= 0 && (key !== "weights" || value <= 1 + 1e-9))),
        "Weights must be between 0 and 1, and energies must be nonnegative. Use null for a missing field."
      );
    }
    verify(
      sample.edited_fields === undefined ||
        (Array.isArray(sample.edited_fields) && sample.edited_fields.every((field) => data.fields.includes(field))),
      "An edited field named in the file does not belong to this layout."
    );
    if (data.dataset === "specimen") {
      verify(
        typeof sample.image === "string" && /^\.\/specimens\/[\w-]+\.(?:png|jpe?g|webp)$/i.test(sample.image) &&
        ["genuine", "forged"].includes(sample.label),
        "Each document needs a bundled image and a dataset label."
      );
    }
    if (sample.boxes !== undefined) {
      verify(
        sample.boxes && typeof sample.boxes === "object" &&
        Object.entries(sample.boxes).every(([field, box]) =>
          data.fields.includes(field) && Array.isArray(box) && box.length === 4 &&
          box.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) &&
          box[0] < box[2] && box[1] < box[3]),
        "Each field outline must lie inside its document image."
      );
    }
  }
  verify(Array.isArray(data.thresholds), "The prediction file is missing its list of thresholds.");
  const budgets = new Set();
  for (const cut of data.thresholds) {
    verify(cut && Number.isFinite(cut.fpr) && cut.fpr > 0 && cut.fpr < 1 && !budgets.has(cut.fpr), "Each threshold needs a different target false-positive rate between 0 and 1.");
    verify(cut.threshold === null || Number.isFinite(cut.threshold), "Each threshold must be a finite score or null.");
    budgets.add(cut.fpr);
  }
  return data;
}

function showStatus(message, isError = false) {
  const status = byId("status");
  status.hidden = !message;
  status.textContent = t(message);
  status.classList.toggle("error", isError);
}

function titleOf(sample) {
  return localized(sample, "title") ?? sample.id;
}

function currentSample() {
  return trace.examples[Number(documentSelect.value)];
}

function currentCut() {
  return thresholdSelect.value === "" ? undefined : trace.thresholds[Number(thresholdSelect.value)];
}

function showFitSummary() {
  const fields = [
    ["genuine", "genuine-count"],
    ["forged", "forged-count"],
    ["calibration", "calibration-count"]
  ];
  let hasCount = false;
  for (const [field, id] of fields) {
    const count = trace.training?.[field];
    const known = Number.isInteger(count) && count >= 0;
    byId(id).textContent = known ? integerFormat.format(count) : "—";
    hasCount ||= known;
  }
  byId("fit-summary").hidden = !hasCount;
}

function useTrace(data, imported = false, imageUrl) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = imageUrl;
  trace = data;
  restoreButton.hidden = !(defaultSource && imported);
  byId("source-control").hidden = imported || builtIn.size < 2;
  if (!imported) sourceSelect.value = data.dataset;
  byId("source-label").textContent = t(imageUrl ? "Uploaded document" : imported ? "Opened prediction file" : sourceNames[data.dataset]);
  byId("dataset-scope").textContent = t(imageUrl ? "NRSW measured this image. Select a target false-positive rate to see its decision." : {
    specimen: "Choose a document image to inspect its saved NRSW prediction.",
    synthetic: "These predictions use simulated features. The illustrations show the fields they represent.",
    custom: "These scores come from a locally trained detector."
  }[data.dataset]);
  documentSelect.replaceChildren(...data.examples.map((sample, i) => option(i, titleOf(sample))));
  const savedSample = data.examples.findIndex((sample) => sample.id === data.selected_id);
  const firstEdit = data.examples.findIndex((sample) => data.dataset === "synthetic" ? sample.id === "number-edit" : sample.label === "forged");
  documentSelect.value = [savedSample, firstEdit, 0].find((index) => index >= 0);
  thresholdSelect.replaceChildren(...data.thresholds.map((cut, i) => {
    const suffix = cut.threshold === null ? t(" (unavailable)") : "";
    return option(i, percentFormat.format(cut.fpr) + suffix);
  }));
  thresholdSelect.disabled = data.thresholds.length === 0;
  if (thresholdSelect.disabled) thresholdSelect.append(option("", t("Score only")));
  const saved = data.thresholds.findIndex((cut) => cut.fpr === data.selected_fpr);
  const primary = data.thresholds.findIndex((cut) => cut.fpr === 0.01 && cut.threshold !== null);
  const available = data.thresholds.findIndex((cut) => cut.threshold !== null);
  const preferred = [saved, primary, available].find((index) => index >= 0);
  if (preferred !== undefined) thresholdSelect.value = preferred;
  showFitSummary();
  showStatus("");
  byId("inspector").hidden = false;
  showDocument();
}

function permit(card, sample) {
  const heading = element("div", undefined, "permit-heading");
  const seal = element("span", "N", "permit-seal");
  seal.setAttribute("aria-hidden", "true");
  const title = element("span", "Northbank Archive", "permit-title");
  title.append(element("small", t("Reader permit · Fictional")));
  heading.append(seal, title);
  const portrait = element("div", undefined, "permit-photo");
  portrait.setAttribute("aria-hidden", "true");
  const fields = element("dl", undefined, "permit-fields");
  trace.fields.forEach((field, i) => {
    const edited = sample?.edited_fields?.includes(field);
    const missing = Boolean(sample && sample.energies?.[i] == null && sample.weights?.[i] == null);
    const values = edited ? editedValues : referenceValues;
    const text = Object.hasOwn(values, field) ? values[field] : t("Example value");
    const item = element("div", undefined, "permit-field");
    item.dataset.field = field;
    item.classList.toggle("edited", Boolean(edited));
    item.classList.toggle("missing", missing);
    item.append(element("dt", fieldLabel(field)), element("dd", missing ? t("Unavailable") : text));
    fields.append(item);
  });
  card.replaceChildren(heading, portrait, fields);
}

function selectField(field) {
  for (const item of document.querySelectorAll(".permit-field")) {
    item.classList.toggle("active", item.dataset.field === field);
  }
  for (const box of byId("specimen-documents").querySelectorAll(".field-box")) {
    box.hidden = box.dataset.field !== field;
  }
  for (const row of byId("field-rows").rows) {
    const active = row.dataset.field === field;
    row.classList.toggle("selected", active);
    row.querySelector("button")?.setAttribute("aria-pressed", String(active));
  }
}

function fieldRow(field, weight, energy, interactive) {
  const row = element("tr");
  row.dataset.field = field;
  const heading = element("th");
  heading.scope = "row";
  if (interactive) {
    const button = element("button", fieldLabel(field), "field-button");
    button.type = "button";
    button.addEventListener("click", () => selectField(field));
    heading.append(button);
  } else {
    heading.textContent = fieldLabel(field);
  }
  const weightCell = element("td");
  if (weight === null) {
    weightCell.append(element("span", t("Unavailable")));
  } else {
    const value = element("div", undefined, "weight-cell");
    const track = element("span", undefined, "weight-track");
    track.setAttribute("aria-hidden", "true");
    const fill = element("span");
    fill.style.width = String(Math.min(weight * 100, 100)) + "%";
    track.append(fill);
    value.append(track, element("span", weightFormat.format(weight), "num"));
    weightCell.append(value);
  }
  const energyCell = element("td", energy === null ? "—" : numberFormat.format(energy), "num");
  row.append(heading, weightCell, energyCell);
  return row;
}

function specimenImage(sample, caption, companionIndex) {
  const frame = element("div", undefined, "specimen-frame");
  const image = element("img");
  image.src = sample.image;
  image.alt = titleOf(sample);
  image.addEventListener("error", () => {
    frame.replaceChildren(element("p", t("Document image unavailable."), "image-message"));
  });
  frame.append(image);
  for (const [field, [x0, y0, x1, y1]] of Object.entries(sample.boxes ?? {})) {
    const box = element("span", undefined, "field-box");
    box.hidden = true;
    box.dataset.field = field;
    box.setAttribute("aria-hidden", "true");
    Object.assign(box.style, {
      left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`
    });
    frame.append(box);
  }
  const figure = element("figure");
  const label = element("figcaption", t(caption));
  if (companionIndex >= 0) {
    const inspect = element("button", t("Inspect"), "quiet-button");
    inspect.type = "button";
    inspect.setAttribute("aria-label", t("Inspect {document}").replace("{document}", titleOf(sample)));
    inspect.addEventListener("click", () => {
      documentSelect.value = companionIndex;
      showDocument();
      documentSelect.focus();
    });
    label.append(inspect);
  }
  figure.append(frame, label);
  return figure;
}

function httpsLink(label, url) {
  try {
    const linkUrl = new URL(url);
    if (linkUrl.protocol === "https:") {
      const link = element("a", label);
      link.href = linkUrl.href;
      return link;
    }
  } catch {}
  return document.createTextNode(label);
}

function showSpecimen(sample) {
  const images = byId("specimen-documents");
  images.replaceChildren(specimenImage(sample, "Scored document"));
  const match = sample.pair_id === undefined ? -1 : trace.examples.findIndex((item) =>
    item.pair_id === sample.pair_id && item.label !== sample.label);
  images.classList.toggle("paired", match >= 0);
  if (match >= 0) {
    const counterpart = trace.examples[match];
    images.append(specimenImage(counterpart, counterpart.label === "genuine" ? "Matching unaltered image" : "Matching altered image", match));
  }

  const facts = byId("specimen-facts");
  facts.replaceChildren(element("dt", t("Dataset label")), element("dd", t(sample.label === "genuine" ? "Unaltered" : "Altered")));
  if (sample.edited_fields?.length) {
    facts.append(element("dt", t("Annotated edit")), element("dd", sample.edited_fields.map(fieldLabel).join(locale === "zh-CN" ? "、" : ", ")));
  }

  const credit = byId("document-credit");
  const source = trace.source;
  credit.hidden = !source || typeof source.name !== "string";
  if (!credit.hidden) {
    credit.replaceChildren(httpsLink(source.name, source.url));
    if (typeof source.license === "string") credit.append(" · ", httpsLink(source.license, source.license_url));
  }
}

function showDocument() {
  const sample = currentSample();
  const synthetic = trace.dataset === "synthetic";
  const specimen = trace.dataset === "specimen";
  const hasImage = specimen || previewUrl;
  const label = byId("image-scope");
  label.hidden = !synthetic && !hasImage;
  const layoutName = slovakLayouts.has(trace.layout) ? "Slovak identity-card layout" : "Source images";
  label.textContent = synthetic ? t("Illustrations") : localized(trace, "layout_name") ?? t(layoutName);
  byId("illustrations").hidden = !synthetic;
  byId("specimen-documents").hidden = !hasImage;
  if (!hasImage) byId("specimen-documents").replaceChildren();
  byId("specimen-facts").hidden = !specimen;
  byId("custom-document").hidden = synthetic || hasImage;
  byId("document-credit").hidden = true;
  if (synthetic) {
    const reference = byId("reference-card");
    const candidate = byId("candidate-card");
    reference.setAttribute("aria-label", t("Fictional reference card"));
    candidate.setAttribute("aria-label", t("Fictional example card"));
    permit(reference);
    permit(candidate, sample);
    byId("candidate-caption").textContent = titleOf(sample);
  } else if (specimen) {
    showSpecimen(sample);
  } else if (previewUrl) {
    const images = byId("specimen-documents");
    images.classList.remove("paired");
    images.replaceChildren(specimenImage({ ...sample, image: previewUrl }, "Uploaded document"));
  } else {
    const title = element("strong", titleOf(sample));
    byId("custom-document").replaceChildren(title);
    if (titleOf(sample) !== sample.id) {
      byId("custom-document").append(element("span", sample.id, "muted"));
    }
    if (typeof trace.layout === "string" && trace.layout) {
      byId("custom-document").append(element("span", t("Registered layout {layout}").replace("{layout}", trace.layout), "muted"));
    }
  }

  const note = byId("example-note");
  note.textContent = (specimen ? localized(trace, "note") : localized(sample, "note")) ?? "";
  note.hidden = !note.textContent;
  const outlined = new Set(Object.keys(sample.boxes ?? {}));
  const interactive = (field) => synthetic || Boolean(hasImage && outlined.has(field));
  byId("evidence-help").textContent = t("Energy measures how unusual a field looks along the learned directions. Its weight determines its share of the combined local features.") +
    (synthetic || (hasImage && outlined.size) ? ` ${t("Select an underlined field to see where it was measured.")}` : "");
  const weights = sample.weights ?? trace.fields.map(() => null);
  byId("field-rows").replaceChildren(...trace.fields.map((field, i) => fieldRow(field, weights[i], sample.energies?.[i] ?? null, interactive(field))));
  const largestWeight = weights.reduce((best, weight, i) => interactive(trace.fields[i]) && (weight ?? -1) > (weights[best] ?? -1) ? i : best, -1);
  const firstEdited = synthetic ? sample.edited_fields?.[0] : undefined;
  selectField(firstEdited ?? trace.fields[largestWeight]);
  showDecision();
}

function showDecision() {
  const sample = currentSample();
  const cut = currentCut();
  const threshold = cut?.threshold;
  const hasCut = Number.isFinite(threshold);
  const review = hasCut && sample.score > threshold;
  const decision = byId("decision-value");
  decision.textContent = t(hasCut ? (review ? "Flagged for review" : "Not flagged") : "Score only");
  decision.closest(".decision").classList.toggle("review", review);
  byId("score-value").textContent = numberFormat.format(sample.score);
  byId("threshold-value").textContent = hasCut ? numberFormat.format(threshold) : t("Unavailable");
}

function downloadPrediction() {
  const sample = currentSample();
  const json = JSON.stringify({ ...trace, examples: [sample], selected_id: sample.id, selected_fpr: currentCut()?.fpr }, null, 2);
  const file = new Blob([json + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = element("a");
  link.href = url;
  link.download = "nrsw-prediction.json";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function uploadStatus(message, isError = false) {
  const status = byId("upload-status");
  status.textContent = t(message);
  status.hidden = !message;
  status.classList.toggle("error", isError);
}

function isLoopback(host) {
  return ["localhost", "127.0.0.1", "[::1]"].includes(host);
}

function predictionApi() {
  const configured = document.querySelector('meta[name="nrsw-prediction-api"]').content.trim();
  if (!configured && !isLoopback(location.hostname)) return;
  try {
    const api = new URL(configured || ".", location.href);
    const loopback = api.origin === location.origin && api.protocol === "http:" && isLoopback(api.hostname);
    if ((!loopback && api.protocol !== "https:") || api.username || api.password || api.search || api.hash) return;
    if (!api.pathname.endsWith("/")) api.pathname += "/";
    return api;
  } catch {
    return;
  }
}

function updateUploadForm() {
  byId("upload-name").textContent = uploadFile.files[0]?.name ?? t("No image selected");
  uploadSubmit.disabled = !uploadFile.files.length || Boolean(pendingUpload);
  uploadForm.setAttribute("aria-busy", String(Boolean(pendingUpload)));
}

uploadFile.addEventListener("change", () => {
  pendingUpload?.abort();
  pendingUpload = undefined;
  uploadStatus("");
  updateUploadForm();
});

const uploadTypes = { "image/jpeg": "JPEG", "image/png": "PNG" };
const uploadRequest = { credentials: "omit", redirect: "error", referrerPolicy: "no-referrer" };
const predictionErrors = {
  invalid_image: "The image could not be read. Choose a clear JPEG or PNG.",
  alignment_failed: "The card could not be aligned. Use a clear, complete image of the supported layout.",
  busy: "The detector is busy. Try again shortly.",
  invalid_fpr: "The selected false-positive rate is unavailable.",
  unavailable: "Live detection is temporarily unavailable.",
  too_large: "The image exceeds the upload limit.",
  unsupported_type: "Choose one of the supported image formats.",
  server_error: "Detection failed. Try again shortly."
};

async function imageType(file) {
  const bytes = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  const png = [137, 80, 78, 71, 13, 10, 26, 10];
  if (png.every((byte, i) => bytes[i] === byte)) return "image/png";
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return "image/jpeg";
  return "";
}

async function submitImage(api, status) {
  const [file] = uploadFile.files;
  if (!file || pendingUpload) return;
  const controller = new AbortController();
  pendingUpload = controller;
  updateUploadForm();
  uploadStatus("Measuring the image…");
  try {
    verify(file.size <= status.max_bytes, predictionErrors.too_large);
    const type = await imageType(file);
    if (pendingUpload !== controller) return;
    verify(status.supported_types.includes(type), predictionErrors.unsupported_type);
    const currentFpr = trace && currentCut()?.fpr;
    const fpr = status.fprs.find((value) => value === currentFpr) ?? status.fprs[0];
    const url = new URL("api/predict", api);
    url.searchParams.set("fpr", fpr);
    const response = await fetch(url, { ...uploadRequest, method: "POST", headers: { "Content-Type": type }, body: file, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (pendingUpload !== controller) return;
    const message = Object.hasOwn(predictionErrors, data?.error) ? predictionErrors[data.error] : predictionErrors.server_error;
    verify(response.ok, message);
    verify(data?.dataset === "custom" && data.layout === status.layout && data.examples?.length === 1 &&
      Array.isArray(data.examples[0].weights), "The server did not return a compatible prediction.");
    const prediction = validate(data);
    verify(prediction.thresholds.some((cut) => cut.fpr === fpr && Number.isFinite(cut.threshold)),
      "The server did not return a compatible prediction.");
    useTrace({ ...prediction, selected_fpr: fpr }, true, URL.createObjectURL(file.slice(0, file.size, type)));
    uploadFile.value = "";
    uploadStatus("Detection complete.");
    byId("demo-title").focus();
  } catch (error) {
    if (pendingUpload === controller) uploadStatus(error instanceof TypeError ? "Could not reach the detection server." : error.message, true);
  } finally {
    if (pendingUpload === controller) {
      pendingUpload = undefined;
      updateUploadForm();
    }
  }
}

async function enableUpload() {
  const api = predictionApi();
  if (!api) return;
  try {
    const response = await fetch(new URL("api/status", api), { ...uploadRequest, cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!response.ok) return;
    const status = await response.json();
    const types = status?.supported_types;
    if (typeof status?.layout !== "string" || !Number.isInteger(status.max_bytes) || status.max_bytes <= 0 ||
        !Array.isArray(types) || !types.length || !types.every((type) => Object.hasOwn(uploadTypes, type)) ||
        !Array.isArray(status.fprs) || !status.fprs.length || !status.fprs.every((fpr) => Number.isFinite(fpr) && fpr > 0 && fpr < 1)) return;
    uploadFile.accept = types.join(",");
    const size = new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(status.max_bytes / (1024 * 1024));
    byId("upload-size").textContent = t("{formats} · up to {limit} MiB").replace("{formats}", types.map((type) => uploadTypes[type]).join(" / ")).replace("{limit}", size);
    if (!slovakLayouts.has(status.layout)) byId("upload-layout").textContent = t("Registered layout {layout}").replace("{layout}", status.layout);
    byId("upload-remote").hidden = isLoopback(api.hostname);
    byId("upload-unavailable").hidden = true;
    uploadForm.hidden = false;
    uploadForm.addEventListener("submit", (event) => {
      event.preventDefault();
      submitImage(api, status);
    });
  } catch {
    // Keep the local setup guide visible if the inference server is unavailable.
  }
}

traceFile.addEventListener("change", async () => {
  const [file] = traceFile.files;
  if (!file) return;
  try {
    verify(file.size <= 10 * 1024 * 1024, "Open a prediction file no larger than 10 MB.");
    useTrace(validate(JSON.parse(await file.text())), true);
  } catch (error) {
    const message = error instanceof SyntaxError ? "The file is not valid JSON." : error.message;
    showStatus(message, true);
  } finally {
    traceFile.value = "";
  }
});
restoreButton.addEventListener("click", () => useTrace(builtIn.get(defaultSource)));
sourceSelect.addEventListener("change", () => useTrace(builtIn.get(sourceSelect.value)));
documentSelect.addEventListener("change", showDocument);
thresholdSelect.addEventListener("change", showDecision);
byId("download").addEventListener("click", downloadPrediction);

function loadExamples() {
  for (const data of [window.NRSW_SPECIMENS, window.NRSW_EXAMPLES]) {
    if (!data) continue;
    try {
      const valid = validate(data);
      builtIn.set(valid.dataset, valid);
    } catch {
      // Keep the other example source available if an asset is incomplete.
    }
  }
  defaultSource = ["custom", "specimen", "synthetic"].find((source) => builtIn.has(source));
  if (!defaultSource) {
    showStatus("Could not load examples. Open a prediction file or start the demo with nrsw serve.", true);
    return;
  }
  sourceSelect.replaceChildren(...Array.from(builtIn.keys(), (source) => option(source, t(sourceNames[source]))));
  useTrace(builtIn.get(defaultSource));
}

loadExamples();
enableUpload();
