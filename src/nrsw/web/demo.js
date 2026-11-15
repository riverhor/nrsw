"use strict";

const byId = (id) => document.getElementById(id);
const documentSelect = byId("example-select");
const thresholdSelect = byId("threshold-select");
const traceFile = byId("trace-file");
const restoreButton = byId("restore");
const numberFormat = new Intl.NumberFormat("en", { maximumSignificantDigits: 8 });
const integerFormat = new Intl.NumberFormat("en");
const percentFormat = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 4 });
const weightFormat = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 2 });

const fieldLabels = { name: "Name", number: "Permit number", valid_until: "Valid until" };
const referenceValues = { name: "Avery Rowan", number: "NB 094 381", valid_until: "18 NOV 2028" };
const editedValues = { name: "Avery Robin", number: "NB 094 881", valid_until: "18 NOV 2030" };

let examples;
let trace;
let usingImport = false;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function fieldLabel(field) {
  if (trace.dataset === "synthetic" && Object.hasOwn(fieldLabels, field)) return fieldLabels[field];
  return field.replaceAll("_", " ");
}

function option(value, label) {
  const node = element("option", label);
  node.value = value;
  return node;
}

function verify(condition, message) {
  if (!condition) throw new Error(message);
}

function validate(data) {
  verify(data && typeof data === "object", "Expected a trace JSON object.");
  verify(data.dataset === "synthetic" || data.dataset === "custom", "The trace must identify its dataset as synthetic or custom.");
  verify(
    Array.isArray(data.fields) && data.fields.length > 0 &&
      data.fields.every((name) => typeof name === "string" && name.trim()) &&
      new Set(data.fields).size === data.fields.length,
    "The trace must include unique field names."
  );
  verify(Array.isArray(data.examples) && data.examples.length > 0, "The trace contains no document predictions.");
  const ids = new Set();
  for (const sample of data.examples) {
    verify(sample && typeof sample.id === "string" && sample.id.trim() && !ids.has(sample.id), "Each prediction needs a unique document ID.");
    ids.add(sample.id);
    verify(Number.isFinite(sample.score), "Each prediction needs a finite score.");
    for (const key of ["weights", "energies"]) {
      const values = sample[key];
      if (values === null) continue;
      verify(Array.isArray(values) && values.length === data.fields.length, "The trace needs one weight and energy per field.");
      verify(
        values.every((value) => value === null ||
          (Number.isFinite(value) && value >= 0 && (key !== "weights" || value <= 1 + 1e-9))),
        "Weights must be between 0 and 1, and energies must be nonnegative. Use null for a missing field."
      );
    }
    verify(
      sample.edited_fields === undefined ||
        (Array.isArray(sample.edited_fields) && sample.edited_fields.every((field) => data.fields.includes(field))),
      "The trace names an edited field outside the layout."
    );
  }
  verify(Array.isArray(data.thresholds), "The trace must include a thresholds array.");
  const budgets = new Set();
  for (const cut of data.thresholds) {
    verify(cut && Number.isFinite(cut.fpr) && cut.fpr > 0 && cut.fpr < 1 && !budgets.has(cut.fpr), "Each threshold needs a unique false-positive budget between 0 and 1.");
    verify(cut.threshold === null || Number.isFinite(cut.threshold), "Each threshold must be a finite score or null.");
    budgets.add(cut.fpr);
  }
  return data;
}

function showStatus(message, isError = false) {
  const status = byId("status");
  status.hidden = !message;
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function titleOf(sample) {
  return typeof sample.title === "string" && sample.title.trim() ? sample.title : sample.id;
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

function useTrace(data, imported = false) {
  trace = data;
  usingImport = imported;
  restoreButton.hidden = !(examples && usingImport);
  byId("source-label").textContent = data.dataset === "synthetic" ? "Synthetic examples" : "Model predictions";
  documentSelect.replaceChildren(...data.examples.map((sample, i) => option(i, titleOf(sample))));
  const firstEdit = data.dataset === "synthetic" ? data.examples.findIndex((sample) => sample.id === "number-edit") : -1;
  documentSelect.value = firstEdit < 0 ? 0 : firstEdit;
  thresholdSelect.replaceChildren(...data.thresholds.map((cut, i) => {
    const suffix = cut.threshold === null ? " (unavailable)" : "";
    return option(i, percentFormat.format(cut.fpr) + suffix);
  }));
  thresholdSelect.disabled = data.thresholds.length === 0;
  if (thresholdSelect.disabled) thresholdSelect.append(option("", "Score only"));
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
  title.append(element("small", "Reader permit · Fictional"));
  heading.append(seal, title);
  const portrait = element("div", undefined, "permit-photo");
  portrait.setAttribute("aria-hidden", "true");
  const fields = element("dl", undefined, "permit-fields");
  trace.fields.forEach((field, i) => {
    const edited = sample?.edited_fields?.includes(field);
    const missing = Boolean(sample && sample.energies?.[i] == null && sample.weights?.[i] == null);
    const values = edited ? editedValues : referenceValues;
    const text = Object.hasOwn(values, field) ? values[field] : "Example value";
    const item = element("div", undefined, "permit-field");
    item.dataset.field = field;
    item.classList.toggle("edited", Boolean(edited));
    item.classList.toggle("missing", missing);
    item.append(element("dt", fieldLabel(field)), element("dd", missing ? "Unavailable" : text));
    fields.append(item);
  });
  card.replaceChildren(heading, portrait, fields);
}

function selectField(field) {
  for (const item of document.querySelectorAll(".permit-field")) {
    item.classList.toggle("active", item.dataset.field === field);
  }
  for (const row of byId("field-rows").rows) {
    const active = row.dataset.field === field;
    row.classList.toggle("selected", active);
    row.querySelector("button")?.setAttribute("aria-pressed", String(active));
  }
}

function fieldRow(field, weight, energy) {
  const row = element("tr");
  row.dataset.field = field;
  const heading = element("th");
  heading.scope = "row";
  if (trace.dataset === "synthetic") {
    const button = element("button", fieldLabel(field), "field-button");
    button.type = "button";
    button.addEventListener("click", () => selectField(field));
    heading.append(button);
  } else {
    heading.textContent = fieldLabel(field);
  }
  const weightCell = element("td");
  if (weight === null) {
    weightCell.append(element("span", "Unavailable"));
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

function showDocument() {
  const sample = currentSample();
  const synthetic = trace.dataset === "synthetic";
  byId("image-scope").hidden = !synthetic;
  byId("illustrations").hidden = !synthetic;
  byId("custom-document").hidden = synthetic;
  if (synthetic) {
    permit(byId("reference-card"));
    permit(byId("candidate-card"), sample);
    byId("candidate-caption").textContent = titleOf(sample);
  } else {
    const title = element("strong", titleOf(sample));
    byId("custom-document").replaceChildren(title);
    if (titleOf(sample) !== sample.id) {
      byId("custom-document").append(element("span", sample.id, "muted"));
    }
    if (typeof trace.layout === "string" && trace.layout) {
      byId("custom-document").append(element("span", "Registered layout " + trace.layout, "muted"));
    }
  }

  const note = typeof sample.note === "string" ? sample.note : "";
  const sourceNote = synthetic ? "The illustrations show fields represented by the synthetic features." : "";
  byId("example-note").textContent = [note, sourceNote].filter(Boolean).join(" ");
  byId("evidence-help").textContent = "Energy measures a field’s departure along learned directions. The weight is its share of local pooling." +
    (synthetic ? " Select a field to locate it on the cards." : "");
  const weights = sample.weights ?? trace.fields.map(() => null);
  byId("field-rows").replaceChildren(...trace.fields.map((field, i) => fieldRow(field, weights[i], sample.energies?.[i] ?? null)));
  const firstEdited = sample.edited_fields?.[0];
  const largestWeight = weights.reduce((best, weight, i) => (weight ?? -1) > (weights[best] ?? -1) ? i : best, 0);
  selectField(synthetic ? firstEdited ?? (weights[largestWeight] === null ? undefined : trace.fields[largestWeight]) : undefined);
  showDecision();
}

function showDecision() {
  const sample = currentSample();
  const cut = currentCut();
  const threshold = cut?.threshold;
  const hasCut = Number.isFinite(threshold);
  const review = hasCut && sample.score > threshold;
  const decision = byId("decision-value");
  decision.textContent = hasCut ? (review ? "Flagged for review" : "Pass") : "Score only";
  decision.closest(".decision").classList.toggle("review", review);
  byId("score-value").textContent = numberFormat.format(sample.score);
  byId("threshold-value").textContent = hasCut ? numberFormat.format(threshold) : "Unavailable";
}

function downloadPrediction() {
  const sample = currentSample();
  const json = JSON.stringify({ ...trace, examples: [sample], selected_fpr: currentCut()?.fpr }, null, 2);
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

traceFile.addEventListener("change", async () => {
  const [file] = traceFile.files;
  if (!file) return;
  try {
    verify(file.size <= 10 * 1024 * 1024, "Open a trace JSON file smaller than 10 MB.");
    useTrace(validate(JSON.parse(await file.text())), true);
  } catch (error) {
    const message = error instanceof SyntaxError ? "The file is not valid JSON." : error.message;
    showStatus(message, true);
  } finally {
    traceFile.value = "";
  }
});
restoreButton.addEventListener("click", () => useTrace(examples));
documentSelect.addEventListener("change", showDocument);
thresholdSelect.addEventListener("change", showDecision);
byId("download").addEventListener("click", downloadPrediction);

function loadExamples() {
  try {
    examples = validate(window.NRSW_EXAMPLES);
    if (usingImport) restoreButton.hidden = false;
    else useTrace(examples);
  } catch {
    if (!usingImport) showStatus("Could not load examples. Open a trace JSON or start the demo with nrsw serve.", true);
  }
}

loadExamples();
