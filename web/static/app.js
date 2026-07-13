const state = {
  queries: [],
  scales: [],
  mode: "tpch",
  running: false,
  generatingData: false,
};

const els = {
  workspace: document.querySelector(".workspace"),
  status: document.getElementById("status"),
  querySelect: document.getElementById("querySelect"),
  scaleSelect: document.getElementById("scaleSelect"),
  warmupInput: document.getElementById("warmupInput"),
  repeatInput: document.getElementById("repeatInput"),
  tpchMode: document.getElementById("tpchMode"),
  customMode: document.getElementById("customMode"),
  runButton: document.getElementById("runButton"),
  dataStrip: document.getElementById("dataStrip"),
  dataTitle: document.getElementById("dataTitle"),
  dataDetails: document.getElementById("dataDetails"),
  generateDataButton: document.getElementById("generateDataButton"),
  sqlInput: document.getElementById("sqlInput"),
  sqlMeta: document.getElementById("sqlMeta"),
  kernelCode: document.getElementById("kernelCode"),
  kernelMeta: document.getElementById("kernelMeta"),
  queryResult: document.getElementById("queryResult"),
  resultMeta: document.getElementById("resultMeta"),
  timingCards: document.getElementById("timingCards"),
  timingText: document.getElementById("timingText"),
  timingMeta: document.getElementById("timingMeta"),
  diagnostics: document.getElementById("diagnostics"),
  diagnosticText: document.getElementById("diagnosticText"),
  commandMeta: document.getElementById("commandMeta"),
};

function setStatus(message, kind = "") {
  els.status.textContent = message;
  els.status.className = `status ${kind}`.trim();
}

function selectedQuery() {
  return state.queries.find((query) => query.id === els.querySelect.value) || state.queries[0];
}

function selectedScale() {
  return state.scales.find((scale) => scale.id === els.scaleSelect.value) || state.scales[0];
}

function setMode(mode) {
  state.mode = mode;
  els.tpchMode.classList.toggle("active", mode === "tpch");
  els.customMode.classList.toggle("active", mode === "custom");
  els.sqlInput.readOnly = mode === "tpch";
  updateSqlMeta();
}

function updateSqlFromSelection() {
  const query = selectedQuery();
  if (!query) return;
  if (state.mode === "tpch") {
    els.sqlInput.value = query.sql;
  }
  updateSqlMeta();
}

function updateSqlMeta() {
  const query = selectedQuery();
  if (state.mode === "custom") {
    els.sqlMeta.textContent = "Custom";
    return;
  }
  els.sqlMeta.textContent = query ? query.label : "";
}

function populateQueries(queries) {
  const previous = els.querySelect.value;
  els.querySelect.replaceChildren();
  for (const query of queries) {
    const option = document.createElement("option");
    option.value = query.id;
    option.textContent = query.label;
    els.querySelect.append(option);
  }
  if (previous && queries.some((query) => query.id === previous)) {
    els.querySelect.value = previous;
  }
}

function populateScales(scales) {
  const previous = els.scaleSelect.value;
  els.scaleSelect.replaceChildren();
  const preferred = scales.find((scale) => scale.ready) || scales.find((scale) => scale.directory_exists) || scales[0];
  for (const scale of scales) {
    const option = document.createElement("option");
    option.value = scale.id;
    const suffix = scale.ready ? "" : ` (${scale.colbin_count || 0}/${scale.expected_count || 8} data)`;
    option.textContent = `${scale.label}${suffix}`;
    els.scaleSelect.append(option);
  }
  if (previous && scales.some((scale) => scale.id === previous)) {
    els.scaleSelect.value = previous;
  } else if (preferred) {
    els.scaleSelect.value = preferred.id;
  }
  updateDataStatus();
}

function updateDataStatus() {
  const scale = selectedScale();
  if (!scale) return;
  const count = `${scale.colbin_count || 0}/${scale.expected_count || 8}`;
  els.workspace.classList.toggle("data-ready", Boolean(scale.ready));
  els.dataStrip.classList.toggle("ready", Boolean(scale.ready));
  els.dataStrip.classList.toggle("missing", !scale.ready);
  els.dataTitle.textContent = scale.ready ? `${scale.label} data ready` : `${scale.label} data missing`;
  els.dataDetails.textContent = scale.ready
    ? `${count} .colbin tables in ${scale.path}`
    : `${count} .colbin tables in ${scale.path}. Generate data before Run.`;
  els.generateDataButton.disabled = state.generatingData || scale.ready;
  els.generateDataButton.querySelector("span:last-child").textContent = state.generatingData ? "Generating" : "Generate Data";
}

function formatMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  if (value >= 10) return `${value.toFixed(1)} ms`;
  return `${value.toFixed(3)} ms`;
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat().format(value);
}

function renderTiming(timing, wallMs) {
  const metrics = [
    ["End-to-End", formatMs(timing.end_to_end_ms)],
    ["Query Execution", formatMs(timing.query_execution_ms)],
    ["GPU Compute", formatMs(timing.gpu_compute_ms)],
    ["Hot Execution", formatMs(timing.hot_execution_ms)],
    ["Compile Overhead", formatMs(timing.compile_overhead_ms)],
    ["Data Load", formatMs(timing.data_load_ms)],
    ["Metal Codegen", formatMs(timing.codegen_ms)],
    ["Metal Compile", formatMs(timing.metal_compile_ms)],
  ];
  els.timingCards.replaceChildren();
  for (const [label, value] of metrics) {
    const card = document.createElement("div");
    card.className = "metric";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = label;
    card.append(strong, span);
    els.timingCards.append(card);
  }

  const details = [];
  if (timing.scale_factor) details.push(`scale=${timing.scale_factor}`);
  if (timing.route) details.push(`route=${timing.route}`);
  if (typeof timing.load_bytes === "number" && timing.load_bytes > 0) {
    details.push(`bytes=${formatNumber(timing.load_bytes)}`);
  }
  if (typeof wallMs === "number" && wallMs > 0) details.push(`wall=${formatMs(wallMs)}`);
  els.timingMeta.textContent = details.join("  ");
}

function showDiagnostics(result) {
  const lines = [];
  if (result.command) lines.push(`$ ${result.command}`);
  if (result.returncode !== undefined) lines.push(`returncode=${result.returncode}`);
  if (result.kernel_path) lines.push(`kernel=${result.kernel_path}`);
  if (result.build_log) lines.push("\n[build]\n" + result.build_log.trim());
  if (result.stderr) lines.push("\n[stderr]\n" + result.stderr.trim());
  if (result.stdout) lines.push("\n[stdout]\n" + result.stdout.trim());

  els.commandMeta.textContent = result.run_id || "";
  els.diagnosticText.textContent = lines.join("\n");
  els.diagnostics.classList.toggle("hidden", result.ok && !result.stderr && !result.build_log);
}

function setRunning(running) {
  state.running = running;
  els.runButton.disabled = running || state.generatingData;
  els.runButton.querySelector("span:last-child").textContent = running ? "Running" : "Run";
}

async function refreshMetadata() {
  const response = await fetch("/api/queries");
  const data = await response.json();
  state.queries = data.queries || [];
  state.scales = data.scales || [];
  populateQueries(state.queries);
  populateScales(state.scales);
  return data;
}

async function loadInitialData() {
  setStatus("Loading metadata...");
  const data = await refreshMetadata();
  setMode("tpch");
  updateSqlFromSelection();
  els.kernelCode.textContent = "";
  els.queryResult.textContent = "";
  els.timingText.textContent = "";
  renderTiming({});
  const binary = data.binary_exists ? "backend ready" : "backend will build on first run";
  setStatus(binary);
}

async function runQuery() {
  if (state.running) return;
  const scale = selectedScale();
  if (scale && !scale.ready) {
    setStatus(`${scale.label} data is missing. Generate data first.`, "error");
    els.queryResult.textContent = `Missing data for ${scale.label}.\nRequired: ${scale.expected_count || 8} .colbin tables\nFound: ${scale.colbin_count || 0}\nPath: ${scale.path}`;
    updateDataStatus();
    return;
  }
  setRunning(true);
  setStatus(state.mode === "custom" ? "Running custom SQL..." : `Running ${els.querySelect.value.toUpperCase()}...`, "running");
  els.kernelMeta.textContent = "";
  els.resultMeta.textContent = "";
  els.timingMeta.textContent = "";
  els.kernelCode.textContent = "";
  els.queryResult.textContent = "";
  els.timingText.textContent = "";
  els.diagnostics.classList.add("hidden");

  const payload = {
    mode: state.mode,
    query_id: els.querySelect.value,
    scale: els.scaleSelect.value,
    warmup: els.warmupInput.value,
    repeat: els.repeatInput.value,
    custom_sql: state.mode === "custom" ? els.sqlInput.value : "",
  };

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      setStatus(result.message || "Run failed.", "error");
      if (result.data_status) {
        const idx = state.scales.findIndex((scale) => scale.id === result.data_status.id);
        if (idx >= 0) state.scales[idx] = result.data_status;
        updateDataStatus();
      }
    } else {
      setStatus(result.message || "Run completed.");
    }

    els.kernelCode.textContent = result.kernel_code || "No kernel code was produced.";
    els.kernelMeta.textContent = result.kernel_path || "";
    els.queryResult.textContent = result.query_result || "No rows were printed.";
    els.resultMeta.textContent = result.query || "";
    els.timingText.textContent = result.timing_text || "No timing block was produced.";
    renderTiming(result.timing || {}, result.wall_ms);
    showDiagnostics(result);
  } catch (error) {
    setStatus(error.message || "Request failed.", "error");
    els.diagnosticText.textContent = String(error);
    els.diagnostics.classList.remove("hidden");
  } finally {
    setRunning(false);
  }
}

async function generateData() {
  if (state.generatingData || state.running) return;
  const scale = selectedScale();
  if (!scale || scale.ready) return;
  state.generatingData = true;
  setRunning(false);
  updateDataStatus();
  setStatus(`Generating data for ${scale.label}...`, "running");
  els.diagnostics.classList.add("hidden");

  try {
    const response = await fetch("/api/generate-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scale: scale.id }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      setStatus(result.message || "Data generation failed.", "error");
    } else {
      setStatus(result.message || "Data is ready.");
    }
    if (result.data_status) {
      const idx = state.scales.findIndex((item) => item.id === result.data_status.id);
      if (idx >= 0) state.scales[idx] = result.data_status;
    }
    showDiagnostics(result);
    await refreshMetadata();
  } catch (error) {
    setStatus(error.message || "Data generation request failed.", "error");
    els.diagnosticText.textContent = String(error);
    els.diagnostics.classList.remove("hidden");
  } finally {
    state.generatingData = false;
    setRunning(false);
    updateDataStatus();
  }
}

els.querySelect.addEventListener("change", updateSqlFromSelection);
els.scaleSelect.addEventListener("change", updateDataStatus);
els.tpchMode.addEventListener("click", () => {
  setMode("tpch");
  updateSqlFromSelection();
});
els.customMode.addEventListener("click", () => setMode("custom"));
els.runButton.addEventListener("click", runQuery);
els.generateDataButton.addEventListener("click", generateData);

loadInitialData().catch((error) => {
  setStatus(error.message || "Failed to load metadata.", "error");
});
