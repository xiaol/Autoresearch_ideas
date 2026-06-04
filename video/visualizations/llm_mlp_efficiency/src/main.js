import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createIcons, icons } from "lucide";
import "./styles.css";

const TRACE_CATALOG = {
  "qwen35-08b": {
    label: "Qwen3.5 0.8B",
    url: "/traces/qwen35-08b-sample.json",
  },
  "rwkv7-01b": {
    label: "RWKV-7 Goose 0.1B",
    url: "/traces/rwkv7-01b-sample.json",
  },
};
const DEFAULT_MODEL_KEY = "qwen35-08b";
const PLAY_INTERVAL_MS = 1250;

const state = {
  trace: null,
  modelKey: DEFAULT_MODEL_KEY,
  traceTemplates: new Map(),
  comparisonTraces: [],
  stepIndex: 0,
  selectedLayerIndex: 0,
  inputTokenIndex: 0,
  channelSampleMode: "strong",
  outputReadout: "channels",
  viewMode: "mlp",
  playing: false,
  playTimer: null,
};

const sceneState = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  root: null,
  tokenRoot: null,
  layerObjects: [],
  connectionObjects: [],
  mlpObjects: null,
  comparisonObjects: [],
  manifoldObjects: null,
  activeMarker: null,
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(),
  startTime: performance.now(),
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  initialize().catch((error) => {
    console.error(error);
    setStatus(`Startup failed: ${error.message}`, true);
  });
});

async function initialize() {
  bindDom();
  syncViewModeButtons();
  createIcons({ icons });
  setupScene();
  installRecordingApi();
  bindEvents();
  await loadCatalogTrace(DEFAULT_MODEL_KEY);
  animate();
}

function bindDom() {
  const ids = [
    "networkCanvas",
    "modelSelect",
    "modelName",
    "layerCount",
    "hiddenSize",
    "contextLength",
    "viewMlpButton",
    "viewBlockButton",
    "viewCompareButton",
    "inputTokenSlider",
    "inputTokenValue",
    "channelSampleSelect",
    "outputReadoutSelect",
    "promptInput",
    "applyPromptButton",
    "traceStatus",
    "traceFileInput",
    "resetViewButton",
    "playPauseButton",
    "prevStepButton",
    "nextStepButton",
    "stepSlider",
    "stepLabel",
    "tokenStrip",
    "selectedLayerBadge",
    "activeTokenLabel",
    "generatedTextLabel",
    "outputPanelTitle",
    "layerDetail",
    "layerList",
    "topTokens",
  ];

  for (const id of ids) {
    dom[id] = document.getElementById(id);
  }
}

function bindEvents() {
  window.addEventListener("resize", resizeRenderer);
  dom.networkCanvas.addEventListener("pointerdown", handleCanvasPick);
  dom.modelSelect.addEventListener("change", (event) => {
    state.viewMode = "mlp";
    loadCatalogTrace(event.target.value).catch((error) => {
      console.error(error);
      setStatus(`Could not load model trace: ${error.message}`, true);
      syncModelSelect();
    });
  });
  dom.resetViewButton.addEventListener("click", resetCamera);
  dom.viewMlpButton.addEventListener("click", () => setViewMode("mlp"));
  dom.viewBlockButton.addEventListener("click", () => setViewMode("block"));
  dom.viewCompareButton.addEventListener("click", () => setViewMode("compare"));
  dom.inputTokenSlider.addEventListener("input", (event) => {
    setInputTokenIndex(Number(event.target.value), { updateStatus: true });
  });
  dom.channelSampleSelect.addEventListener("change", (event) => {
    state.channelSampleMode = event.target.value;
    rebuildNetworkScene();
    renderAllPanels();
    updateSceneMetrics();
    setStatus(`Channel sample: ${getSampleModeLabel(state.channelSampleMode)}.`);
  });
  dom.outputReadoutSelect.addEventListener("change", (event) => {
    if (!state.trace) return;
    state.outputReadout = event.target.value;
    renderTopTokens();
    renderLayerDetail();
    setStatus(state.outputReadout === "tokens" ? "Showing next-token output readout." : "Showing residual channel output readout.");
  });
  dom.applyPromptButton.addEventListener("click", () => {
    if (!state.trace) return;
    const prompt = dom.promptInput.value.trim() || state.trace.prompt || "";
    if (state.viewMode === "compare") {
      buildComparisonForPrompt(prompt).catch((error) => {
        console.error(error);
        setStatus(`Could not build comparison: ${error.message}`, true);
      });
      return;
    }
    const trace = buildSyntheticPromptTrace(prompt, state.trace);
    applyTrace(trace, `Local prompt trace built from ${getModelDisplayName(state.trace)} metadata.`);
  });
  dom.traceFileInput.addEventListener("change", handleTraceFile);
  dom.playPauseButton.addEventListener("click", togglePlayback);
  dom.prevStepButton.addEventListener("click", () => setStep(state.stepIndex - 1));
  dom.nextStepButton.addEventListener("click", () => setStep(state.stepIndex + 1));
  dom.stepSlider.addEventListener("input", (event) => setStep(Number(event.target.value), { fromSlider: true }));
}

async function loadCatalogTrace(modelKey) {
  const catalogEntry = TRACE_CATALOG[modelKey];
  if (!catalogEntry) return;
  state.modelKey = modelKey;
  syncModelSelect();
  await loadTraceFromUrl(catalogEntry.url, {
    modelKey,
    loadingText: `Loading sample ${catalogEntry.label} trace.`,
    loadedText: `${catalogEntry.label} sample trace loaded.`,
  });
}

async function fetchCatalogTraceTemplate(modelKey) {
  if (state.traceTemplates.has(modelKey)) {
    return state.traceTemplates.get(modelKey);
  }
  const catalogEntry = TRACE_CATALOG[modelKey];
  if (!catalogEntry) {
    throw new Error(`Unknown model trace: ${modelKey}`);
  }
  const response = await fetch(catalogEntry.url);
  if (!response.ok) {
    throw new Error(`${catalogEntry.label} trace fetch failed with HTTP ${response.status}`);
  }
  const trace = normalizeTrace(await response.json());
  state.traceTemplates.set(modelKey, trace);
  return trace;
}

async function buildComparisonForPrompt(prompt) {
  setStatus("Building Qwen/RWKV comparison from the shared prompt.");
  const modelKeys = ["qwen35-08b", "rwkv7-01b"];
  const templates = await Promise.all(modelKeys.map((modelKey) => fetchCatalogTraceTemplate(modelKey)));
  state.comparisonTraces = templates.map((template) => normalizeTrace(buildSyntheticPromptTrace(prompt, template)));
  state.trace = state.comparisonTraces.find((trace) => inferTraceCatalogKey(trace) === state.modelKey) ?? state.comparisonTraces[0];
  state.modelKey = inferTraceCatalogKey(state.trace) ?? state.modelKey;
  state.stepIndex = 0;
  const sharedLayerCount = Math.min(...state.comparisonTraces.map((trace) => trace.architecture.layers.length));
  state.selectedLayerIndex = clamp(state.selectedLayerIndex ?? 0, 0, Math.max(sharedLayerCount - 1, 0));
  state.inputTokenIndex = getActiveStep()?.activeTokenIndex ?? 0;
  dom.promptInput.value = prompt;
  dom.stepSlider.max = String(Math.max(0, getComparisonStepCount() - 1));
  dom.stepSlider.value = "0";
  syncModelSelect();
  syncIoControls();
  renderModelStats();
  rebuildNetworkScene();
  resetCamera();
  setStep(0, { force: true });
  setStatus("Comparison built: Qwen3.5 0.8B and RWKV-7 Goose 0.1B use the same prompt.");
}

async function loadTraceFromUrl(url, options = {}) {
  setStatus(options.loadingText ?? "Loading sample trace.");
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Trace fetch failed with HTTP ${response.status}`);
  }
  const trace = await response.json();
  applyTrace(trace, options.loadedText ?? "Sample trace loaded.", { modelKey: options.modelKey });
}

function applyTrace(rawTrace, statusText, options = {}) {
  stopPlayback();
  state.trace = normalizeTrace(rawTrace);
  state.modelKey = options.modelKey ?? inferTraceCatalogKey(state.trace) ?? "custom";
  if (TRACE_CATALOG[state.modelKey]) {
    state.traceTemplates.set(state.modelKey, state.trace);
  }
  state.stepIndex = 0;
  state.selectedLayerIndex = Math.min(state.selectedLayerIndex ?? 0, state.trace.architecture.layers.length - 1);
  state.inputTokenIndex = getActiveStep()?.activeTokenIndex ?? 0;
  dom.promptInput.value = state.trace.prompt ?? "";
  dom.stepSlider.max = String(Math.max(0, state.trace.steps.length - 1));
  dom.stepSlider.value = "0";
  syncModelSelect();
  syncViewModeButtons();
  syncIoControls();
  renderModelStats();
  rebuildNetworkScene();
  setStep(0, { force: true });
  setStatus(statusText);
}

function setViewMode(mode) {
  const nextMode = mode === "compare" ? "compare" : mode === "block" ? "block" : "mlp";
  if (state.viewMode === nextMode && state.trace) return;
  state.viewMode = nextMode;
  syncViewModeButtons();
  if (nextMode === "compare") {
    const prompt = dom.promptInput.value.trim() || state.trace?.prompt || "";
    buildComparisonForPrompt(prompt).catch((error) => {
      console.error(error);
      setStatus(`Could not build comparison: ${error.message}`, true);
    });
    return;
  }
  dom.stepSlider.max = String(Math.max(0, state.trace.steps.length - 1));
  rebuildNetworkScene();
  resetCamera();
  renderAllPanels();
  updateSceneMetrics();
  setStatus(
    nextMode === "mlp"
      ? isRwkvTrace()
        ? "Showing selected RWKV time/channel mix block."
        : "Showing selected block FFN/MLP only."
      : isRwkvTrace()
        ? "Showing full RWKV recurrent block flow."
        : "Showing full transformer block flow.",
  );
}

function syncViewModeButtons() {
  const mlpActive = state.viewMode === "mlp";
  const compareActive = state.viewMode === "compare";
  dom.viewMlpButton?.classList.toggle("active", mlpActive);
  dom.viewBlockButton?.classList.toggle("active", state.viewMode === "block");
  dom.viewCompareButton?.classList.toggle("active", compareActive);
  dom.viewMlpButton?.setAttribute("aria-pressed", String(mlpActive));
  dom.viewBlockButton?.setAttribute("aria-pressed", String(state.viewMode === "block"));
  dom.viewCompareButton?.setAttribute("aria-pressed", String(compareActive));
}

function syncModelSelect() {
  if (!dom.modelSelect) return;
  if (state.viewMode === "compare") {
    dom.modelSelect.value = "comparison";
    return;
  }
  dom.modelSelect.value = TRACE_CATALOG[state.modelKey] ? state.modelKey : "custom";
}

function inferTraceCatalogKey(trace) {
  const modelName = String(trace?.model?.name ?? "").toLowerCase();
  if (isRwkvTrace(trace)) return "rwkv7-01b";
  if (modelName.includes("qwen")) return "qwen35-08b";
  return null;
}

function isRwkvTrace(trace = state.trace) {
  const model = trace?.model ?? {};
  const architecture = trace?.architecture ?? {};
  return [model.family, model.modelType, model.model_type, architecture.blockKind, architecture.block_kind, architecture.modelType]
    .some((value) => String(value ?? "").toLowerCase().includes("rwkv"));
}

function isRwkvLayer(layer) {
  return String(layer?.kind ?? "").toLowerCase().includes("rwkv");
}

function getModelDisplayName(trace = state.trace) {
  return trace?.model?.displayName ?? trace?.model?.name ?? "the selected model";
}

function getRenderContext(trace = state.trace, overrides = {}) {
  const stepIndex = clamp(
    Math.round(overrides.stepIndex ?? state.stepIndex),
    0,
    Math.max((trace?.steps?.length ?? 1) - 1, 0),
  );
  const layerIndex = clamp(
    Math.round(overrides.layerIndex ?? state.selectedLayerIndex),
    0,
    Math.max((trace?.architecture?.layers?.length ?? 1) - 1, 0),
  );
  const activeStep = trace?.steps?.[stepIndex] ?? trace?.steps?.[0];
  return {
    trace,
    stepIndex,
    layerIndex,
    inputTokenIndex: overrides.inputTokenIndex ?? state.inputTokenIndex ?? activeStep?.activeTokenIndex ?? 0,
  };
}

function getComparisonStepCount() {
  if (!state.comparisonTraces.length) return state.trace?.steps?.length ?? 1;
  return Math.max(...state.comparisonTraces.map((trace) => trace.steps.length), 1);
}

function syncIoControls() {
  if (!state.trace) return;
  const maxTokenIndex = Math.max(0, state.trace.tokens.length - 1);
  state.inputTokenIndex = clamp(Math.round(state.inputTokenIndex), 0, maxTokenIndex);
  if (dom.inputTokenSlider) {
    dom.inputTokenSlider.max = String(maxTokenIndex);
    dom.inputTokenSlider.value = String(state.inputTokenIndex);
  }
  if (dom.channelSampleSelect) {
    dom.channelSampleSelect.value = state.channelSampleMode;
  }
  if (dom.outputReadoutSelect) {
    dom.outputReadoutSelect.value = state.outputReadout;
  }
  updateInputTokenValue();
}

function setInputTokenIndex(nextIndex, options = {}) {
  if (!state.trace) return;
  const maxTokenIndex = Math.max(0, state.trace.tokens.length - 1);
  state.inputTokenIndex = clamp(Math.round(nextIndex), 0, maxTokenIndex);
  syncIoControls();
  if (state.viewMode === "mlp") {
    rebuildTokenObjects();
    rebuildEmbeddingManifold();
  } else if (state.viewMode === "compare") {
    rebuildTokenObjects();
  } else {
    rebuildNetworkScene();
  }
  renderAllPanels();
  updateSceneMetrics();
  if (options.updateStatus) {
    const token = state.trace.tokens[state.inputTokenIndex];
    setStatus(`Input token: ${state.inputTokenIndex} ${token ? visibleTokenText(token.text) : ""}`.trim());
  }
}

function updateInputTokenValue() {
  if (!dom.inputTokenValue || !state.trace) return;
  const token = state.trace.tokens[state.inputTokenIndex];
  dom.inputTokenValue.textContent = token
    ? `${state.inputTokenIndex}: ${visibleTokenText(token.text)}`
    : `Token ${state.inputTokenIndex}`;
}

function getSampleModeLabel(mode) {
  if (mode === "spread") return "even spread";
  if (mode === "seeded") return "seeded sample";
  return "high activity";
}

function normalizeTrace(rawTrace) {
  const model = rawTrace.model ?? {};
  const architecture = rawTrace.architecture ?? {};
  const layers = Array.isArray(architecture.layers) && architecture.layers.length
    ? architecture.layers.map((layer, index) => normalizeLayer(layer, index))
    : buildFallbackLayers(model, architecture);
  const tokens = Array.isArray(rawTrace.tokens) && rawTrace.tokens.length
    ? rawTrace.tokens.map((token, index) => normalizeToken(token, index))
    : tokenizeDisplayText(rawTrace.prompt ?? "").map((text, index) => ({
        index,
        id: null,
        text,
        source: "prompt",
      }));

  const steps = Array.isArray(rawTrace.steps) && rawTrace.steps.length
    ? rawTrace.steps.map((step, index) => normalizeStep(step, index, layers, tokens, rawTrace.seed ?? 17))
    : buildFallbackSteps(tokens, layers, rawTrace.seed ?? 17, rawTrace);

  return {
    schemaVersion: rawTrace.schemaVersion ?? rawTrace.schema_version ?? "llm_trace_v1",
    traceKind: rawTrace.traceKind ?? rawTrace.trace_kind ?? "trace",
    model,
    prompt: rawTrace.prompt ?? "",
    generatedText: rawTrace.generatedText ?? rawTrace.generated_text ?? "",
    architecture: {
      ...architecture,
      layers,
      numLayers: layers.length,
      hiddenSize: architecture.hiddenSize ?? architecture.hidden_size ?? model.hiddenSize ?? model.hidden_size,
      contextLength:
        architecture.contextLength ?? architecture.context_length ?? model.contextLength ?? model.context_length,
    },
    tokens,
    steps,
  };
}

function normalizeLayer(layer, index) {
  const kind = layer.kind ?? layer.type ?? "linear_attention";
  return {
    index: Number.isFinite(layer.index) ? layer.index : index,
    name: layer.name ?? `Block ${String(index + 1).padStart(2, "0")}`,
    kind,
    heads: layer.heads ?? inferHeadCount(kind),
    hiddenSize: layer.hiddenSize ?? layer.hidden_size,
    intermediateSize: layer.intermediateSize ?? layer.intermediate_size,
    headDim: layer.headDim ?? layer.head_dim,
  };
}

function buildFallbackLayers(model, architecture) {
  const count = Number(architecture.numLayers ?? architecture.num_layers ?? model.numLayers ?? model.num_layers ?? 24);
  const rwkvKind = String(model.family ?? model.modelType ?? model.model_type ?? architecture.blockKind ?? "")
    .toLowerCase()
    .includes("rwkv");
  return Array.from({ length: count }, (_, index) => {
    const kind = rwkvKind ? "rwkv7_block" : (index + 1) % 4 === 0 ? "full_attention" : "linear_attention";
    return normalizeLayer({ index, kind }, index);
  });
}

function normalizeToken(token, index) {
  if (typeof token === "string") {
    return { index, id: null, text: token, source: "prompt" };
  }
  return {
    index: Number.isFinite(token.index) ? token.index : index,
    id: token.id ?? null,
    text: visibleTokenText(token.text ?? token.token ?? ""),
    source: token.source ?? (token.generated ? "generated" : "prompt"),
  };
}

function normalizeStep(step, index, layers, tokens, seed) {
  const activeTokenIndex = clamp(
    Number(step.activeTokenIndex ?? step.active_token_index ?? index),
    0,
    Math.max(tokens.length - 1, 0),
  );
  const rawLayers = step.layers ?? step.layerMetrics ?? step.layer_metrics;
  const normalizedLayers = Array.isArray(rawLayers) && rawLayers.length
    ? layers.map((layer, layerIndex) => normalizeLayerMetrics(rawLayers[layerIndex], layer, layerIndex, index, seed))
    : layers.map((layer, layerIndex) => synthesizeLayerMetrics(layer, layerIndex, index, seed));

  return {
    index: Number.isFinite(step.index) ? step.index : index,
    phase: step.phase ?? (index === 0 ? "prompt" : "decode"),
    label: step.label ?? (index === 0 ? "Prompt pass" : `Decode ${index}`),
    activeTokenIndex,
    generatedToken: visibleTokenText(step.generatedToken ?? step.generated_token ?? ""),
    generatedText: step.generatedText ?? step.generated_text ?? "",
    topTokens: normalizeTopTokens(step.topTokens ?? step.top_tokens ?? []),
    layers: normalizedLayers,
  };
}

function normalizeLayerMetrics(metric, layer, layerIndex, stepIndex, seed) {
  const fallback = synthesizeLayerMetrics(layer, layerIndex, stepIndex, seed);
  if (!metric || typeof metric !== "object") return fallback;
  const heads = Array.isArray(metric.heads) && metric.heads.length
    ? metric.heads.slice(0, 6).map((head, index) => ({
        index: head.index ?? index,
        value: clamp01(Number(head.value ?? head.strength ?? head.score ?? 0)),
        label: head.label ?? `H${index + 1}`,
      }))
    : fallback.heads;

  const normalized = {
    layerIndex,
    attention: clamp01(Number(metric.attention ?? metric.attentionStrength ?? metric.timeMix ?? metric.time_mix ?? fallback.attention)),
    mlp: clamp01(Number(metric.mlp ?? metric.mlpActivity ?? metric.channelMix ?? metric.channel_mix ?? fallback.mlp)),
    residualNorm: clamp01(Number(metric.residualNorm ?? metric.residual_norm ?? metric.stateNorm ?? metric.state_norm ?? fallback.residualNorm)),
    entropy: clamp01(Number(metric.entropy ?? fallback.entropy)),
    heads,
    note: metric.note ?? metric.summary ?? fallback.note,
  };
  normalized.efficiency = normalizeMlpEfficiency(metric, layer, normalized, fallback.efficiency);
  return normalized;
}

function synthesizeLayerMetrics(layer, layerIndex, stepIndex, seed) {
  const phase = seededWave(seed + layerIndex * 19 + stepIndex * 31);
  const typeBoost = layer.kind === "full_attention" ? 0.24 : isRwkvLayer(layer) ? 0.14 : 0.08;
  const attention = clamp01(0.18 + typeBoost + phase * 0.44 + Math.sin((stepIndex + 1) * 0.8 + layerIndex) * 0.08);
  const mlp = clamp01(0.22 + seededWave(seed + layerIndex * 7 + stepIndex * 43) * 0.58);
  const residualNorm = clamp01(0.3 + layerIndex / Math.max(1, 2.2 * (state.trace?.architecture?.layers?.length ?? 24)) + phase * 0.34);
  const entropy = clamp01(layer.kind === "full_attention" ? 0.35 + phase * 0.38 : 0.2 + phase * 0.28);
  const headCount = Math.min(6, inferHeadCount(layer.kind));
  const heads = Array.from({ length: headCount }, (_, index) => ({
    index,
    label: isRwkvLayer(layer) ? `T${index + 1}` : layer.kind === "linear_attention" ? `D${index + 1}` : `H${index + 1}`,
    value: clamp01(0.18 + seededWave(seed + index * 13 + layerIndex * 5 + stepIndex * 11) * 0.76),
  }));
  const metrics = {
    layerIndex,
    attention,
    mlp,
    residualNorm,
    entropy,
    heads,
    note: layer.kind === "full_attention"
      ? "global attention block"
      : isRwkvLayer(layer)
        ? "RWKV recurrent time/channel mix block"
        : "linear recurrent state block",
  };
  metrics.efficiency = synthesizeMlpEfficiency(layer, metrics, seed + layerIndex * 71 + stepIndex * 101);
  return metrics;
}

function estimateMlpFlopsPerToken(layer) {
  const hiddenSize = Number(layer?.hiddenSize ?? state.trace?.architecture?.hiddenSize ?? 1024);
  const intermediateSize = Number(layer?.intermediateSize ?? state.trace?.architecture?.intermediateSize ?? hiddenSize * 4);
  if (!Number.isFinite(hiddenSize) || !Number.isFinite(intermediateSize)) return 0;
  return 6 * hiddenSize * intermediateSize;
}

function synthesizeMlpEfficiency(layer, metrics, seed) {
  const flopsPerToken = estimateMlpFlopsPerToken(layer);
  const megaFlopsPerToken = flopsPerToken / 1_000_000;
  const activeFraction = clamp01(0.06 + metrics.mlp * 0.34 + seededWave(seed) * 0.08);
  const topKCoverage = clamp01(0.42 + metrics.mlp * 0.34 + metrics.residualNorm * 0.1 + seededWave(seed + 19) * 0.08);
  const deltaPerMFlop = megaFlopsPerToken > 0 ? metrics.mlp / megaFlopsPerToken : 0;
  return {
    flopsPerToken,
    megaFlopsPerToken,
    activeFraction,
    activeNeurons: Math.round(activeFraction * Math.max(1, Number(layer?.intermediateSize ?? 0))),
    topK: 32,
    topKCoverage,
    deltaPerMFlop,
    estimated: true,
  };
}

function normalizeMlpEfficiency(metric, layer, normalizedMetric, fallback) {
  const source = metric.efficiency ?? metric.mlpEfficiency ?? metric.mlp_efficiency ?? metric;
  const flopsPerToken = finiteOr(
    source.flopsPerToken ?? source.flops_per_token,
    fallback?.flopsPerToken ?? estimateMlpFlopsPerToken(layer),
  );
  const megaFlopsPerToken = finiteOr(
    source.megaFlopsPerToken ?? source.mega_flops_per_token,
    fallback?.megaFlopsPerToken ?? flopsPerToken / 1_000_000,
  );
  const activeFraction = clamp01(finiteOr(source.activeFraction ?? source.active_fraction, fallback?.activeFraction ?? 0));
  const topKCoverage = clamp01(finiteOr(source.topKCoverage ?? source.top_k_coverage, fallback?.topKCoverage ?? 0));
  const deltaPerMFlop = finiteOr(
    source.deltaPerMFlop ?? source.delta_per_mflop,
    fallback?.deltaPerMFlop ?? (megaFlopsPerToken > 0 ? normalizedMetric.mlp / megaFlopsPerToken : 0),
  );
  const inferredActiveNeurons = Math.round(activeFraction * Math.max(1, Number(layer?.intermediateSize ?? 0)));
  return {
    flopsPerToken,
    megaFlopsPerToken,
    activeFraction,
    activeNeurons: Math.round(finiteOr(source.activeNeurons ?? source.active_neurons, fallback?.activeNeurons ?? inferredActiveNeurons)),
    topK: Number(source.topK ?? source.top_k ?? fallback?.topK ?? 32),
    topKCoverage,
    deltaPerMFlop,
    estimated: Boolean(source.estimated ?? fallback?.estimated ?? true),
  };
}

function normalizeTopTokens(candidates) {
  if (!Array.isArray(candidates)) return [];
  const mapped = candidates.slice(0, 8).map((candidate) => {
    if (typeof candidate === "string") {
      return { token: candidate, probability: 0 };
    }
    return {
      token: visibleTokenText(candidate.token ?? candidate.text ?? ""),
      probability: clamp01(Number(candidate.probability ?? candidate.prob ?? candidate.p ?? candidate.score ?? 0)),
    };
  });
  const max = Math.max(...mapped.map((candidate) => candidate.probability), 0);
  if (max > 1) {
    for (const candidate of mapped) {
      candidate.probability = clamp01(candidate.probability / max);
    }
  }
  return mapped;
}

function buildFallbackSteps(tokens, layers, seed, trace) {
  const generated = isRwkvTrace(trace) ? [" state", " mix", " carries", " context"] : [" visual", " trace", " shows", " layers"];
  const promptLength = tokens.length;
  const allTokens = [...tokens];
  for (const token of generated) {
    allTokens.push({ index: allTokens.length, id: null, text: token, source: "generated" });
  }
  return Array.from({ length: Math.min(6, generated.length + 1) }, (_, index) => ({
    index,
    phase: index === 0 ? "prompt" : "decode",
    label: index === 0 ? "Prompt pass" : `Decode ${index}`,
    activeTokenIndex: index === 0 ? Math.max(0, promptLength - 1) : promptLength + index - 1,
    generatedToken: index === 0 ? "" : generated[index - 1],
    topTokens: makeSyntheticTopTokens(seed + index * 23, trace),
    layers: layers.map((layer, layerIndex) => synthesizeLayerMetrics(layer, layerIndex, index, seed)),
  }));
}

function buildSyntheticPromptTrace(prompt, templateTrace) {
  const seed = hashText(prompt) || 71;
  const promptTokens = tokenizeDisplayText(prompt).slice(0, 48).map((text, index) => ({
    index,
    id: null,
    text,
    source: "prompt",
  }));
  const generatedPlan = chooseGeneratedTokens(seed, templateTrace);
  const tokens = [...promptTokens];
  for (const token of generatedPlan) {
    tokens.push({ index: tokens.length, id: null, text: token, source: "generated" });
  }
  const rawTrace = {
    ...templateTrace,
    traceKind: "synthetic_prompt",
    prompt,
    generatedText: generatedPlan.join(""),
    tokens,
    steps: Array.from({ length: generatedPlan.length + 1 }, (_, index) => ({
      index,
      phase: index === 0 ? "prompt" : "decode",
      label: index === 0 ? "Prompt pass" : `Decode ${index}`,
      activeTokenIndex: index === 0 ? Math.max(0, promptTokens.length - 1) : promptTokens.length + index - 1,
      generatedToken: index === 0 ? "" : generatedPlan[index - 1],
      topTokens: makeSyntheticTopTokens(seed + index * 29, templateTrace),
    })),
    seed,
  };
  return rawTrace;
}

function chooseGeneratedTokens(seed, trace = state.trace) {
  const plans = isRwkvTrace(trace) ? [
    [" keeps", " state", " across", " tokens", "."],
    [" mixes", " time", " and", " channels", "."],
    [" updates", " recurrent", " memory", " compactly", "."],
    [" routes", " state", " through", " gates", "."],
  ] : [
    [" visual", "izes", " attention", " flow", "."],
    [" maps", " residual", " paths", " clearly", "."],
    [" reveals", " active", " blocks", " now", "."],
    [" traces", " Qwen", " layers", " compactly", "."],
  ];
  return plans[seed % plans.length];
}

function makeSyntheticTopTokens(seed, trace = state.trace) {
  const pool = isRwkvTrace(trace)
    ? [" state", " mix", " recurrent", " token", " memory", " channel", " gate", " context"]
    : [" visual", " layer", " token", " attention", " residual", " Qwen", " map", " state"];
  return pool.map((token, index) => ({
    token,
    probability: clamp01(0.04 + seededWave(seed + index * 17) * (0.48 - index * 0.035)),
  })).sort((a, b) => b.probability - a.probability);
}

function setupScene() {
  const canvas = dom.networkCanvas;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101a33);
  scene.fog = new THREE.Fog(0x101a33, 18, 42);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
  camera.position.set(0, 7.2, 20);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.minDistance = 8;
  controls.maxDistance = 38;

  scene.add(new THREE.HemisphereLight(0xf5f7ff, 0x101a33, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 2.1);
  key.position.set(-6, 10, 9);
  scene.add(key);
  const rim = new THREE.PointLight(0x88a4ff, 35, 28, 2);
  rim.position.set(8, 2, -8);
  scene.add(rim);

  const floor = new THREE.GridHelper(28, 28, 0x2d3f72, 0x16213f);
  floor.position.y = -2.55;
  floor.material.opacity = 0.32;
  floor.material.transparent = true;
  scene.add(floor);

  const root = new THREE.Group();
  const tokenRoot = new THREE.Group();
  scene.add(root, tokenRoot);

  const activeMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 28, 18),
    new THREE.MeshStandardMaterial({
      color: 0x9be15d,
      emissive: 0x4fd1c5,
      emissiveIntensity: 0.85,
      roughness: 0.36,
    }),
  );
  activeMarker.visible = false;
  scene.add(activeMarker);

  Object.assign(sceneState, {
    renderer,
    scene,
    camera,
    controls,
    root,
    tokenRoot,
    activeMarker,
  });

  resizeRenderer();
  resetCamera();
}

function installRecordingApi() {
  window.__LLM_VISUALIZER__ = {
    setCameraOrbit(progress, options = {}) {
      const { camera, controls } = sceneState;
      if (!camera || !controls) return;
      const p = clamp01(Number(progress));
      const compare = state.viewMode === "compare";
      const radius = Number(options.radius ?? (compare ? 28.2 : 31.5));
      const height = Number(options.height ?? (compare ? 8.6 : 8.5));
      const targetY = Number(options.targetY ?? (compare ? -0.15 : -0.8));
      const sweep = Number(options.sweep ?? (compare ? 0.86 : 0.96));
      const angle = -sweep * 0.5 + p * sweep;
      const lift = Math.sin(p * Math.PI) * 0.55;
      camera.position.set(Math.sin(angle) * radius, height + lift, Math.cos(angle) * radius);
      controls.target.set(0, targetY, 0);
      controls.update();
    },
    resetCamera,
  };
}

function rebuildNetworkScene() {
  clearGroup(sceneState.root);
  clearGroup(sceneState.tokenRoot);
  sceneState.layerObjects = [];
  sceneState.connectionObjects = [];
  sceneState.mlpObjects = null;
  sceneState.comparisonObjects = [];
  sceneState.manifoldObjects = null;

  if (!state.trace) return;
  if (state.viewMode === "compare") {
    rebuildComparisonScene();
    return;
  }
  if (state.viewMode === "mlp") {
    rebuildMlpScene();
    return;
  }

  const layers = state.trace.architecture.layers;
  const layerSpacing = Math.min(0.86, 18 / Math.max(layers.length - 1, 1));
  const startX = -((layers.length - 1) * layerSpacing) / 2;
  const groupY = 0.15;

  layers.forEach((layer, index) => {
    const group = new THREE.Group();
    group.position.set(startX + index * layerSpacing, groupY, 0);
    group.userData.layerIndex = index;

    const typeColor = getLayerColor(layer.kind);
    const baseMaterial = new THREE.MeshStandardMaterial({
      color: typeColor.base,
      emissive: typeColor.emissive,
      emissiveIntensity: 0.18,
      roughness: 0.4,
      metalness: 0.16,
    });
    const residual = new THREE.Mesh(new THREE.SphereGeometry(0.18, 24, 16), baseMaterial.clone());
    residual.position.y = 0;
    residual.userData.layerIndex = index;

    const attention = new THREE.Mesh(
      new THREE.TorusGeometry(0.28, 0.045, 12, 34),
      baseMaterial.clone(),
    );
    attention.position.y = 0.78;
    attention.rotation.x = Math.PI / 2;
    attention.userData.layerIndex = index;

    const mlp = new THREE.Mesh(
      new THREE.BoxGeometry(0.36, 0.28, 0.36),
      new THREE.MeshStandardMaterial({
        color: 0xff735c,
        emissive: 0x7a1f12,
        emissiveIntensity: 0.18,
        roughness: 0.42,
      }),
    );
    mlp.position.y = -0.78;
    mlp.userData.layerIndex = index;

    const stem = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -1.15, 0),
        new THREE.Vector3(0, 1.15, 0),
      ]),
      new THREE.LineBasicMaterial({ color: 0x6f7568, transparent: true, opacity: 0.26 }),
    );

    group.add(stem, residual, attention, mlp);
    sceneState.root.add(group);
    sceneState.layerObjects.push({ group, residual, attention, mlp, layer });

    if (index > 0) {
      const prevX = startX + (index - 1) * layerSpacing;
      const x = startX + index * layerSpacing;
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(prevX, groupY, 0),
        new THREE.Vector3((prevX + x) / 2, groupY + 0.16, 0.34),
        new THREE.Vector3(x, groupY, 0),
      ]);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(12)),
        new THREE.LineBasicMaterial({ color: 0xf6bd4b, transparent: true, opacity: 0.22 }),
      );
      sceneState.root.add(line);
      sceneState.connectionObjects.push(line);
    }
  });

  rebuildTokenObjects();
}

function rebuildMlpScene() {
  sceneState.activeMarker.visible = false;
  const layer = state.trace.architecture.layers[state.selectedLayerIndex] ?? state.trace.architecture.layers[0];
  const context = getRenderContext(state.trace, { layerIndex: state.selectedLayerIndex });
  const hiddenSize = layer?.hiddenSize ?? state.trace.architecture.hiddenSize ?? 1024;
  const intermediateSize = layer?.intermediateSize ?? state.trace.architecture.intermediateSize ?? 3584;
  const stages = getMixStages(layer, hiddenSize, intermediateSize);
  const activeStep = getActiveStep();
  const metrics =
    activeStep.layers[state.selectedLayerIndex] ?? synthesizeLayerMetrics(layer, state.selectedLayerIndex, state.stepIndex, 17);
  const objects = {
    stages: [],
    connections: [],
    labels: [],
  };

  for (const stage of stages) {
    const group = new THREE.Group();
    group.position.set(stage.x, 0.08, 0);
    const sampledChannels = buildSampledChannels(stage, metrics, context);
    const columns = Math.ceil(Math.sqrt(sampledChannels.length));
    const rows = Math.ceil(sampledChannels.length / columns);
    const zGap = 1.15;
    const yGap = 1.15;
    const nodes = [];
    const geometry = new THREE.SphereGeometry(stage.key === "input" || stage.key === "output" ? 0.22 : 0.2, 16, 12);

    for (let index = 0; index < sampledChannels.length; index += 1) {
      const col = index % columns;
      const row = Math.floor(index / columns);
      const node = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({ color: 0x111111, toneMapped: false }),
      );
      node.position.set(0, (rows - 1) * yGap * 0.5 - row * yGap, (col - (columns - 1) / 2) * zGap);
      node.userData = { stage: stage.key, unitIndex: index, channelIndex: sampledChannels[index] };
      group.add(node);
      nodes.push(node);
    }

    const label = makeTextSprite(stage.label, stage.key === "output" ? "#dfd1ff" : "#f4f1e8");
    label.position.set(stage.x, 7.35, -0.2);
    label.scale.set(1.7, 0.38, 1);
    sceneState.root.add(label);
    objects.labels.push(label);
    sceneState.root.add(group);
    objects.stages.push({ ...stage, group, nodes, sampledChannels });
  }

  sceneState.root.updateMatrixWorld(true);
  connectMlpStages(objects, 0, 1, 8, getRenderContext());
  connectMlpStages(objects, 1, 2, 3, getRenderContext());
  connectMlpStages(objects, 2, 3, 8, getRenderContext());

  const title = makeTextSprite(
    isRwkvLayer(layer) ? `Block ${state.selectedLayerIndex + 1} RWKV Mix` : `Block ${state.selectedLayerIndex + 1} FFN`,
    "#9be15d",
  );
  title.position.set(0, 8.6, -0.3);
  title.scale.set(1.9, 0.42, 1);
  sceneState.root.add(title);
  objects.labels.push(title);

  sceneState.mlpObjects = objects;
  rebuildTokenObjects();
  rebuildEmbeddingManifold();
}

function rebuildComparisonScene() {
  sceneState.activeMarker.visible = false;
  const traces = state.comparisonTraces.length ? state.comparisonTraces : [state.trace].filter(Boolean);
  const xOffsets = traces.length > 1 ? [-4.7, 4.7] : [0];

  traces.slice(0, 2).forEach((trace, traceIndex) => {
    const context = getRenderContext(trace, {
      layerIndex: Math.min(state.selectedLayerIndex, trace.architecture.layers.length - 1),
      stepIndex: Math.min(state.stepIndex, trace.steps.length - 1),
    });
    const step = trace.steps[context.stepIndex] ?? trace.steps[0];
    const layer = trace.architecture.layers[context.layerIndex] ?? trace.architecture.layers[0];
    const hiddenSize = layer?.hiddenSize ?? trace.architecture.hiddenSize ?? 1024;
    const intermediateSize = layer?.intermediateSize ?? trace.architecture.intermediateSize ?? 3584;
    const metrics = step.layers[context.layerIndex] ?? synthesizeLayerMetrics(layer, context.layerIndex, context.stepIndex, 17);
    const graphGroup = new THREE.Group();
    graphGroup.position.set(xOffsets[traceIndex], 0.2, 0);
    graphGroup.scale.setScalar(0.62);
    sceneState.root.add(graphGroup);

    const objects = {
      stages: [],
      connections: [],
      labels: [],
      graphGroup,
      trace,
      context,
    };

    const stages = getMixStages(layer, hiddenSize, intermediateSize).map((stage) => ({
      ...stage,
      count: getComparisonNodeCount(stage),
    }));

    for (const stage of stages) {
      const group = new THREE.Group();
      group.position.set(stage.x, 0.08, 0);
      const sampledChannels = buildSampledChannels(stage, metrics, context);
      const columns = Math.ceil(Math.sqrt(sampledChannels.length));
      const rows = Math.ceil(sampledChannels.length / columns);
      const zGap = 1.15;
      const yGap = 1.15;
      const nodes = [];
      const geometry = new THREE.SphereGeometry(stage.key === "input" || stage.key === "output" ? 0.22 : 0.2, 16, 12);

      for (let index = 0; index < sampledChannels.length; index += 1) {
        const col = index % columns;
        const row = Math.floor(index / columns);
        const node = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({ color: 0x111111, toneMapped: false }),
        );
        node.position.set(0, (rows - 1) * yGap * 0.5 - row * yGap, (col - (columns - 1) / 2) * zGap);
        node.userData = { stage: stage.key, unitIndex: index, channelIndex: sampledChannels[index] };
        group.add(node);
        nodes.push(node);
      }

      const label = makeTextSprite(stage.label, stage.key === "output" ? "#dfd1ff" : "#f4f1e8");
      label.position.set(stage.x, 7.35, -0.2);
      label.scale.set(2.05, 0.46, 1);
      graphGroup.add(label);
      objects.labels.push(label);
      graphGroup.add(group);
      objects.stages.push({ ...stage, group, nodes, sampledChannels });
    }

    sceneState.root.updateMatrixWorld(true);
    connectMlpStages(objects, 0, 1, 7, context);
    connectMlpStages(objects, 1, 2, 3, context);
    connectMlpStages(objects, 2, 3, 7, context);

    const title = makeTextSprite(getComparisonTitle(trace), "#9be15d");
    title.position.set(0, 9.1, -0.3);
    title.scale.set(2.55, 0.54, 1);
    graphGroup.add(title);
    objects.labels.push(title);

    const blockLabel = makeTextSprite(
      isRwkvLayer(layer) ? `Block ${context.layerIndex + 1} RWKV Mix` : `Block ${context.layerIndex + 1} FFN`,
      "#adcfff",
    );
    blockLabel.position.set(0, -7.25, -0.3);
    blockLabel.scale.set(2.15, 0.44, 1);
    graphGroup.add(blockLabel);
    objects.labels.push(blockLabel);

    sceneState.comparisonObjects.push(objects);
  });

  const sharedPrompt = makeTextSprite("same prompt", "#adcfff");
  sharedPrompt.position.set(0, -7.35, 5.7);
  sharedPrompt.scale.set(1.25, 0.28, 1);
  sceneState.root.add(sharedPrompt);
  rebuildTokenObjects();
}

function getComparisonNodeCount(stage) {
  if (stage.count <= 64) return 48;
  if (stage.count <= 96) return 72;
  return 80;
}

function getComparisonTitle(trace) {
  if (isRwkvTrace(trace)) return "RWKV-7 0.1B";
  return "Qwen3.5 0.8B";
}

function getMixStages(layer, hiddenSize, intermediateSize) {
  if (isRwkvLayer(layer)) {
    return [
      { key: "input", label: `State In ${formatNumber(hiddenSize)}`, x: -5.25, count: 64, fullSize: hiddenSize },
      { key: "gate", label: `Time Mix ${formatNumber(hiddenSize)}`, x: -1.75, count: 96, fullSize: hiddenSize },
      { key: "activation", label: `Channel Mix ${formatNumber(intermediateSize)}`, x: 1.75, count: 128, fullSize: intermediateSize },
      { key: "output", label: `State Out ${formatNumber(hiddenSize)}`, x: 5.25, count: 64, fullSize: hiddenSize },
    ];
  }
  return [
    { key: "input", label: `Input ${formatNumber(hiddenSize)}`, x: -5.25, count: 64, fullSize: hiddenSize },
    { key: "gate", label: `Gate/Up ${formatNumber(intermediateSize)}`, x: -1.75, count: 128, fullSize: intermediateSize },
    { key: "activation", label: "SiLU Gate", x: 1.75, count: 128, fullSize: intermediateSize },
    { key: "output", label: `Output ${formatNumber(hiddenSize)}`, x: 5.25, count: 64, fullSize: hiddenSize },
  ];
}

function connectMlpStages(objects, fromStageIndex, toStageIndex, fanIn, context = getRenderContext()) {
  const fromStage = objects.stages[fromStageIndex];
  const toStage = objects.stages[toStageIndex];
  for (let toIndex = 0; toIndex < toStage.nodes.length; toIndex += 1) {
    const rankedInputs = Array.from({ length: fromStage.nodes.length }, (_, fromIndex) => {
      const sourceChannel = fromStage.nodes[fromIndex].userData.channelIndex;
      const targetChannel = toStage.nodes[toIndex].userData.channelIndex;
      const weight = getSampledMlpWeight(fromStage.key, toStage.key, sourceChannel, targetChannel, context);
      return { fromIndex, weight, magnitude: Math.abs(weight) };
    })
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, fanIn);

    for (const { fromIndex, weight } of rankedInputs) {
      const from = fromStage.nodes[fromIndex].getWorldPosition(new THREE.Vector3());
      const to = toStage.nodes[toIndex].getWorldPosition(new THREE.Vector3());
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([from, to]),
        new THREE.LineBasicMaterial({
          color: 0x000000,
          transparent: true,
          opacity: 0.18,
        }),
      );
      line.userData = {
        fromStage: fromStage.key,
        toStage: toStage.key,
        fromIndex,
        toIndex,
        weight,
      };
      sceneState.root.add(line);
      objects.connections.push(line);
      sceneState.connectionObjects.push(line);
    }
  }
}

function rebuildTokenObjects() {
  clearGroup(sceneState.tokenRoot);
  if (!state.trace) return;
  const tokens = state.trace.tokens.slice(-28);
  const startX = -Math.min(14, tokens.length - 1) * 0.42;
  tokens.forEach((token, index) => {
    const sprite = makeTextSprite(token.text, token.source === "generated" ? "#ffb3a5" : "#f4f1e8");
    const col = index % 14;
    const row = Math.floor(index / 14);
    sprite.position.set(startX + col * 0.84, -9.65 - row * 0.42, -4.6);
    sprite.scale.set(0.72, 0.22, 1);
    sprite.userData.tokenIndex = token.index;
    sceneState.tokenRoot.add(sprite);
  });
}

function rebuildEmbeddingManifold() {
  if (!state.trace || !sceneState.tokenRoot) return;
  const manifoldGroup = new THREE.Group();
  manifoldGroup.name = "embedding-manifold";
  const tokens = state.trace.tokens.slice(0, 36);
  const positions = [];
  const pointGeometry = new THREE.SphereGeometry(0.13, 16, 10);
  const selectedGeometry = new THREE.SphereGeometry(0.24, 20, 14);

  tokens.forEach((token, localIndex) => {
    const position = projectTokenToManifold(token, localIndex, tokens.length);
    positions.push({ token, position });
    const selected = token.index === state.inputTokenIndex;
    const material = new THREE.MeshBasicMaterial({
      color: selected ? 0x5ba0ff : token.source === "generated" ? 0x93c5fd : 0x1e3a8a,
      toneMapped: false,
    });
    const point = new THREE.Mesh(selected ? selectedGeometry : pointGeometry, material);
    point.position.copy(position);
    point.userData.tokenIndex = token.index;
    manifoldGroup.add(point);
  });

  if (positions.length > 1) {
    const sequenceLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(positions.map((item) => item.position)),
      new THREE.LineBasicMaterial({ color: 0x5ba0ff, transparent: true, opacity: 0.38 }),
    );
    manifoldGroup.add(sequenceLine);
  }

  const selected = positions.find((item) => item.token.index === state.inputTokenIndex) ?? positions.at(-1);
  if (selected) {
    const inputGuide = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([selected.position, new THREE.Vector3(-5.25, -6.5, 1.2)]),
      new THREE.LineBasicMaterial({ color: 0x5ba0ff, transparent: true, opacity: 0.64 }),
    );
    manifoldGroup.add(inputGuide);
  }

  const outputTarget = new THREE.Vector3(6.25, -7.2, 5.6);
  const outputGuide = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(5.25, -6.5, 1.2), outputTarget]),
    new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.54 }),
  );
  manifoldGroup.add(outputGuide);

  const manifoldLabel = makeTextSprite(isRwkvTrace() ? "input state manifold" : "input embedding manifold", "#adcfff");
  manifoldLabel.position.set(-4.1, -6.55, 5.6);
  manifoldLabel.scale.set(1.35, 0.3, 1);
  manifoldGroup.add(manifoldLabel);

  const outputLabel = makeTextSprite(getOutputGuideLabel(), "#9be15d");
  outputLabel.position.copy(outputTarget);
  outputLabel.position.y -= 0.55;
  outputLabel.scale.set(1.25, 0.28, 1);
  manifoldGroup.add(outputLabel);

  sceneState.tokenRoot.add(manifoldGroup);
  sceneState.manifoldObjects = { group: manifoldGroup, positions };
}

function getOutputGuideLabel() {
  if (state.outputReadout === "tokens") return "LM head readout";
  return isRwkvTrace() ? "RWKV state readout" : "FFN output channels";
}

function projectTokenToManifold(token, localIndex, tokenCount) {
  const t = tokenCount > 1 ? localIndex / (tokenCount - 1) : 0;
  const tokenHash = hashText(`${token.id ?? ""}:${token.text}`);
  const angle = t * Math.PI * 2.4 + seededWave(tokenHash) * Math.PI * 0.65;
  const radius = 1.15 + seededWave(tokenHash + 17) * 0.7;
  const x = -5.2 + t * 10.4 + Math.cos(angle) * 0.42;
  const y = -7.55 + Math.sin(angle * 0.85) * 0.72;
  const z = 5.2 + Math.cos(angle) * radius;
  return new THREE.Vector3(x, y, z);
}

function updateSceneMetrics() {
  if (!state.trace) return;
  const step = getActiveStep();
  const elapsed = (performance.now() - sceneState.startTime) / 1000;
  if (state.viewMode === "compare") {
    updateComparisonSceneMetrics();
    return;
  }
  if (state.viewMode === "mlp") {
    updateMlpSceneMetrics(step, elapsed);
    return;
  }
  const activeLayerPosition = getActiveLayerPosition(elapsed);

  sceneState.layerObjects.forEach((object, index) => {
    const metrics = step.layers[index] ?? synthesizeLayerMetrics(object.layer, index, state.stepIndex, 17);
    const selected = index === state.selectedLayerIndex;
    const pulse = selected ? 1.2 : 1 + Math.sin(elapsed * 2.2 + index * 0.31) * 0.03;
    object.residual.scale.setScalar((0.75 + metrics.residualNorm * 1.35) * pulse);
    object.attention.scale.setScalar(0.75 + metrics.attention * 1.15);
    object.mlp.scale.set(0.7 + metrics.mlp * 1.4, 0.75 + metrics.mlp * 1.1, 0.7 + metrics.mlp * 1.4);

    const typeColor = getLayerColor(object.layer.kind);
    object.residual.material.color.set(selected ? 0x9be15d : typeColor.base);
    object.attention.material.color.set(
      object.layer.kind === "full_attention" ? 0xf6bd4b : isRwkvLayer(object.layer) ? 0x5ba0ff : 0x4fd1c5,
    );
    object.mlp.material.color.set(metrics.mlp > 0.58 ? 0xff735c : 0xb18cff);
    object.residual.material.emissiveIntensity = selected ? 0.72 : 0.18 + metrics.residualNorm * 0.38;
    object.attention.material.emissiveIntensity = 0.16 + metrics.attention * 0.44;
    object.mlp.material.emissiveIntensity = 0.14 + metrics.mlp * 0.42;
  });

  sceneState.connectionObjects.forEach((line, index) => {
    const left = step.layers[index] ?? {};
    const right = step.layers[index + 1] ?? {};
    const strength = clamp01(((left.residualNorm ?? 0.2) + (right.residualNorm ?? 0.2)) / 2);
    line.material.opacity = 0.1 + strength * 0.5;
  });

  if (sceneState.activeMarker && sceneState.layerObjects.length) {
    const layerObject = sceneState.layerObjects[activeLayerPosition.index];
    const nextObject = sceneState.layerObjects[Math.min(activeLayerPosition.index + 1, sceneState.layerObjects.length - 1)];
    const from = layerObject.group.position;
    const to = nextObject.group.position;
    sceneState.activeMarker.visible = true;
    sceneState.activeMarker.position.set(
      THREE.MathUtils.lerp(from.x, to.x, activeLayerPosition.t),
      1.58 + Math.sin(elapsed * 3.6) * 0.08,
      THREE.MathUtils.lerp(-0.38, 0.38, activeLayerPosition.t),
    );
    const markerScale = 0.85 + Math.sin(elapsed * 6) * 0.1;
    sceneState.activeMarker.scale.setScalar(markerScale);
  }
}

function updateMlpSceneMetrics(step, elapsed) {
  const objects = sceneState.mlpObjects;
  if (!objects) return;
  sceneState.activeMarker.visible = false;
  const context = getRenderContext();
  const layer = context.trace.architecture.layers[context.layerIndex] ?? {};
  const metrics =
    step.layers[context.layerIndex] ?? synthesizeLayerMetrics(layer, context.layerIndex, context.stepIndex, 17);
  updateMlpObjectsMetrics(objects, metrics, context);
}

function updateComparisonSceneMetrics() {
  sceneState.activeMarker.visible = false;
  for (const objects of sceneState.comparisonObjects) {
    const context = getRenderContext(objects.trace, {
      layerIndex: objects.context.layerIndex,
      stepIndex: state.stepIndex,
    });
    objects.context = context;
    const step = objects.trace.steps[context.stepIndex] ?? objects.trace.steps[0];
    const layer = objects.trace.architecture.layers[context.layerIndex] ?? {};
    const metrics = step.layers[context.layerIndex] ?? synthesizeLayerMetrics(layer, context.layerIndex, context.stepIndex, 17);
    updateMlpObjectsMetrics(objects, metrics, context);
  }
}

function updateMlpObjectsMetrics(objects, metrics, context) {
  for (const stage of objects.stages) {
    const baseLevel = getMlpStageLevel(stage.key, metrics, context);
    const values = stage.nodes.map((node) => getMlpUnitValue(stage.key, node.userData.channelIndex, baseLevel, metrics, context));
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = Math.max(maxValue - minValue, 1e-5);
    const ranks = new Array(values.length);
    values
      .map((value, index) => ({ value, index }))
      .sort((a, b) => a.value - b.value)
      .forEach((item, rank) => {
        ranks[item.index] = values.length > 1 ? rank / (values.length - 1) : 1;
      });
    for (const [nodeIndex, node] of stage.nodes.entries()) {
      const value = values[nodeIndex];
      const normalizedValue = (value - minValue) / valueRange;
      const displayedValue = clamp01(normalizedValue * 0.35 + ranks[nodeIndex] * 0.65);
      node.userData.activation = value;
      node.userData.displayActivation = displayedValue;
      node.scale.setScalar(1);
      node.material.color.copy(getMlpNodeColor(displayedValue));
    }
  }

  let maxContribution = 0;
  for (const line of objects.connections) {
    const fromStage = objects.stages.find((stage) => stage.key === line.userData.fromStage);
    const fromValue = fromStage?.nodes[line.userData.fromIndex]?.userData.activation ?? 0;
    const contribution = fromValue * (line.userData.weight ?? 0);
    line.userData.contribution = contribution;
    maxContribution = Math.max(maxContribution, Math.abs(contribution));
  }
  const contributionScale = maxContribution > 1e-6 ? maxContribution : 1;
  for (const line of objects.connections) {
    const normalized = clamp((line.userData.contribution ?? 0) / contributionScale, -1, 1);
    const magnitude = Math.abs(normalized);
    line.material.opacity = magnitude < 0.35 ? 0.2 : 0.16 + magnitude * 0.42;
    line.material.color.copy(getMlpConnectionColor(normalized));
  }
}

function getMlpStageLevel(stageKey, metrics, context = getRenderContext()) {
  const layer = context.trace?.architecture?.layers?.[context.layerIndex];
  if (isRwkvLayer(layer)) {
    if (stageKey === "input") return clamp01(metrics.residualNorm);
    if (stageKey === "gate") return clamp01(metrics.attention * 0.78 + metrics.residualNorm * 0.16 + metrics.mlp * 0.06);
    if (stageKey === "activation") return clamp01(metrics.mlp * 0.86 + metrics.attention * 0.08 + metrics.entropy * 0.06);
    if (stageKey === "output") return clamp01(metrics.residualNorm * 0.45 + metrics.mlp * 0.35 + metrics.attention * 0.2);
    return clamp01(metrics.mlp);
  }
  if (stageKey === "input") return clamp01(metrics.residualNorm);
  if (stageKey === "gate") return clamp01(metrics.mlp * 0.76 + metrics.residualNorm * 0.18 + metrics.attention * 0.06);
  if (stageKey === "activation") return clamp01(metrics.mlp * 0.86 + metrics.entropy * 0.14);
  if (stageKey === "output") return clamp01(metrics.mlp * 0.58 + metrics.residualNorm * 0.42);
  return clamp01(metrics.mlp);
}

function buildSampledChannels(stage, metrics, context = getRenderContext()) {
  const fullSize = Math.max(1, Math.floor(stage.fullSize ?? stage.count));
  const count = Math.min(stage.count, fullSize);
  if (state.channelSampleMode === "spread") {
    if (count === 1) return [0];
    return Array.from({ length: count }, (_, index) => Math.round((index * (fullSize - 1)) / (count - 1)));
  }
  if (state.channelSampleMode === "seeded") {
    const selected = new Set();
    let cursor = 0;
    while (selected.size < count && cursor < fullSize * 3) {
      const candidate = Math.floor(seededWave((context.layerIndex + 1) * 997 + cursor * 41 + stage.key.length * 17) * fullSize);
      selected.add(candidate);
      cursor += 1;
    }
    return Array.from(selected).slice(0, count).sort((a, b) => a - b);
  }

  const rankingContext = {
    ...context,
    stepIndex: 0,
    inputTokenIndex: 0,
  };
  const baseLevel = getMlpStageLevel(stage.key, metrics, rankingContext);
  return Array.from({ length: fullSize }, (_, channelIndex) => ({
    channelIndex,
    value: getMlpUnitValue(stage.key, channelIndex, baseLevel, metrics, rankingContext),
  }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count)
    .map((item) => item.channelIndex)
    .sort((a, b) => a - b);
}

function getMlpUnitValue(stageKey, unitIndex, baseLevel, metrics, context = getRenderContext()) {
  const layerSeed =
    hashText(context.trace?.model?.name ?? "") +
    (context.layerIndex + 1) * 97 +
    (context.stepIndex + 1) * 53 +
    (context.inputTokenIndex + 1) * 79;
  const stable = seededWave(layerSeed + unitIndex * 11 + stageKey.length * 31);
  const gateBoost = stageKey === "activation" ? metrics.mlp * stable : 0;
  return clamp01(baseLevel * 0.72 + stable * 0.2 + gateBoost * 0.08);
}

function getMlpNodeColor(value) {
  const normalized = clamp01(value);
  const contrast = Math.pow(normalized, 1.35);
  const tonalSteps = [
    0.015, 0.035, 0.06, 0.09, 0.13, 0.18, 0.24, 0.31, 0.39, 0.48, 0.58, 0.69, 0.8, 0.9, 1,
  ];
  const index = clamp(Math.round(contrast * (tonalSteps.length - 1)), 0, tonalSteps.length - 1);
  const brightness = tonalSteps[index];
  return new THREE.Color(brightness, brightness, brightness);
}

function getMlpConnectionColor(normalizedContribution) {
  const magnitude = Math.abs(normalizedContribution);
  if (magnitude < 0.35) return new THREE.Color(0x000000);
  const black = new THREE.Color(0x000000);
  const target = normalizedContribution >= 0 ? new THREE.Color(0x00ff00) : new THREE.Color(0xff0000);
  return black.lerp(target, clamp01((magnitude - 0.35) / 0.65) * 0.85);
}

function getSampledMlpWeight(fromStageKey, toStageKey, fromIndex, toIndex, context = getRenderContext()) {
  const seed =
    hashText(context.trace?.model?.name ?? "") +
    (context.layerIndex + 1) * 1009 +
    fromIndex * 37 +
    toIndex * 67 +
    fromStageKey.length * 17 +
    toStageKey.length * 23;
  const magnitude = 0.18 + seededWave(seed) * 0.82;
  const sign = seededWave(seed + 13) > 0.38 ? 1 : -1;
  return sign * magnitude;
}

function getActiveLayerPosition(elapsed) {
  const layerCount = sceneState.layerObjects.length;
  if (layerCount <= 1) return { index: 0, t: 0 };
  const sweep = (elapsed * 0.42 + state.stepIndex * 0.17) % 1;
  const scaled = sweep * (layerCount - 1);
  const index = Math.min(layerCount - 2, Math.floor(scaled));
  return { index, t: scaled - index };
}

function handleCanvasPick(event) {
  if (!state.trace) return;
  const rect = dom.networkCanvas.getBoundingClientRect();
  sceneState.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  sceneState.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  sceneState.raycaster.setFromCamera(sceneState.pointer, sceneState.camera);
  const meshes = sceneState.layerObjects.flatMap((object) => [object.residual, object.attention, object.mlp]);
  const [hit] = sceneState.raycaster.intersectObjects(meshes, false);
  if (!hit) return;
  const layerIndex = hit.object.userData.layerIndex;
  if (Number.isFinite(layerIndex)) {
    state.selectedLayerIndex = layerIndex;
    renderLayerDetail();
    renderLayerList();
  }
}

function renderModelStats() {
  if (!state.trace) return;
  if (state.viewMode === "compare" && state.comparisonTraces.length >= 2) {
    const [leftTrace, rightTrace] = state.comparisonTraces;
    dom.modelName.textContent = "Qwen3.5 vs RWKV-7";
    dom.layerCount.textContent = `${leftTrace.architecture.layers.length} / ${rightTrace.architecture.layers.length}`;
    dom.hiddenSize.textContent = `${formatNumber(leftTrace.architecture.hiddenSize)} / ${formatNumber(rightTrace.architecture.hiddenSize)}`;
    dom.contextLength.textContent = `${formatNumber(leftTrace.architecture.contextLength)} / ${formatNumber(rightTrace.architecture.contextLength)}`;
    dom.viewMlpButton.textContent = "MLP Only";
    return;
  }
  const { model, architecture } = state.trace;
  dom.modelName.textContent = getModelDisplayName(state.trace) ?? "Unknown model";
  dom.layerCount.textContent = String(architecture.layers.length);
  dom.hiddenSize.textContent = formatNumber(architecture.hiddenSize ?? model.hiddenSize ?? model.hidden_size);
  dom.contextLength.textContent = formatNumber(architecture.contextLength ?? model.contextLength ?? model.context_length);
  dom.viewMlpButton.textContent = isRwkvTrace() ? "RWKV Mix" : "MLP Only";
}

function renderAllPanels() {
  renderStepHeader();
  renderTokens();
  renderLayerList();
  renderLayerDetail();
  renderTopTokens();
}

function renderStepHeader() {
  const step = getActiveStep();
  dom.stepSlider.value = String(state.stepIndex);
  const stepCount = state.viewMode === "compare" ? getComparisonStepCount() : state.trace.steps.length;
  dom.stepLabel.textContent = `Step ${state.stepIndex + 1}/${stepCount}`;
  const token = state.trace.tokens[state.inputTokenIndex] ?? state.trace.tokens[step.activeTokenIndex];
  dom.activeTokenLabel.textContent = token ? visibleTokenText(token.text) : "--";
  dom.generatedTextLabel.textContent = step.generatedToken ? visibleTokenText(step.generatedToken) : step.phase;
  updateInputTokenValue();
}

function renderTokens() {
  const step = getActiveStep();
  dom.tokenStrip.replaceChildren();
  for (const token of state.trace.tokens) {
    const chip = document.createElement("span");
    chip.className = `token-chip ${token.source === "generated" ? "generated" : ""} ${
      token.index === state.inputTokenIndex ? "active" : ""
    }`;
    chip.textContent = visibleTokenText(token.text);
    chip.title = `${token.index}: ${visibleTokenText(token.text)}`;
    chip.addEventListener("click", () => setInputTokenIndex(token.index, { updateStatus: true }));
    dom.tokenStrip.appendChild(chip);
  }
}

function getSelectedBlockLabel(layer) {
  return isRwkvLayer(layer) ? "Mix" : "MLP";
}

function getBlockMetricLabels(layer) {
  if (isRwkvLayer(layer)) {
    return { state: "State", route: "Time Mix", mix: "Channel Mix" };
  }
  return { state: "State", route: "Route", mix: "MLP" };
}

function renderLayerList() {
  const step = getActiveStep();
  dom.layerList.replaceChildren();
  const layers = state.viewMode === "compare" && state.comparisonTraces.length >= 2
    ? state.trace.architecture.layers.slice(
        0,
        Math.min(...state.comparisonTraces.map((trace) => trace.architecture.layers.length)),
      )
    : state.trace.architecture.layers;
  layers.forEach((layer, index) => {
    const metrics = step.layers[index];
    const row = document.createElement("button");
    row.type = "button";
    row.className = `layer-row ${index === state.selectedLayerIndex ? "selected" : ""}`;
    row.addEventListener("click", () => {
      state.selectedLayerIndex = index;
      if (state.viewMode === "mlp" || state.viewMode === "compare") {
        rebuildNetworkScene();
      }
      renderLayerList();
      renderLayerDetail();
      updateSceneMetrics();
    });

    const name = document.createElement("strong");
    name.textContent =
      state.viewMode === "mlp" ? `${index + 1}. ${getSelectedBlockLabel(layer)}` : `${index + 1}. ${shortLayerKind(layer.kind)}`;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    const rowValue = state.viewMode === "mlp" ? metrics?.mlp ?? 0 : metrics?.residualNorm ?? 0;
    fill.style.width = `${Math.round(rowValue * 100)}%`;
    track.appendChild(fill);
    const value = document.createElement("span");
    value.textContent = `${Math.round(rowValue * 100)}%`;

    row.append(name, track, value);
    dom.layerList.appendChild(row);
  });
}

function renderLayerDetail() {
  if (state.viewMode === "compare" && state.comparisonTraces.length >= 2) {
    renderComparisonLayerDetail();
    return;
  }
  const layer = state.trace.architecture.layers[state.selectedLayerIndex];
  const step = getActiveStep();
  const metrics = step.layers[state.selectedLayerIndex];
  dom.selectedLayerBadge.textContent = layer ? `Block ${state.selectedLayerIndex + 1}` : "None";
  if (!layer || !metrics) {
    dom.layerDetail.innerHTML = '<div class="error-message">No layer selected.</div>';
    return;
  }

  if (state.viewMode === "mlp") {
    renderMlpLayerDetail(layer, metrics);
    return;
  }

  const metricLabels = getBlockMetricLabels(layer);
  const headRows = metrics.heads
    .map(
      (head) => `
        <div class="head-row">
          <span>${escapeHtml(head.label ?? `H${head.index + 1}`)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.round(head.value * 100)}%"></div></div>
          <span>${Math.round(head.value * 100)}%</span>
        </div>
      `,
    )
    .join("");

  dom.layerDetail.innerHTML = `
    <div>
      <div class="detail-title">${escapeHtml(layer.name)}</div>
      <div class="detail-subtitle">${escapeHtml(layer.kind)} · ${escapeHtml(metrics.note ?? "")}</div>
    </div>
    <div class="metric-grid">
      ${metricCell(metricLabels.state, metrics.residualNorm)}
      ${metricCell(metricLabels.route, metrics.attention)}
      ${metricCell(metricLabels.mix, metrics.mlp)}
    </div>
    <div class="head-list">${headRows}</div>
  `;
}

function renderComparisonLayerDetail() {
  dom.selectedLayerBadge.textContent = `Block ${state.selectedLayerIndex + 1}`;
  const rows = state.comparisonTraces.slice(0, 2).map((trace) => {
    const context = getRenderContext(trace, {
      layerIndex: Math.min(state.selectedLayerIndex, trace.architecture.layers.length - 1),
      stepIndex: Math.min(state.stepIndex, trace.steps.length - 1),
    });
    const step = trace.steps[context.stepIndex] ?? trace.steps[0];
    const layer = trace.architecture.layers[context.layerIndex] ?? {};
    const metrics = step.layers[context.layerIndex] ?? synthesizeLayerMetrics(layer, context.layerIndex, context.stepIndex, 17);
    const labels = getBlockMetricLabels(layer);
    const efficiency = metrics.efficiency ?? synthesizeMlpEfficiency(layer, metrics, context.layerIndex + context.stepIndex + 17);
    return `
      <div class="mlp-stage-detail">
        <div class="mlp-stage-title">
          <span>${escapeHtml(getComparisonTitle(trace))}</span>
          <strong>${escapeHtml(isRwkvLayer(layer) ? "RWKV" : "FFN")}</strong>
        </div>
        <div class="metric-grid">
          ${metricCell(labels.state, metrics.residualNorm)}
          ${metricCell(labels.route, metrics.attention)}
          ${metricCell(labels.mix, metrics.mlp)}
        </div>
        <div class="efficiency-grid efficiency-grid--compact">
          ${metricValueCell("MFLOPs/token", formatMFlops(efficiency.megaFlopsPerToken))}
          ${metricValueCell("Delta/MFLOP", formatDecimal(efficiency.deltaPerMFlop, 3))}
          ${metricValueCell("Active", `${formatPercent(efficiency.activeFraction)} · ${formatNumber(efficiency.activeNeurons)} units`)}
          ${metricValueCell(`Top-${formatNumber(efficiency.topK)}`, formatPercent(efficiency.topKCoverage))}
        </div>
      </div>
    `;
  }).join("");

  dom.layerDetail.innerHTML = `
    <div>
      <div class="detail-title">Same Prompt Comparison</div>
      <div class="detail-subtitle">Selected block index is matched where both architectures have that block.</div>
    </div>
    <div class="head-list">${rows}</div>
  `;
}

function renderMlpLayerDetail(layer, metrics) {
  const rwkv = isRwkvLayer(layer);
  dom.selectedLayerBadge.textContent = `${rwkv ? "RWKV" : "FFN"} Block ${state.selectedLayerIndex + 1}`;
  const hiddenSize = layer.hiddenSize ?? state.trace.architecture.hiddenSize ?? 1024;
  const intermediateSize = layer.intermediateSize ?? state.trace.architecture.intermediateSize ?? 3584;
  const efficiency = metrics.efficiency ?? synthesizeMlpEfficiency(layer, metrics, state.selectedLayerIndex + state.stepIndex + 17);
  const efficiencyNote = efficiency.estimated
    ? "Active neuron and top-k coverage are estimated unless the trace includes intermediate activations."
    : "Active neuron and top-k coverage come from exported intermediate activations.";
  const rows = ["input", "gate", "activation", "output"]
    .map((stageKey) => {
      const base = getMlpStageLevel(stageKey, metrics);
      const label = getMlpStageLabel(stageKey);
      const stageObject = sceneState.mlpObjects?.stages.find((stage) => stage.key === stageKey);
      const channelSamples = stageObject?.sampledChannels?.slice(0, 6) ?? Array.from({ length: 6 }, (_, index) => index);
      const samples = channelSamples.map((channelIndex) => {
        const unitValue = getMlpUnitValue(stageKey, channelIndex, base, metrics);
        return `
          <div class="head-row">
            <span>${label.prefix}${channelIndex}</span>
            <div class="bar-track"><div class="bar-fill bar-fill--${stageKey}" style="width:${Math.round(
              unitValue * 100,
            )}%"></div></div>
            <span>${Math.round(unitValue * 100)}%</span>
          </div>
        `;
      }).join("");
      return `
        <div class="mlp-stage-detail">
          <div class="mlp-stage-title">
            <span>${label.title}</span>
            <strong>${Math.round(base * 100)}%</strong>
          </div>
          ${samples}
        </div>
      `;
    })
    .join("");

  dom.layerDetail.innerHTML = `
    <div>
      <div class="detail-title">${escapeHtml(layer.name)} ${rwkv ? "RWKV mix" : "FFN"}</div>
      <div class="detail-subtitle">${escapeHtml(getMixDetailSubtitle(layer, hiddenSize, intermediateSize))}</div>
    </div>
    <div class="metric-grid">
      ${metricCell(rwkv ? "State In" : "Input", metrics.residualNorm)}
      ${metricCell(rwkv ? "Time Mix" : "Gate-Up", getMlpStageLevel("gate", metrics))}
      ${metricCell(rwkv ? "State Out" : "Output", getMlpStageLevel("output", metrics))}
    </div>
    <div class="mlp-stage-detail">
      <div class="mlp-stage-title">
        <span>${rwkv ? "Mix Efficiency" : "MLP Efficiency"}</span>
        <strong>${formatMFlops(efficiency.megaFlopsPerToken)} MFLOPs/token</strong>
      </div>
      <div class="efficiency-grid">
        ${metricValueCell("Active Neurons", `${formatNumber(efficiency.activeNeurons)} / ${formatNumber(intermediateSize)}`)}
        ${metricValueCell("Active Fraction", formatPercent(efficiency.activeFraction))}
        ${metricValueCell(`Top-${formatNumber(efficiency.topK)} Coverage`, formatPercent(efficiency.topKCoverage))}
        ${metricValueCell("Delta/MFLOP", formatDecimal(efficiency.deltaPerMFlop, 3))}
      </div>
      <div class="metric-note">${escapeHtml(efficiencyNote)}</div>
    </div>
    <div class="head-list">${rows}</div>
  `;
}

function getMlpStageLabel(stageKey) {
  const layer = state.trace?.architecture?.layers?.[state.selectedLayerIndex];
  if (isRwkvLayer(layer)) {
    if (stageKey === "input") return { title: "State input sample", prefix: "s" };
    if (stageKey === "gate") return { title: "Time-mix sample", prefix: "t" };
    if (stageKey === "activation") return { title: "Channel-mix sample", prefix: "c" };
    return { title: "State output sample", prefix: "o" };
  }
  if (stageKey === "input") return { title: "Residual input sample", prefix: "h" };
  if (stageKey === "gate") return { title: "Gate and up projection sample", prefix: "u" };
  if (stageKey === "activation") return { title: "SiLU-gated intermediate sample", prefix: "a" };
  return { title: "Down projection output sample", prefix: "o" };
}

function getMixDetailSubtitle(layer, hiddenSize, intermediateSize) {
  if (isRwkvLayer(layer)) {
    return `${formatNumber(hiddenSize)} state -> time mix -> ${formatNumber(
      intermediateSize,
    )} channel mix -> ${formatNumber(hiddenSize)} state · strongest sampled links`;
  }
  return `${formatNumber(hiddenSize)} hidden -> ${formatNumber(intermediateSize)} intermediate -> ${formatNumber(
    hiddenSize,
  )} hidden · strongest sampled links`;
}

function metricCell(label, value) {
  return `
    <div class="metric-cell">
      <span>${escapeHtml(label)}</span>
      <strong>${formatPercent(value)}</strong>
    </div>
  `;
}

function metricValueCell(label, value) {
  return `
    <div class="metric-cell">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderTopTokens() {
  const step = getActiveStep();
  dom.topTokens.replaceChildren();
  if (state.viewMode === "compare" && state.comparisonTraces.length >= 2) {
    renderComparisonOutputs();
    return;
  }
  if (state.outputReadout === "channels") {
    const selectedLayer = state.trace.architecture.layers[state.selectedLayerIndex] ?? {};
    dom.outputPanelTitle.textContent = isRwkvLayer(selectedLayer) ? "Output State" : "Output Channels";
    dom.generatedTextLabel.textContent = getSampleModeLabel(state.channelSampleMode);
    const layer = selectedLayer;
    const metrics =
      step.layers[state.selectedLayerIndex] ?? synthesizeLayerMetrics(layer, state.selectedLayerIndex, state.stepIndex, 17);
    const outputStage = sceneState.mlpObjects?.stages.find((stage) => stage.key === "output");
    const channels = outputStage?.sampledChannels?.slice(0, 8) ?? buildSampledChannels(
      { key: "output", count: 8, fullSize: layer.hiddenSize ?? state.trace.architecture.hiddenSize ?? 1024 },
      metrics,
    );
    const base = getMlpStageLevel("output", metrics);
    for (const channelIndex of channels) {
      const value = getMlpUnitValue("output", channelIndex, base, metrics);
      const row = document.createElement("div");
      row.className = "top-token-row";
      const token = document.createElement("span");
      token.className = "token-text";
      token.textContent = `${getMlpStageLabel("output").prefix}${channelIndex}`;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill bar-fill--output";
      fill.style.width = `${Math.max(3, Math.round(value * 100))}%`;
      track.appendChild(fill);
      const amount = document.createElement("span");
      amount.textContent = `${Math.round(value * 100)}%`;
      row.append(token, track, amount);
      dom.topTokens.appendChild(row);
    }
    return;
  }

  dom.outputPanelTitle.textContent = "Next Token";
  dom.generatedTextLabel.textContent = step.generatedToken ? visibleTokenText(step.generatedToken) : step.phase;
  const candidates = step.topTokens?.length ? step.topTokens : makeSyntheticTopTokens(state.stepIndex + 100, state.trace);
  const max = Math.max(...candidates.map((candidate) => candidate.probability), 0.001);
  for (const candidate of candidates.slice(0, 6)) {
    const row = document.createElement("div");
    row.className = "top-token-row";
    const token = document.createElement("span");
    token.className = "token-text";
    token.textContent = visibleTokenText(candidate.token);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(3, Math.round((candidate.probability / max) * 100))}%`;
    track.appendChild(fill);
    const value = document.createElement("span");
    value.textContent = `${Math.round(candidate.probability * 100)}%`;
    row.append(token, track, value);
    dom.topTokens.appendChild(row);
  }
}

function renderComparisonOutputs() {
  dom.outputPanelTitle.textContent = "Compare Outputs";
  dom.generatedTextLabel.textContent = "same prompt";
  for (const trace of state.comparisonTraces.slice(0, 2)) {
    const stepIndex = Math.min(state.stepIndex, trace.steps.length - 1);
    const step = trace.steps[stepIndex] ?? trace.steps[0];
    const candidates = step.topTokens?.length ? step.topTokens.slice(0, 3) : makeSyntheticTopTokens(state.stepIndex + 100, trace).slice(0, 3);
    const header = document.createElement("div");
    header.className = "mlp-stage-title comparison-output-title";
    const name = document.createElement("span");
    name.textContent = getComparisonTitle(trace);
    const generated = document.createElement("strong");
    generated.textContent = step.generatedToken ? visibleTokenText(step.generatedToken) : step.phase;
    header.append(name, generated);
    dom.topTokens.appendChild(header);

    const max = Math.max(...candidates.map((candidate) => candidate.probability), 0.001);
    for (const candidate of candidates) {
      const row = document.createElement("div");
      row.className = "top-token-row";
      const token = document.createElement("span");
      token.className = "token-text";
      token.textContent = visibleTokenText(candidate.token);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${Math.max(3, Math.round((candidate.probability / max) * 100))}%`;
      track.appendChild(fill);
      const value = document.createElement("span");
      value.textContent = `${Math.round(candidate.probability * 100)}%`;
      row.append(token, track, value);
      dom.topTokens.appendChild(row);
    }
  }
}

function setStep(nextIndex, options = {}) {
  if (!state.trace) return;
  const stepCount = state.viewMode === "compare" ? getComparisonStepCount() : state.trace.steps.length;
  const bounded = clamp(Math.round(nextIndex), 0, stepCount - 1);
  if (!options.force && bounded === state.stepIndex && !options.fromSlider) return;
  state.stepIndex = bounded;
  state.inputTokenIndex = getActiveStep().activeTokenIndex;
  syncIoControls();
  if (state.viewMode === "mlp") {
    rebuildTokenObjects();
    rebuildEmbeddingManifold();
  }
  renderAllPanels();
  updateSceneMetrics();
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (state.playing || !state.trace) return;
  state.playing = true;
  dom.playPauseButton.innerHTML = '<i data-lucide="pause" aria-hidden="true"></i>';
  dom.playPauseButton.setAttribute("aria-label", "Pause trace");
  createIcons({ icons });
  state.playTimer = window.setInterval(() => {
    const next = state.stepIndex + 1;
    const stepCount = state.viewMode === "compare" ? getComparisonStepCount() : state.trace.steps.length;
    setStep(next >= stepCount ? 0 : next);
  }, PLAY_INTERVAL_MS);
}

function stopPlayback() {
  if (state.playTimer) {
    window.clearInterval(state.playTimer);
  }
  state.playTimer = null;
  state.playing = false;
  if (dom.playPauseButton) {
    dom.playPauseButton.innerHTML = '<i data-lucide="play" aria-hidden="true"></i>';
    dom.playPauseButton.setAttribute("aria-label", "Play trace");
    createIcons({ icons });
  }
}

async function handleTraceFile(event) {
  const [file] = event.target.files ?? [];
  if (!file) return;
  try {
    const text = await file.text();
    const trace = JSON.parse(text);
    applyTrace(trace, `${file.name} loaded.`, { modelKey: "custom" });
  } catch (error) {
    console.error(error);
    setStatus(`Could not load trace JSON: ${error.message}`, true);
  } finally {
    event.target.value = "";
  }
}

function resizeRenderer() {
  const { renderer, camera } = sceneState;
  if (!renderer || !camera) return;
  const rect = dom.networkCanvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function resetCamera() {
  const { camera, controls } = sceneState;
  if (!camera || !controls) return;
  if (state.viewMode === "compare") {
    camera.position.set(0, 8.6, 28.2);
    controls.target.set(0, -0.15, 0);
  } else if (state.viewMode === "mlp") {
    camera.position.set(0, 8.5, 31.5);
    controls.target.set(0, -0.8, 0);
  } else {
    camera.position.set(0, 7.2, 20);
    controls.target.set(0, -0.18, 0);
  }
  controls.update();
}

function animate() {
  requestAnimationFrame(animate);
  resizeRenderer();
  updateSceneMetrics();
  sceneState.controls?.update();
  sceneState.renderer?.render(sceneState.scene, sceneState.camera);
}

function getActiveStep() {
  return state.trace.steps[state.stepIndex] ?? state.trace.steps[0];
}

function makeTextSprite(text, color = "#f4f1e8") {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(10, 16, 30, 0.9)";
  roundRect(context, 10, 18, 492, 88, 18);
  context.fill();
  context.strokeStyle = "rgba(91, 160, 255, 0.35)";
  context.lineWidth = 3;
  roundRect(context, 10, 18, 492, 88, 18);
  context.stroke();
  context.fillStyle = color;
  context.font = "600 38px Inter, system-ui, sans-serif";
  context.textBaseline = "middle";
  const visible = visibleTokenText(text);
  const clipped = visible.length > 16 ? `${visible.slice(0, 15)}...` : visible;
  context.fillText(clipped, 32, 64, 448);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function clearGroup(group) {
  if (!group) return;
  while (group.children.length) {
    const child = group.children.pop();
    child.traverse?.((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        const disposeMaterial = (material) => {
          if (material.map) material.map.dispose();
          material.dispose();
        };
        if (Array.isArray(object.material)) {
          object.material.forEach(disposeMaterial);
        } else {
          disposeMaterial(object.material);
        }
      }
    });
  }
}

function getLayerColor(kind) {
  if (kind === "full_attention") {
    return { base: 0xf6bd4b, emissive: 0x5a3600 };
  }
  if (String(kind).includes("rwkv")) {
    return { base: 0x5ba0ff, emissive: 0x0b2b6d };
  }
  if (kind === "mlp") {
    return { base: 0xff735c, emissive: 0x5f2018 };
  }
  return { base: 0x4fd1c5, emissive: 0x064641 };
}

function inferHeadCount(kind) {
  if (String(kind).includes("rwkv")) return 12;
  return kind === "full_attention" ? 8 : 16;
}

function shortLayerKind(kind) {
  if (kind === "full_attention") return "Full Attn";
  if (kind === "linear_attention") return "Delta";
  if (String(kind).includes("rwkv")) return "RWKV";
  return kind.replaceAll("_", " ");
}

function tokenizeDisplayText(text) {
  const matches = String(text)
    .replace(/\n/g, " \\n ")
    .match(/\s*[^\s]+|\s+/g);
  return (matches ?? [String(text)]).filter(Boolean).slice(0, 64).map(visibleTokenText);
}

function visibleTokenText(value) {
  const text = String(value ?? "");
  if (text === "\n") return "\\n";
  return text.replace(/\n/g, "\\n").replace(/\t/g, "\\t") || "∅";
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return new Intl.NumberFormat("en-US").format(number);
}

function formatDecimal(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return number.toFixed(digits);
}

function formatMFlops(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (number >= 100) return formatNumber(Math.round(number));
  if (number >= 10) return formatDecimal(number, 1);
  return formatDecimal(number, 2);
}

function formatPercent(value) {
  return `${Math.round(clamp01(value) * 100)}%`;
}

function finiteOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function seededWave(value) {
  const x = Math.sin(value * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
}

function setStatus(message, isError = false) {
  dom.traceStatus.textContent = message;
  dom.traceStatus.parentElement.classList.toggle("error-message", isError);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return clamp(value, 0, 1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function smoothstep(edge0, edge1, value) {
  const t = clamp((value - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
}
