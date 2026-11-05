/* Dub — frontend sem bundler. Usa window.__TAURI__ (withGlobalTauri). */
"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const WHISPER_LABELS = {
  "tiny.en": "Rascunho (tiny, inglês) — muito rápido",
  "base.en": "Rápida (base, inglês)",
  small: "Equilibrada (small)",
  "medium-q5_0": "Boa, compacta (medium q5)",
  medium: "Boa (medium) — padrão",
  "large-v3-turbo-q5_0": "Muito boa, rápida (large-v3-turbo q5)",
  "large-v3": "Máxima (large-v3) — lenta sem GPU",
};
const VOICE_LABELS = {
  "pt_BR-faber-medium": "Faber — masculina, clara (padrão)",
  "pt_BR-edresson-low": "Edresson — masculina, leve",
  "pt_BR-cadu-medium": "Cadu — masculina",
  "pt_BR-jeff-medium": "Jeff — masculina",
};
const KIND_TITLES = { whisper: "Reconhecimento de fala (whisper.cpp)", voices: "Vozes (Piper)", translate: "Tradução (CTranslate2)" };

let settings = null;
let doctor = null;
let lastResult = null;

// ---------- navegação ----------
function show(view) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  if (view === "models") refreshModels();
  if (view === "settings") renderDoctor();
}
$$(".nav-item").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]");
  if (go) show(go.dataset.go);
  const open = e.target.closest("[data-open]");
  if (open) {
    e.preventDefault();
    invoke("open_path", { path: open.dataset.open }).catch(() => {});
  }
});

// ---------- inicialização ----------
async function init() {
  settings = await invoke("get_settings");
  doctor = await invoke("doctor");
  fillSettingsForm();
  fillSelects();
  renderDeviceChip();
  renderReadiness();
  renderDoctor();
  $("#output-dir").value = settings.output_dir || "";
}

function renderDeviceChip() {
  const chip = $("#device-chip");
  const gpu = doctor.device !== "CPU";
  const usable = gpu && doctor.gpu_backends.length > 0;
  chip.textContent = usable ? `${doctor.gpu_name || doctor.device}` : `CPU · ${doctor.cpu_threads} threads${gpu ? " (GPU não usada neste build)" : ""}`;
  chip.classList.toggle("gpu", usable);
  chip.title = `Dispositivos: ${doctor.devices.join(", ")}`;
}

function renderReadiness() {
  // Falta um "tipo" inteiro (nenhum whisper, nenhuma voz ou nenhum tradutor)?
  const kinds = ["whisper", "voices", "translate"];
  const missingKinds = kinds.filter((k) => !doctor.models.some((m) => m.kind === k && m.installed));
  const ready = missingKinds.length === 0;
  $("#not-ready").classList.toggle("hidden", ready);
  $("#start").disabled = !ready;
  const badge = $("#models-badge");
  badge.classList.toggle("hidden", ready);
  badge.textContent = missingKinds.length;
}

function fillSelects() {
  const voices = doctor.models.filter((m) => m.kind === "voices");
  const vs = $("#voice");
  vs.innerHTML = "";
  for (const v of voices) {
    const o = document.createElement("option");
    o.value = v.name;
    o.textContent = (VOICE_LABELS[v.name] || v.name) + (v.installed ? "" : "  (não baixada)");
    o.disabled = !v.installed;
    vs.appendChild(o);
  }
  vs.value = voices.some((v) => v.name === settings.voice && v.installed) ? settings.voice : (voices.find((v) => v.installed) || {}).name || "";

  const ws = $("#whisper");
  ws.innerHTML = "";
  for (const w of doctor.models.filter((m) => m.kind === "whisper")) {
    const o = document.createElement("option");
    o.value = w.name;
    o.textContent = (WHISPER_LABELS[w.name] || w.name) + (w.installed ? "" : "  (não baixado)");
    o.disabled = !w.installed;
    ws.appendChild(o);
  }
  const installedW = doctor.models.filter((m) => m.kind === "whisper" && m.installed).map((m) => m.name);
  ws.value = installedW.includes(settings.whisper_model) ? settings.whisper_model : installedW[0] || "";
}

// ---------- dublar ----------
$("#pick-video").addEventListener("click", async () => {
  const p = await invoke("pick_video");
  if (p) $("#source").value = p;
});
$("#pick-folder").addEventListener("click", async () => {
  const p = await invoke("pick_folder");
  if (p) {
    $("#output-dir").value = p;
    settings.output_dir = p;
    saveSettings();
  }
});
$("#output-dir").addEventListener("change", () => {
  settings.output_dir = $("#output-dir").value.trim() || null;
  saveSettings();
});

// arrastar e soltar (Tauri entrega paths via evento de drag-drop)
listen("tauri://drag-enter", () => document.body.classList.add("dragging"));
listen("tauri://drag-leave", () => document.body.classList.remove("dragging"));
listen("tauri://drag-drop", (ev) => {
  document.body.classList.remove("dragging");
  const paths = ev.payload?.paths || [];
  if (paths.length) {
    $("#source").value = paths[0];
    show("dub");
  }
});

$("#start").addEventListener("click", async () => {
  const source = $("#source").value.trim();
  if (!source) {
    $("#source").focus();
    return;
  }
  settings.voice = $("#voice").value || settings.voice;
  settings.whisper_model = $("#whisper").value || settings.whisper_model;
  await saveSettings();

  resetProgress();
  $("#form-card").classList.add("hidden");
  $("#result-card").classList.add("hidden");
  $("#error-card").classList.add("hidden");
  $("#progress-card").classList.remove("hidden");
  $("#job-title").textContent = source.startsWith("http") ? "Baixando vídeo…" : fileName(source);
  try {
    await invoke("start_dub", {
      args: { source, voice: settings.voice, whisper_model: settings.whisper_model, output_dir: settings.output_dir || null },
    });
  } catch (e) {
    showError(String(e));
  }
});

$("#cancel").addEventListener("click", () => invoke("cancel_dub").catch(() => {}));
$("#new-job").addEventListener("click", backToForm);
$("#error-back").addEventListener("click", backToForm);
$("#open-video").addEventListener("click", () => lastResult && invoke("open_path", { path: lastResult.video }));
$("#reveal-video").addEventListener("click", () => lastResult && invoke("reveal_path", { path: lastResult.video }));
$("#open-srt").addEventListener("click", () => lastResult && invoke("open_path", { path: lastResult.srt }));

function backToForm() {
  $("#progress-card").classList.add("hidden");
  $("#result-card").classList.add("hidden");
  $("#error-card").classList.add("hidden");
  $("#form-card").classList.remove("hidden");
}

function resetProgress() {
  $$("#stages li").forEach((li) => {
    li.className = "";
    li.querySelector(".pct").textContent = "";
    li.querySelector(".bar i").style.width = "0%";
  });
  // etapas opcionais só aparecem se estiverem ligadas
  $("#stages li[data-stage=separate]").classList.toggle("skipped", !settings.separate);
  $("#stages li[data-stage=diarize]").classList.toggle("skipped", !settings.diarize);
  $("#engine-list").innerHTML = "";
  $("#job-status").textContent = "iniciando";
}

function roleLabel(r) {
  return { transcribe: "transcrição", translate: "tradução", synthesize: "voz", separate: "separação" }[r] || r;
}
function deviceLabel(d) {
  return { cuda: "GPU (CUDA)", directml: "GPU (DirectML)", vulkan: "GPU (Vulkan)", metal: "GPU (Metal)", cpu: "CPU" }[d] || d;
}
const STAGE_ORDER = ["separate", "transcribe", "diarize", "translate", "synthesize", "fit", "mix"];

listen("job-event", (ev) => {
  const e = ev.payload;
  switch (e.type) {
    case "status":
      $("#job-status").textContent = e.message + "…";
      break;
    case "device": {
      const ram = e.ram_free_mb ? ` · ${(e.ram_free_mb / 1024).toFixed(1)} GB de RAM livre` : "";
      const plan = (e.plan || [])
        .map((p) => `<li><strong>${roleLabel(p.role)}</strong> → ${deviceLabel(p.device)} <span class="muted">${p.reason}</span></li>`)
        .join("");
      $("#engine-list").innerHTML =
        `<li class="muted">${e.gpu_name || "sem GPU detectada"}${ram} · modo ${e.mode}</li>` + plan;
      break;
    }
    case "opened":
      $("#job-title").textContent = e.title;
      break;
    case "engines":
      $("#engine-list").innerHTML += e.engines
        .map((x) => `<li><strong>${x.role}</strong>: ${x.name} [${x.model}] em ${x.device} <span class="${x.native ? "native" : ""}">${x.native ? "nativa" : "python"}</span></li>`)
        .join("");
      break;
    case "progress": {
      const p = e; // {stage, done, total, message}
      if (p.stage === "done") break;
      const li = $(`#stages li[data-stage=${p.stage}]`);
      if (!li) break;
      // etapas anteriores ficam concluídas
      const idx = STAGE_ORDER.indexOf(p.stage);
      STAGE_ORDER.slice(0, idx).forEach((s) => {
        const prev = $(`#stages li[data-stage=${s}]`);
        if (prev && !prev.classList.contains("skipped")) prev.className = "done";
      });
      const pct = p.total ? Math.min(100, Math.round((100 * p.done) / p.total)) : 0;
      li.classList.add("active");
      li.querySelector(".bar i").style.width = pct + "%";
      li.querySelector(".pct").textContent = p.total > 1 ? `${p.done}/${p.total}` : pct === 100 ? "ok" : "";
      if (p.done >= p.total) li.className = "done";
      $("#job-status").textContent = li.querySelector(".name").textContent.toLowerCase() + (p.message ? " — " + p.message : "");
      break;
    }
    case "done":
      showResult(e);
      break;
    case "error":
      showError(e.message);
      break;
  }
});

function showResult(r) {
  lastResult = r;
  $$("#stages li").forEach((li) => { if (!li.classList.contains("skipped")) li.className = "done"; });
  $("#progress-card").classList.add("hidden");
  $("#result-card").classList.remove("hidden");
  const m = Math.floor(r.elapsed_secs / 60), s = r.elapsed_secs % 60;
  $("#result-summary").textContent = `${r.title} · ${r.segments.length} falas · ${m}m${String(s).padStart(2, "0")}s`;
  $("#result-path").textContent = r.video;
  const warn = $("#result-warn");
  if (r.rushed_segments > 0) {
    warn.textContent = `${r.rushed_segments} fala(s) ficaram aceleradas acima de 1,25× porque a tradução é bem mais longa que o original. Se soar corrido, aumente a "aceleração máxima" em Configurações ou use uma voz mais lenta.`;
    warn.classList.remove("hidden");
  } else warn.classList.add("hidden");
}

function showError(msg) {
  $("#progress-card").classList.add("hidden");
  $("#error-card").classList.remove("hidden");
  $("#error-text").textContent = humanizeError(msg);
}

function humanizeError(msg) {
  if (/cancelado/i.test(msg)) return "Dublagem cancelada.";
  if (/yt-dlp/i.test(msg) && /não encontrad/i.test(msg)) return "Para baixar do YouTube é preciso o yt-dlp. Instale com `pip install yt-dlp` ou use um arquivo local.\n\n" + msg;
  if (/modelo .* não encontrado|não encontrado em/i.test(msg)) return "Um modelo necessário não está instalado. Vá em Modelos e baixe o conjunto padrão.\n\n" + msg;
  return msg;
}

// ---------- modelos ----------
async function refreshModels() {
  doctor = await invoke("doctor");
  renderReadiness();
  fillSelects();
  const list = $("#models-list");
  list.innerHTML = "";
  const inst = doctor.models.find((m) => m.installed && m.path);
  $("#models-path").textContent = inst ? inst.path.replace(/[\\/][^\\/]+[\\/][^\\/]+$/, "") : "~/.cache/dub";
  for (const kind of ["whisper", "translate", "voices"]) {
    const g = document.createElement("div");
    g.className = "model-group";
    g.innerHTML = `<h2>${KIND_TITLES[kind]}</h2>`;
    for (const m of doctor.models.filter((x) => x.kind === kind)) {
      const row = document.createElement("div");
      row.className = "model";
      row.innerHTML = `
        <span class="st ${m.installed ? "ok" : ""}">${m.installed ? "✓" : ""}</span>
        <span class="name">${m.name}${m.default ? " <em class='tag'>padrão</em>" : ""}<small>${m.description}</small></span>
        <span class="size">${m.size_mb} MB</span>
        <button class="btn small" data-dl="${m.name}" ${m.installed ? "disabled" : ""}>${m.installed ? "Instalado" : "Baixar"}</button>`;
      g.appendChild(row);
    }
    list.appendChild(g);
  }
}
$("#refresh-models").addEventListener("click", refreshModels);
$("#dl-defaults").addEventListener("click", () => download([]));
$("#models-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-dl]");
  if (b) download([b.dataset.dl]);
});

async function download(names) {
  $("#dl-status").textContent = "preparando download…";
  $("#dl-progress").classList.remove("hidden");
  $("#dl-bar").style.width = "0%";
  $$("[data-dl]").forEach((b) => (b.disabled = true));
  $("#dl-defaults").disabled = true;
  try {
    await invoke("download_models", { names });
  } catch (e) {
    $("#dl-status").textContent = String(e);
    $("#dl-progress").classList.add("hidden");
    $("#dl-defaults").disabled = false;
    refreshModels();
  }
}
listen("models-progress", (ev) => {
  const p = ev.payload;
  $("#dl-file").textContent = `${p.name} — ${p.file}`;
  const mb = (n) => (n / 1048576).toFixed(0) + " MB";
  $("#dl-bytes").textContent = p.total ? `${mb(p.downloaded)} / ${mb(p.total)}` : mb(p.downloaded);
  $("#dl-bar").style.width = p.total ? Math.round((100 * p.downloaded) / p.total) + "%" : "30%";
  $("#dl-status").textContent = "baixando…";
});
listen("models-done", (ev) => {
  const d = ev.payload;
  $("#dl-status").textContent = d.ok ? "✓ " + d.message : "⚠ " + d.message;
  $("#dl-progress").classList.add("hidden");
  $("#dl-defaults").disabled = false;
  refreshModels();
});

// ---------- configurações ----------
function fillSettingsForm() {
  $("#s-keep-original").checked = settings.keep_original_audio;
  $("#s-subtitles").checked = settings.embed_subtitles;
  $("#s-open-done").checked = settings.open_when_done;
  $("#s-bg").value = settings.background_gain;
  $("#s-speed").value = settings.max_speed;
  $("#s-from").value = settings.source_lang ?? "en";
  $("#s-to").value = settings.target_lang;
  $("#s-cpu").checked = settings.force_cpu;
  $("#s-perf").value = settings.performance || "balanced";
  $("#s-device").value = settings.device_mode || "auto";
  $("#s-separate").checked = settings.separate;
  $("#s-diarize").checked = settings.diarize;
  updateOutputs();
}
function updateOutputs() {
  $("#s-bg-out").value = Math.round($("#s-bg").value * 100) + "%";
  $("#s-speed-out").value = Number($("#s-speed").value).toFixed(2) + "×";
}
function readSettingsForm() {
  settings.keep_original_audio = $("#s-keep-original").checked;
  settings.embed_subtitles = $("#s-subtitles").checked;
  settings.open_when_done = $("#s-open-done").checked;
  settings.background_gain = Number($("#s-bg").value);
  settings.max_speed = Number($("#s-speed").value);
  settings.source_lang = $("#s-from").value;
  settings.target_lang = $("#s-to").value;
  settings.force_cpu = $("#s-cpu").checked;
  settings.performance = $("#s-perf").value;
  settings.device_mode = $("#s-device").value;
  settings.separate = $("#s-separate").checked;
  settings.diarize = $("#s-diarize").checked;
}
$("#view-settings").addEventListener("input", () => {
  updateOutputs();
  readSettingsForm();
  saveSettingsDebounced();
});
let saveTimer = null;
function saveSettingsDebounced() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings, 300);
}
async function saveSettings() {
  try {
    await invoke("save_settings", { settings });
  } catch (e) {
    console.error("save_settings", e);
  }
}

function renderDoctor() {
  if (!doctor) return;
  const el = $("#doctor");
  const row = (t) => `<div class="doc-row"><span class="${t.ok ? "ok" : "bad"}">${t.ok ? "✓" : "✕"}</span><span>${t.name}</span><span class="detail">${t.detail}</span></div>`;
  el.innerHTML =
    `<div class="doc-title">Hardware</div>
     <div class="doc-row"><span class="ok">✓</span><span>dispositivo</span><span class="detail">${doctor.device}${doctor.gpu_name ? " · " + doctor.gpu_name : ""}${doctor.gpu_vram_mb ? " · " + Math.round(doctor.gpu_vram_mb / 1024) + " GB VRAM" : ""} · ${doctor.cpu_threads} threads · disponíveis: ${doctor.devices.join(", ")}</span></div>
     <div class="doc-row"><span class="${doctor.gpu_backends.length ? "ok" : ""}">${doctor.gpu_backends.length ? "✓" : "–"}</span><span>GPU neste build</span><span class="detail">${doctor.gpu_backends.length ? doctor.gpu_backends.join(", ") : "nenhum backend de GPU compilado — tudo roda em CPU"}</span></div>
     <div class="doc-title">Ferramentas</div>${doctor.tools.map(row).join("")}
     <div class="doc-title">Engines nativas</div>${doctor.native.map(row).join("")}
     <div class="ready ${doctor.ready_native ? "yes" : "no"}">${doctor.ready_native ? "Pronto para dublar sem Python." : "Ainda falta algo para o fluxo 100% nativo — veja os itens acima e a aba Modelos."}</div>`;
}
$("#refresh-doctor").addEventListener("click", async () => {
  doctor = await invoke("doctor");
  renderDoctor();
  renderDeviceChip();
  renderReadiness();
});

function fileName(p) {
  return p.split(/[\\/]/).pop();
}

init().catch((e) => {
  console.error(e);
  $("#job-status") && ($("#job-status").textContent = String(e));
});
