//! Assistente interativo: `dub` sem argumentos (ou `dub video` sem fonte).
//!
//! Menus navegáveis com as setas do teclado, para quem não quer decorar
//! flags. Tudo que o assistente pergunta tem a flag equivalente em `dub video`,
//! e no fim ele mostra o comando completo para repetir sem perguntas.

use anyhow::{bail, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use dub_core::{DeviceMode, Performance, Role};
use dub_models::InstallStatus;
use std::path::PathBuf;

/// O que o usuário escolheu.
pub struct Choices {
    pub source: String,
    pub device: DeviceMode,
    pub performance: Performance,
    pub whisper_model: String,
    pub voice: String,
    pub out: Option<PathBuf>,
}

pub fn run(initial_source: Option<String>) -> Result<Choices> {
    let theme = ColorfulTheme::default();
    if !dialoguer::console::user_attended() {
        bail!("sem terminal interativo: passe a fonte e as opções por argumentos (veja `dub video --help`)");
    }

    eprintln!("Dub — dublagem automática (EN → PT-BR)");
    eprintln!("Use ↑/↓ para escolher e Enter para confirmar. Ctrl+C cancela.\n");

    // Hardware, para as opções fazerem sentido na máquina do usuário.
    let doc = dub_runtime::doctor();
    let gpu = doc.gpu_name.clone().unwrap_or_else(|| "nenhuma GPU detectada".into());
    eprintln!(
        "Hardware: {gpu}{}{} · {} threads de CPU{}",
        if doc.gpu_integrated { " (integrada)" } else { "" },
        doc.gpu_vram_mb.map(|v| format!(", {} MB VRAM", v)).unwrap_or_default(),
        doc.cpu_threads,
        doc.ram_free_mb.map(|m| format!(" · {:.1} GB de RAM livre", m as f64 / 1024.0)).unwrap_or_default()
    );
    if doc.gpu_backends.is_empty() {
        eprintln!("Este build não tem backend de GPU: tudo roda na CPU.");
    } else {
        eprintln!("GPU neste build: {}", doc.gpu_backends.join(", "));
    }
    eprintln!();

    // 1. Fonte -----------------------------------------------------------------
    let source: String = match initial_source {
        Some(s) => s,
        None => Input::with_theme(&theme)
            .with_prompt("Vídeo (URL do YouTube ou caminho de um arquivo)")
            .validate_with(|s: &String| -> Result<(), String> {
                let s = s.trim();
                if s.is_empty() {
                    return Err("informe uma URL ou um arquivo".into());
                }
                if dub_media::YtDlp::is_url(s) || std::path::Path::new(s.trim_matches('"')).is_file() {
                    Ok(())
                } else {
                    Err("não é uma URL nem um arquivo existente".into())
                }
            })
            .interact_text()?
            .trim()
            .trim_matches('"')
            .to_string(),
    };

    // 2. Onde processar --------------------------------------------------------
    let auto_plan = doc.plan.iter().map(|p| format!("{} na {}", short_role(p.role), p.device)).collect::<Vec<_>>().join(", ");
    let has_gpu = !doc.gpu_backends.is_empty() && doc.devices.iter().any(|d| d != "CPU");
    let device_items: Vec<(DeviceMode, String)> = if has_gpu {
        vec![
            (DeviceMode::Auto, format!("Automático — decide pelo hardware ({auto_plan})")),
            (DeviceMode::Split, "Dividir — transcrição na CPU e voz na GPU, em paralelo".into()),
            (DeviceMode::Gpu, "Só GPU — tudo que tiver backend vai para a GPU".into()),
            (DeviceMode::Cpu, "Só CPU — ignora a GPU".into()),
        ]
    } else {
        vec![(DeviceMode::Cpu, "CPU (não há GPU utilizável neste build/máquina)".into())]
    };
    let device = pick(&theme, "Onde processar", &device_items, 0)?;

    // 3. Quanto do hardware usar --------------------------------------------------
    let perf_items = vec![
        (Performance::Balanced, "Equilibrado — deixa um núcleo livre; o PC continua usável (padrão)".to_string()),
        (Performance::Eco, "Econômico — metade dos núcleos, sem GPU; para notebook ou máquina compartilhada".to_string()),
        (Performance::Max, "Máximo — todos os núcleos e GPU sempre que possível".to_string()),
    ];
    let performance = pick(&theme, "Uso do hardware", &perf_items, 0)?;

    // 4. Modelo do whisper (só os instalados; oferece baixar) -------------------
    let whisper_model = pick_model(&theme, dub_core::paths::ModelKind::Whisper, "Modelo de transcrição (whisper)", "medium")?;

    // 5. Voz ----------------------------------------------------------------------
    let voice = pick_voice(&theme)?;

    // 6. Saída --------------------------------------------------------------------
    let default_out = default_output_name(&source);
    let out: String = Input::with_theme(&theme).with_prompt("Arquivo de saída (.mp4)").default(default_out).interact_text()?;
    let out = PathBuf::from(out.trim().trim_matches('"'));

    // Resumo + comando equivalente ---------------------------------------------
    eprintln!();
    eprintln!("Comando equivalente (para repetir sem perguntas):");
    eprintln!(
        "  dub video \"{}\" --device {} --performance {} --whisper-model {} --voice {} -o \"{}\"",
        source,
        device,
        perf_name(performance),
        whisper_model,
        voice,
        out.display()
    );
    eprintln!();
    if !Confirm::with_theme(&theme).with_prompt("Começar a dublagem?").default(true).interact()? {
        bail!("cancelado");
    }

    Ok(Choices { source, device, performance, whisper_model, voice, out: Some(out) })
}

fn pick<T: Copy>(theme: &ColorfulTheme, prompt: &str, items: &[(T, String)], default: usize) -> Result<T> {
    let labels: Vec<&str> = items.iter().map(|(_, l)| l.as_str()).collect();
    let i = Select::with_theme(theme).with_prompt(prompt).items(&labels).default(default).interact()?;
    Ok(items[i].0)
}

/// Lista o catálogo daquele tipo marcando o que já está instalado; se o usuário
/// escolher um que falta, baixa na hora.
fn pick_model(theme: &ColorfulTheme, kind: dub_core::paths::ModelKind, prompt: &str, preferred: &str) -> Result<String> {
    let entries: Vec<(dub_models::ModelEntry, InstallStatus)> =
        dub_models::summary().into_iter().find(|(k, _)| *k == kind).map(|(_, v)| v).unwrap_or_default();
    if entries.is_empty() {
        return Ok(preferred.to_string());
    }
    let labels: Vec<String> = entries
        .iter()
        .map(|(m, st)| {
            let mark = match st {
                InstallStatus::Installed(_) => "✔",
                InstallStatus::Missing => " ",
            };
            let extra = match st {
                InstallStatus::Installed(_) => String::new(),
                InstallStatus::Missing => format!("  [baixar {} MB]", m.total_mb()),
            };
            format!("{mark} {:<28} {}{extra}", m.name, m.description)
        })
        .collect();
    // Padrão: o preferido se instalado; senão o primeiro instalado; senão o preferido.
    let default = entries
        .iter()
        .position(|(m, st)| m.name == preferred && matches!(st, InstallStatus::Installed(_)))
        .or_else(|| entries.iter().position(|(_, st)| matches!(st, InstallStatus::Installed(_))))
        .or_else(|| entries.iter().position(|(m, _)| m.name == preferred))
        .unwrap_or(0);
    let i = Select::with_theme(theme).with_prompt(prompt).items(&labels).default(default).interact()?;
    let (entry, status) = &entries[i];
    if matches!(status, InstallStatus::Missing) {
        download_entry(entry)?;
    }
    Ok(entry.name.clone())
}

/// Menu de vozes: automática (masculina/feminina por trecho) primeiro, depois
/// Kokoro e Piper com ♀/♂, marcando instaladas e baixando o que faltar.
fn pick_voice(theme: &ColorfulTheme) -> Result<String> {
    let voices = dub_runtime::voices::list();
    let labels: Vec<String> = voices
        .iter()
        .map(|v| {
            let mark = match v.gender.as_str() {
                "feminina" => "♀",
                "masculina" => "♂",
                _ => "⚥",
            };
            let st = if v.installed { "✔" } else { " " };
            let extra = if v.installed { String::new() } else { format!("  [baixar {} MB]", v.download_mb) };
            let rec = if v.name == dub_runtime::voices::AUTO_VOICE { " (recomendado)" } else { "" };
            format!("{st} {mark} {:<20}{rec} {}{extra}", v.name, v.description)
        })
        .collect();
    let i = Select::with_theme(theme).with_prompt("Voz da dublagem").items(&labels).default(0).interact()?;
    let chosen = &voices[i];
    // "auto" precisa das duas vozes padrão; uma voz específica precisa dela mesma.
    let needed: Vec<&str> = if chosen.name == dub_runtime::voices::AUTO_VOICE {
        vec![dub_runtime::voices::DEFAULT_MALE_VOICE, dub_runtime::voices::DEFAULT_FEMALE_VOICE]
    } else {
        vec![chosen.name.as_str()]
    };
    for name in needed {
        let installed = voices.iter().any(|v| v.name == name && v.installed);
        if installed {
            continue;
        }
        if let Some(entry) = dub_runtime::voices::catalog_entry_for(name) {
            download_entry(&entry)?;
        }
    }
    Ok(chosen.name.clone())
}

fn download_entry(entry: &dub_models::ModelEntry) -> Result<()> {
    eprintln!("▶ baixando {} ({} MB)…", entry.name, entry.total_mb());
    let bar = indicatif::ProgressBar::new(0);
    bar.set_style(
        indicatif::ProgressStyle::with_template("{msg:<28} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec}")
            .unwrap()
            .progress_chars("=> "),
    );
    dub_models::install(entry, |p| {
        if let Some(t) = p.total {
            bar.set_length(t);
        }
        bar.set_position(p.downloaded);
        bar.set_message(p.file.clone());
    })?;
    bar.finish_and_clear();
    Ok(())
}

fn short_role(r: Role) -> &'static str {
    match r {
        Role::Transcribe => "transcrição",
        Role::Translate => "tradução",
        Role::Synthesize => "voz",
        Role::Separate => "separação",
    }
}

fn perf_name(p: Performance) -> &'static str {
    match p {
        Performance::Eco => "eco",
        Performance::Balanced => "balanced",
        Performance::Max => "max",
    }
}

fn default_output_name(source: &str) -> String {
    if dub_media::YtDlp::is_url(source) {
        "dublado.mp4".to_string()
    } else {
        let p = std::path::Path::new(source);
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
        p.with_file_name(format!("{stem}.dub.pt-BR.mp4")).to_string_lossy().into_owned()
    }
}
