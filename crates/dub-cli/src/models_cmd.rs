//! `dub models list|download|path`

use anyhow::{anyhow, Result};
use clap::{Args, Subcommand};
use dub_core::paths::{self, ModelKind};
use dub_models::{InstallStatus, ModelEntry, Progress};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

#[derive(Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    cmd: Option<ModelsCmd>,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Lista o catálogo e o que já está instalado (padrão)
    List,
    /// Baixa modelos. Sem argumentos: o conjunto padrão (whisper medium + voz faber + tradutor en-pt)
    Download {
        /// Nomes do catálogo (ver `dub models list`)
        names: Vec<String>,
        /// Baixa tudo do catálogo
        #[arg(long)]
        all: bool,
    },
    /// Mostra as pastas de modelos (busca e instalação)
    Path,
}

pub fn run(args: ModelsArgs) -> Result<()> {
    match args.cmd.unwrap_or(ModelsCmd::List) {
        ModelsCmd::List => list(),
        ModelsCmd::Download { names, all } => download(names, all),
        ModelsCmd::Path => {
            println!("Busca (em ordem):");
            for r in paths::model_roots() {
                println!("  {}{}", r.display(), if r.is_dir() { "" } else { "  (não existe)" });
            }
            println!("Instalação:");
            for k in ModelKind::all() {
                println!("  {:<10} {}", k.to_string(), paths::install_dir(k).display());
            }
            Ok(())
        }
    }
}

fn list() -> Result<()> {
    for (kind, entries) in dub_models::download::summary() {
        println!("{}", kind_title(kind));
        for (m, st) in entries {
            let mark = match st {
                InstallStatus::Installed(_) => "✔",
                InstallStatus::Missing => " ",
            };
            let def = if m.default { " (padrão)" } else { "" };
            println!("  {mark} {:<36} {:>5} MB  {}{def}", m.name, m.total_mb(), m.description);
        }
        println!();
    }
    println!("Baixar: dub models download [nomes...]   |   tudo que é padrão: dub models download");
    Ok(())
}

fn kind_title(k: ModelKind) -> &'static str {
    match k {
        ModelKind::Whisper => "Transcrição (whisper.cpp, ggml)",
        ModelKind::Voice => "Vozes (Piper)",
        ModelKind::Translate => "Tradução (CTranslate2)",
        ModelKind::Separate => "Separação voz/instrumental (MDX-Net)",
        ModelKind::Kokoro => "Vozes Kokoro (feminina pf_dora, masculinas pm_alex/pm_santa)",
    }
}

fn download(names: Vec<String>, all: bool) -> Result<()> {
    let entries: Vec<ModelEntry> = if all {
        dub_models::catalog()
    } else if names.is_empty() {
        dub_models::defaults()
    } else {
        names
            .iter()
            .map(|n| dub_models::find(n).ok_or_else(|| anyhow!("'{n}' não está no catálogo (veja: dub models list)")))
            .collect::<Result<Vec<_>>>()?
    };

    let total_mb: u32 = entries.iter().filter(|m| dub_models::status(m) == InstallStatus::Missing).map(|m| m.total_mb()).sum();
    if total_mb > 0 {
        eprintln!(
            "▶ baixando ~{total_mb} MB para {}",
            paths::install_dir(ModelKind::Whisper).parent().map(|p| p.display().to_string()).unwrap_or_default()
        );
    }

    let style = ProgressStyle::with_template("{prefix:<34} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} {eta}")
        .unwrap()
        .progress_chars("=> ");
    let mut bars: HashMap<String, ProgressBar> = HashMap::new();

    for m in &entries {
        if let InstallStatus::Installed(p) = dub_models::status(m) {
            eprintln!("  ✔ {:<34} já instalado ({})", m.name, p.display());
            continue;
        }
        let path = dub_models::install(m, |p: Progress| {
            let bar = bars.entry(p.file.clone()).or_insert_with(|| {
                let b = ProgressBar::new(p.total.unwrap_or(0));
                b.set_style(style.clone());
                b.set_prefix(p.file.clone());
                b
            });
            if let Some(t) = p.total {
                bar.set_length(t);
            }
            bar.set_position(p.downloaded);
            if p.total.map(|t| p.downloaded >= t).unwrap_or(false) {
                bar.finish();
            }
        })
        .map_err(|e| anyhow!("{}: {e}", m.name))?;
        eprintln!("  ✔ {:<34} → {}", m.name, path.display());
    }
    eprintln!("\nPronto. Confira com: dub doctor");
    Ok(())
}
