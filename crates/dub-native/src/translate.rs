//! Tradução nativa com CTranslate2 (`ct2rs`), rodando modelos Marian/OPUS-MT
//! convertidos para o formato CT2 (int8). ~240 MB para en→pt, rápido em CPU.
//!
//! Um modelo é uma **pasta** com `model.bin`, `config.json`, `source.spm` e
//! `target.spm` (SentencePiece). O workflow `models.yml` do repositório gera
//! essas pastas a partir dos modelos do Hugging Face; `dub models download`
//! as instala em `translate/<nome>/`.

use dub_core::engines::{Engine, EngineInfo, Translator};
use dub_core::paths::{self, ModelKind};
use dub_core::{Device, DubError, Lang, Result};
use std::path::{Path, PathBuf};
use tracing::info;

pub struct Ct2Translator {
    inner: ct2rs::Translator<ct2rs::tokenizers::auto::Tokenizer>,
    model_name: String,
    device: Device,
    /// Prefixo de variante exigido por modelos multi-alvo (ex.: `>>por_BR<<`).
    target_prefix: Option<String>,
}

impl Ct2Translator {
    /// `model` é o nome de uma pasta em `translate/` (ex.: `opus-mt-tc-big-en-pt-ct2-int8`)
    /// ou um caminho para a pasta do modelo.
    pub fn load(model: &str, device: Device, threads: usize) -> Result<Self> {
        let dir = resolve_model_dir(model)?;
        let device = effective_device(device);
        let cfg = ct2rs::Config {
            device: if device == Device::Cuda { ct2rs::Device::CUDA } else { ct2rs::Device::CPU },
            compute_type: if device == Device::Cuda { ct2rs::ComputeType::INT8_FLOAT16 } else { ct2rs::ComputeType::INT8 },
            num_threads_per_replica: threads.max(1),
            ..Default::default()
        };
        info!(model = %dir.display(), %device, threads, "carregando CTranslate2");
        let inner = ct2rs::Translator::new(&dir, &cfg).map_err(|e| DubError::EngineUnavailable {
            engine: "ctranslate2".into(),
            reason: format!("{e} ({})", dir.display()),
        })?;
        let model_name = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| model.into());
        let target_prefix = read_target_prefix(&dir);
        Ok(Self { inner, model_name, device, target_prefix })
    }

    pub fn model_available(model: &str) -> bool {
        resolve_model_dir(model).is_ok()
    }
}

/// CTranslate2 só acelera em CUDA; qualquer outro dispositivo vira CPU.
pub fn effective_device(requested: Device) -> Device {
    match requested {
        Device::Cuda if cfg!(feature = "ct2-cuda") => Device::Cuda,
        _ => {
            if requested == Device::Cuda {
                info!("CTranslate2 compilado sem CUDA; tradução em CPU (feature ct2-cuda)");
            }
            Device::Cpu
        }
    }
}

/// Lê `dub.json` opcional na pasta do modelo (gerado pelo conversor) para
/// saber se o modelo precisa de prefixo de idioma-alvo.
fn read_target_prefix(dir: &Path) -> Option<String> {
    let meta = std::fs::read(dir.join("dub.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&meta).ok()?;
    v.get("target_prefix").and_then(|p| p.as_str()).map(str::to_string)
}

fn resolve_model_dir(model: &str) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.is_dir() && p.join("model.bin").exists() {
        return Ok(p.to_path_buf());
    }
    paths::find(ModelKind::Translate, model).filter(|d| d.join("model.bin").exists()).ok_or_else(|| DubError::EngineUnavailable {
        engine: "ctranslate2".into(),
        reason: format!(
            "modelo de tradução '{model}' não encontrado em {}. Rode: dub models download {model}",
            paths::model_dirs(ModelKind::Translate).iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
        ),
    })
}

impl Engine for Ct2Translator {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "ctranslate2".into(), model: self.model_name.clone(), device: self.device.to_string() }
    }
}

impl Translator for Ct2Translator {
    fn translate(&self, texts: &[&str], _from: &Lang, _to: &Lang) -> Result<Vec<String>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // Linhas vazias não vão ao modelo (ele alucina em entrada vazia).
        let idx: Vec<usize> = texts.iter().enumerate().filter(|(_, t)| !t.trim().is_empty()).map(|(i, _)| i).collect();
        let inputs: Vec<&str> = idx.iter().map(|&i| texts[i].trim()).collect();

        let opts = ct2rs::TranslationOptions {
            beam_size: 4,
            max_decoding_length: 256,
            repetition_penalty: 1.1,
            disable_unk: true,
            ..Default::default()
        };
        // O prefixo de variante (">>por_BR<<") é um token do vocabulário: entra
        // como token na frente da fonte, nunca como texto (o SentencePiece o
        // quebraria em pedaços e o modelo os ecoaria na saída).
        let results = match &self.target_prefix {
            Some(p) => {
                let prefixes: Vec<Vec<&str>> = inputs.iter().map(|_| vec![p.as_str()]).collect();
                self.inner.translate_batch_with_source_prefix(&inputs, &prefixes, &opts, None)
            }
            None => self.inner.translate_batch(&inputs, &opts, None),
        }
        .map_err(|e| DubError::EngineFailed {
            engine: "ctranslate2".into(),
            stage: "translate",
            message: e.to_string(),
        })?;

        let mut out = vec![String::new(); texts.len()];
        for (k, (text, _score)) in results.into_iter().enumerate() {
            out[idx[k]] = clean_output(&text);
        }
        Ok(out)
    }
}

/// Modelos OPUS-MT treinados em legendas às vezes emitem marcador de diálogo
/// ("- Vamos começar.") ou espaços antes da pontuação; limpamos.
fn clean_output(s: &str) -> String {
    let mut t = s.trim();
    for marker in ["- ", "– ", "— "] {
        if let Some(rest) = t.strip_prefix(marker) {
            t = rest.trim_start();
        }
    }
    t.replace('⁇', "").replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_dialogue_marker() {
        assert_eq!(clean_output("- Vamos começar."), "Vamos começar.");
        assert_eq!(clean_output("Olá , mundo ."), "Olá, mundo.");
        assert_eq!(clean_output("  Tudo bem?  "), "Tudo bem?");
    }
}
