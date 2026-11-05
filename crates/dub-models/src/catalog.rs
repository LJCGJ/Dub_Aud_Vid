//! Catálogo de modelos conhecidos. Cada entrada sabe de onde baixar e como
//! verificar se já está instalada.

use dub_core::paths::ModelKind;
use serde::{Deserialize, Serialize};

/// Release do próprio repositório onde ficam modelos convertidos por nós
/// (tradutores CTranslate2). Gerado pelo workflow `models.yml`.
pub const MODELS_RELEASE: &str = "https://github.com/LJCGJ/Dub_Aud_Vid/releases/download/models-v1";
const HF_WHISPER: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const HF_PIPER: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSpec {
    pub url: String,
    /// Nome final no disco (relativo à pasta do tipo).
    pub name: String,
    /// Tamanho aproximado em MB, só para mostrar ao usuário.
    pub size_mb: u32,
    /// Se `true`, é um .tar.gz extraído em `<pasta do tipo>/<nome da entrada>/`.
    #[serde(default)]
    pub archive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub kind: ModelKind,
    /// Identificador usado em `--whisper-model`, `--voice`, `--translator-model`.
    pub name: String,
    pub description: String,
    pub files: Vec<FileSpec>,
    /// Faz parte do conjunto padrão (`dub models download` sem argumentos).
    #[serde(default)]
    pub default: bool,
}

impl ModelEntry {
    pub fn total_mb(&self) -> u32 {
        self.files.iter().map(|f| f.size_mb).sum()
    }
}

fn whisper(name: &str, size_mb: u32, desc: &str, default: bool) -> ModelEntry {
    ModelEntry {
        kind: ModelKind::Whisper,
        name: name.into(),
        description: desc.into(),
        files: vec![FileSpec {
            url: format!("{HF_WHISPER}/ggml-{name}.bin"),
            name: format!("ggml-{name}.bin"),
            size_mb,
            archive: false,
        }],
        default,
    }
}

fn piper(lang: &str, region_lang: &str, speaker: &str, quality: &str, size_mb: u32, desc: &str, default: bool) -> ModelEntry {
    let name = format!("{region_lang}-{speaker}-{quality}");
    let base = format!("{HF_PIPER}/{lang}/{region_lang}/{speaker}/{quality}/{name}");
    ModelEntry {
        kind: ModelKind::Voice,
        name: name.clone(),
        description: desc.into(),
        files: vec![
            FileSpec { url: format!("{base}.onnx"), name: format!("{name}.onnx"), size_mb, archive: false },
            FileSpec { url: format!("{base}.onnx.json"), name: format!("{name}.onnx.json"), size_mb: 0, archive: false },
        ],
        default,
    }
}

fn translator(name: &str, size_mb: u32, desc: &str, default: bool) -> ModelEntry {
    ModelEntry {
        kind: ModelKind::Translate,
        name: name.into(),
        description: desc.into(),
        files: vec![FileSpec {
            url: format!("{MODELS_RELEASE}/{name}.tar.gz"),
            name: format!("{name}.tar.gz"),
            size_mb,
            archive: true,
        }],
        default,
    }
}

const UVR_RELEASE: &str = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models";

fn separator(name: &str, size_mb: u32, desc: &str, default: bool) -> ModelEntry {
    ModelEntry {
        kind: ModelKind::Separate,
        name: name.into(),
        description: desc.into(),
        files: vec![FileSpec {
            url: format!("{UVR_RELEASE}/{name}.onnx"),
            name: format!("{name}.onnx"),
            size_mb,
            archive: false,
        }],
        default,
    }
}

/// Catálogo embutido. Curto de propósito: só o que faz sentido para dublagem
/// EN → PT-BR. Modelos fora daqui continuam funcionando por caminho.
pub fn catalog() -> Vec<ModelEntry> {
    vec![
        // --- whisper (ggml) ---------------------------------------------------
        whisper("tiny.en", 75, "Inglês, muito rápido, qualidade baixa — só para testes", false),
        whisper("base.en", 142, "Inglês, rápido, qualidade razoável", false),
        whisper("small", 466, "Multilíngue, bom equilíbrio em CPU", false),
        whisper("medium", 1500, "Multilíngue, boa qualidade", true),
        whisper("medium-q5_0", 514, "medium quantizado: quase a mesma qualidade, 1/3 do tamanho", false),
        whisper("large-v3", 2950, "Melhor qualidade; precisa de GPU ou paciência", false),
        whisper("large-v3-turbo-q5_0", 547, "large-v3-turbo quantizado: rápido e muito bom", false),
        // --- vozes Piper PT-BR ------------------------------------------------
        piper("pt", "pt_BR", "faber", "medium", 61, "Voz masculina, clara, 22 kHz", true),
        piper("pt", "pt_BR", "edresson", "low", 61, "Voz masculina, 16 kHz, mais leve", false),
        piper("pt", "pt_BR", "cadu", "medium", 61, "Voz masculina alternativa", false),
        piper("pt", "pt_BR", "jeff", "medium", 61, "Voz masculina alternativa", false),
        // --- tradutores CTranslate2 (convertidos pelo workflow models.yml) -----
        translator("opus-mt-tc-big-en-pt-ct2-int8", 240, "Inglês → Português (Marian tc-big), int8", true),
        translator("opus-mt-en-ROMANCE-ct2-int8", 80, "Inglês → línguas românicas (menor, qualidade menor)", false),
        // --- separação voz/instrumental (MDX-Net do Ultimate Vocal Remover) ---
        separator("Kim_Vocal_2", 64, "Isola a voz; música e efeitos ficam intactos (recomendado)", true),
        separator("UVR-MDX-NET-Inst_HQ_3", 64, "Alternativa focada no instrumental", false),
    ]
}

pub fn find(name: &str) -> Option<ModelEntry> {
    catalog().into_iter().find(|m| m.name == name)
}

pub fn defaults() -> Vec<ModelEntry> {
    catalog().into_iter().filter(|m| m.default).collect()
}
