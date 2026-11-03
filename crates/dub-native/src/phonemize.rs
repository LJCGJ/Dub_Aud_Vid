//! Fonemização com espeak-ng (compilado estaticamente via `espeak-rs-sys`).
//!
//! Reproduz o que o Piper faz em Python: texto → IPA por cláusula → pontuação
//! preservada → decomposição NFD em codepoints (cada um vira um id do modelo).
//!
//! espeak-ng tem estado global e não é thread-safe: todas as chamadas passam
//! por um mutex único.

use dub_core::{DubError, Result};
use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing::{debug, info};
use unicode_normalization::UnicodeNormalization;

static ESPEAK: OnceLock<Mutex<EspeakState>> = OnceLock::new();

struct EspeakState {
    current_voice: Option<String>,
}

/// Localiza a pasta `espeak-ng-data` (phontab, phondata, dicionários), em ordem:
/// `DUB_ESPEAK_DATA` → ao lado do executável (ou um nível acima, para
/// `target/<profile>/examples`) → pacote Python `piper` → instalação do sistema.
///
/// O `espeak-rs-sys` compila só a biblioteca; os dados vêm separados e são
/// distribuídos com o app (Parte 3). Em desenvolvimento, `pip install piper-tts`
/// já fornece a pasta completa.
pub fn find_espeak_data() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("DUB_ESPEAK_DATA") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("espeak-ng-data"));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("espeak-ng-data"));
            }
        }
    }
    if let Some(p) = python_piper_data() {
        candidates.push(p);
    }
    candidates.extend(
        [
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
            "/usr/share/espeak-ng-data",
            "/opt/homebrew/share/espeak-ng-data",
            "C:\\Program Files\\eSpeak NG\\espeak-ng-data",
        ]
        .iter()
        .map(PathBuf::from),
    );
    candidates.into_iter().find(|p| p.join("phontab").is_file())
}

fn python_piper_data() -> Option<PathBuf> {
    let out = std::process::Command::new(if cfg!(windows) { "python" } else { "python3" })
        .args(["-c", "import piper,os;print(os.path.join(os.path.dirname(piper.__file__),'espeak-ng-data'))"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

fn init() -> Result<&'static Mutex<EspeakState>> {
    if let Some(m) = ESPEAK.get() {
        return Ok(m);
    }
    let data = find_espeak_data().ok_or_else(|| DubError::EngineUnavailable {
        engine: "espeak-ng".into(),
        reason: "pasta espeak-ng-data não encontrada (defina DUB_ESPEAK_DATA)".into(),
    })?;
    info!(path = %data.display(), "inicializando espeak-ng");
    let c_path = CString::new(data.to_string_lossy().as_bytes()).map_err(|e| DubError::Audio(e.to_string()))?;
    // SAFETY: espeak_Initialize só lê o caminho; sem saída de áudio (só fonemas).
    let rc = unsafe {
        espeak_rs_sys::espeak_Initialize(
            espeak_rs_sys::espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_SYNCHRONOUS,
            0,
            c_path.as_ptr(),
            espeak_rs_sys::espeakINITIALIZE_DONT_EXIT as i32,
        )
    };
    if rc < 0 {
        return Err(DubError::EngineUnavailable {
            engine: "espeak-ng".into(),
            reason: format!("espeak_Initialize retornou {rc}"),
        });
    }
    Ok(ESPEAK.get_or_init(|| Mutex::new(EspeakState { current_voice: None })))
}

/// Uma cláusula fonemizada: fonemas IPA + o sinal de pontuação que a encerrou.
#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    pub phonemes: String,
    pub terminator: Option<char>,
    pub end_of_sentence: bool,
}

/// Fonemiza `text` com a voz espeak `voice` (ex.: `pt-br`, `en-us`).
/// Devolve as sentenças, cada uma como lista de "fonemas" (codepoints NFD),
/// no formato que `phonemes_to_ids` do Piper espera.
pub fn phonemize(voice: &str, text: &str) -> Result<Vec<Vec<char>>> {
    let state = init()?;
    let mut st = state.lock().map_err(|_| DubError::ChannelClosed("espeak mutex"))?;

    if st.current_voice.as_deref() != Some(voice) {
        let c_voice = CString::new(voice).map_err(|e| DubError::Audio(e.to_string()))?;
        // SAFETY: string válida por toda a chamada.
        let rc = unsafe { espeak_rs_sys::espeak_SetVoiceByName(c_voice.as_ptr()) };
        if rc != espeak_rs_sys::espeak_ERROR_EE_OK {
            return Err(DubError::EngineFailed {
                engine: "espeak-ng".into(),
                stage: "synthesize",
                message: format!("voz '{voice}' não disponível no espeak-ng"),
            });
        }
        st.current_voice = Some(voice.to_string());
    }

    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for clause in clauses(text)? {
        let mut s = strip_lang_switches(&clause.phonemes);
        if let Some(t) = clause.terminator {
            s.push(t);
            if matches!(t, ',' | ':' | ';') {
                s.push(' ');
            }
        }
        current.extend(s.nfd());
        if clause.end_of_sentence {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    debug!(sentences = sentences.len(), "fonemizado");
    Ok(sentences)
}

/// Chama `espeak_TextToPhonemes` cláusula a cláusula. O espeak avança o
/// ponteiro do texto; comparando com o texto consumido descobrimos a pontuação
/// que encerrou a cláusula (o espeak não devolve isso na API pública).
fn clauses(text: &str) -> Result<Vec<Clause>> {
    let c_text = CString::new(text.replace('\0', " ")).map_err(|e| DubError::Audio(e.to_string()))?;
    let base = c_text.as_ptr() as *const u8;
    let total = c_text.as_bytes().len();
    let mut cursor: *const c_void = c_text.as_ptr() as *const c_void;
    let mut consumed_before = 0usize;
    let mut out = Vec::new();

    loop {
        if cursor.is_null() {
            break;
        }
        // SAFETY: cursor aponta dentro de c_text; espeak atualiza para a próxima cláusula ou NULL.
        let ptr = unsafe {
            espeak_rs_sys::espeak_TextToPhonemes(
                &mut cursor,
                espeak_rs_sys::espeakCHARS_UTF8 as i32,
                espeak_rs_sys::espeakPHONEMES_IPA as i32,
            )
        };
        if ptr.is_null() {
            break;
        }
        // SAFETY: espeak devolve uma string C válida até a próxima chamada.
        let phonemes = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().trim().to_string();

        let consumed_now = if cursor.is_null() { total } else { (cursor as usize).saturating_sub(base as usize).min(total) };
        let chunk = &text.as_bytes()[consumed_before.min(total)..consumed_now];
        let chunk_str = String::from_utf8_lossy(chunk);
        consumed_before = consumed_now;

        let terminator = chunk_str.chars().rev().find(|c| ".!?,;:".contains(*c));
        let end_of_sentence = matches!(terminator, Some('.') | Some('!') | Some('?')) || cursor.is_null();
        if !phonemes.is_empty() {
            out.push(Clause { phonemes, terminator, end_of_sentence });
        }
        if consumed_now >= total {
            break;
        }
    }
    Ok(out)
}

/// Remove marcadores de troca de idioma como "(en)" que o espeak insere.
fn strip_lang_switches(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Converte fonemas em ids do modelo, intercalando PAD como o Piper faz:
/// `^ _ p1 _ p2 _ ... $`.
pub fn phonemes_to_ids(
    phonemes: &[char],
    id_map: &std::collections::HashMap<String, Vec<i64>>,
    phoneme_map: &std::collections::HashMap<String, Vec<String>>,
) -> Vec<i64> {
    let get = |k: &str| id_map.get(k).cloned().unwrap_or_default();
    let pad = get("_");
    let mut ids = Vec::with_capacity(phonemes.len() * 2 + 3);
    ids.extend(get("^"));
    ids.extend(pad.iter().copied());
    for ph in phonemes {
        let key = ph.to_string();
        // phoneme_map do config (ex.: "c" → ["k"]) substitui antes do lookup.
        let mapped: Vec<String> = phoneme_map.get(&key).cloned().unwrap_or_else(|| vec![key.clone()]);
        for m in mapped {
            match id_map.get(&m) {
                Some(v) => {
                    ids.extend(v.iter().copied());
                    ids.extend(pad.iter().copied());
                }
                None => debug!(phoneme = %m, "fonema sem id no modelo; ignorado"),
            }
        }
    }
    ids.extend(get("$"));
    ids
}

pub fn espeak_available() -> bool {
    find_espeak_data().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_lang_switch() {
        assert_eq!(strip_lang_switches("ola (en)hello(pt) mundo"), "ola hello mundo");
    }

    #[test]
    fn ids_have_bos_pad_eos() {
        let mut map = std::collections::HashMap::new();
        for (k, v) in [("_", 0), ("^", 1), ("$", 2), ("a", 14), ("b", 15)] {
            map.insert(k.to_string(), vec![v]);
        }
        let ids = phonemes_to_ids(&['a', 'b'], &map, &Default::default());
        assert_eq!(ids, vec![1, 0, 14, 0, 15, 0, 2]);
    }

    #[test]
    fn phonemizes_portuguese_if_data_present() {
        if !espeak_available() {
            eprintln!("espeak-ng-data ausente; pulando");
            return;
        }
        let s = phonemize("pt-br", "Olá, mundo. Tudo bem?").unwrap();
        assert_eq!(s.len(), 2, "{s:?}");
        let joined: String = s[0].iter().collect();
        assert!(joined.contains(','), "{joined}");
        assert!(joined.ends_with('.'), "{joined}");
    }
}
