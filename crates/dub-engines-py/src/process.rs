use crate::python::PythonEnv;
use dub_core::{Device, DubError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use tracing::{debug, error, info, warn};

/// Como subir uma engine Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PySpec {
    /// Nome no registry (`whisper`, `marian`, `piper`...).
    pub engine: String,
    pub model: Option<String>,
    pub device: Option<Device>,
    pub threads: Option<usize>,
    /// Opções extras (`--opt chave=valor`).
    #[serde(default)]
    pub options: Vec<(String, String)>,
}

impl PySpec {
    pub fn new(engine: &str) -> Self {
        Self { engine: engine.into(), model: None, device: None, threads: None, options: vec![] }
    }
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }
    pub fn opt(mut self, k: &str, v: &str) -> Self {
        self.options.push((k.into(), v.into()));
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PyInfo {
    pub name: String,
    pub model: String,
    pub device: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Deserialize)]
struct Ready {
    ready: bool,
    info: Option<PyInfo>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Response {
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<Value>,
    error: Option<String>,
}

struct Io {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Um processo Python rodando uma engine. Thread-safe: as chamadas são
/// serializadas por um mutex (o protocolo é síncrono, uma requisição por vez).
pub struct PyProcess {
    child: Mutex<Child>,
    io: Mutex<Io>,
    next_id: AtomicU64,
    pub info: PyInfo,
    pub spec: PySpec,
}

impl PyProcess {
    pub fn spawn(env: &PythonEnv, spec: PySpec) -> Result<Self> {
        let mut cmd = Command::new(&env.interpreter);
        cmd.arg("-m").arg("dub_engines").arg(&spec.engine);
        if let Some(m) = &spec.model {
            cmd.arg("--model").arg(m);
        }
        if let Some(d) = spec.device {
            cmd.arg("--device").arg(d.torch_name());
        }
        if let Some(t) = spec.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        for (k, v) in &spec.options {
            cmd.arg("--opt").arg(format!("{k}={v}"));
        }
        // PYTHONPATH: prepend a pasta do pacote, preservando a existente.
        let mut pp = env.package_dir.as_os_str().to_os_string();
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            pp.push(if cfg!(windows) { ";" } else { ":" });
            pp.push(existing);
        }
        cmd.env("PYTHONPATH", pp).env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

        info!(engine = %spec.engine, model = ?spec.model, "iniciando engine Python");
        let mut child = cmd.spawn().map_err(|e| DubError::EngineUnavailable {
            engine: spec.engine.clone(),
            reason: format!("não foi possível executar {}: {e}", env.interpreter.display()),
        })?;

        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let stderr = child.stderr.take().expect("stderr");

        // Encaminha stderr do Python para o tracing.
        let tag = spec.engine.clone();
        thread::Builder::new().name(format!("py-stderr-{tag}")).spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
                if line.contains("ERROR") || line.contains("Traceback") {
                    error!(engine = %tag, "{line}");
                } else if line.contains("WARNING") {
                    warn!(engine = %tag, "{line}");
                } else {
                    debug!(engine = %tag, "{line}");
                }
            }
        })?;

        let mut io = Io { stdin, stdout };

        // Espera o `ready`.
        let mut line = String::new();
        io.stdout.read_line(&mut line)?;
        if line.trim().is_empty() {
            let _ = child.kill();
            return Err(DubError::EngineUnavailable {
                engine: spec.engine.clone(),
                reason: "processo terminou sem anunciar ready (veja o log)".into(),
            });
        }
        let ready: Ready = serde_json::from_str(&line)?;
        if !ready.ready {
            let _ = child.kill();
            return Err(DubError::EngineUnavailable {
                engine: spec.engine.clone(),
                reason: ready.error.unwrap_or_else(|| "erro desconhecido ao carregar".into()),
            });
        }
        let info = ready
            .info
            .ok_or_else(|| DubError::EngineUnavailable { engine: spec.engine.clone(), reason: "ready sem info".into() })?;
        info!(engine = %info.name, model = %info.model, device = %info.device, "engine pronta");

        Ok(Self { child: Mutex::new(child), io: Mutex::new(io), next_id: AtomicU64::new(1), info, spec })
    }

    pub fn has_capability(&self, cap: &str) -> bool {
        self.info.capabilities.iter().any(|c| c == cap)
    }

    /// Chama um método do protocolo e desserializa o `result`.
    pub fn call<T: DeserializeOwned>(&self, method: &str, params: Value) -> Result<T> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = serde_json::json!({ "id": id, "method": method, "params": params });
        let mut io = self.io.lock().map_err(|_| DubError::ChannelClosed("py io mutex"))?;

        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        io.stdin
            .write_all(line.as_bytes())
            .and_then(|_| io.stdin.flush())
            .map_err(|e| self.failed(method, format!("escrever stdin: {e}")))?;

        let mut buf = String::new();
        let n = io.stdout.read_line(&mut buf).map_err(|e| self.failed(method, format!("ler stdout: {e}")))?;
        if n == 0 {
            return Err(self.failed(method, "processo Python encerrou".into()));
        }
        let resp: Response =
            serde_json::from_str(&buf).map_err(|e| self.failed(method, format!("resposta inválida: {e} — {}", buf.trim())))?;
        if let Some(err) = resp.error {
            return Err(self.failed(method, err));
        }
        let value = resp.result.unwrap_or(Value::Object(Default::default()));
        serde_json::from_value(value).map_err(|e| self.failed(method, format!("result inesperado: {e}")))
    }

    fn failed(&self, method: &str, message: String) -> DubError {
        DubError::EngineFailed {
            engine: self.info.name.clone(),
            stage: match method {
                "transcribe" => "transcribe",
                "translate" => "translate",
                "synthesize" => "synthesize",
                "separate" => "separate",
                "diarize" => "diarize",
                _ => "call",
            },
            message,
        }
    }
}

impl Drop for PyProcess {
    fn drop(&mut self) {
        let _: Result<Value> = self.call("shutdown", Value::Object(Default::default()));
        if let Ok(mut c) = self.child.lock() {
            // Dá um instante para sair limpo; depois mata.
            for _ in 0..20 {
                if matches!(c.try_wait(), Ok(Some(_))) {
                    return;
                }
                thread::sleep(std::time::Duration::from_millis(50));
            }
            let _ = c.kill();
        }
    }
}
