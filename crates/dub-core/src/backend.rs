//! Detecção de hardware e escolha de dispositivo.
//!
//! A ideia é simples: o pipeline pede um `Backend` uma vez na inicialização,
//! ele sonda o que existe na máquina e devolve a melhor opção. Cada engine
//! recebe o `Device` escolhido e decide como usá-lo (CUDA no ONNX Runtime,
//! Vulkan no whisper.cpp, `device="cuda"` no PyTorch...).
//!
//! A sondagem aqui é **leve e sem dependências nativas**: procura binários e
//! bibliotecas conhecidas. As engines podem refinar (ex.: tentar criar a sessão
//! CUDA e cair para CPU se falhar) — o `Device` é uma preferência, não garantia.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// NVIDIA via CUDA.
    Cuda,
    /// Qualquer GPU no Windows via DirectML (NVIDIA, AMD, Intel).
    DirectMl,
    /// Qualquer GPU via Vulkan (whisper.cpp / ggml).
    Vulkan,
    /// Apple Silicon.
    Metal,
    Cpu,
}

impl Device {
    pub fn is_gpu(self) -> bool {
        !matches!(self, Device::Cpu)
    }

    /// Nome como o PyTorch espera (`cuda`, `cpu`, `mps`).
    pub fn torch_name(self) -> &'static str {
        match self {
            Device::Cuda => "cuda",
            Device::Metal => "mps",
            // DirectML no torch é via torch-directml ("privateuseone"); tratamos como cpu
            // para as engines Python por padrão.
            Device::DirectMl | Device::Vulkan | Device::Cpu => "cpu",
        }
    }

    /// Tipo de computação sugerido para CTranslate2 / faster-whisper.
    pub fn ct2_compute_type(self) -> &'static str {
        match self {
            Device::Cuda => "float16",
            _ => "int8",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Device::Cuda => "CUDA",
            Device::DirectMl => "DirectML",
            Device::Vulkan => "Vulkan",
            Device::Metal => "Metal",
            Device::Cpu => "CPU",
        };
        f.write_str(s)
    }
}

/// Quanto do hardware usar. O padrão deixa folga para o sistema continuar
/// respondendo e para a máquina não trabalhar no limite por minutos a fio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Performance {
    /// Metade dos núcleos, sem GPU. Notebook no colo, máquina fraca ou compartilhada.
    Eco,
    /// Deixa um núcleo livre; GPU quando o modelo couber com margem. Padrão.
    #[default]
    Balanced,
    /// Todos os núcleos; GPU sempre que possível.
    Max,
}

impl Performance {
    /// Threads de trabalho para esta política, dado o total de núcleos lógicos.
    pub fn threads(self, logical_cores: usize) -> usize {
        let n = logical_cores.max(1);
        match self {
            Performance::Eco => (n / 2).max(1),
            Performance::Balanced => n.saturating_sub(1).max(1),
            Performance::Max => n,
        }
    }

    /// Margem de VRAM exigida além do tamanho do modelo (0.2 = 20 %).
    pub fn vram_margin(self) -> f64 {
        match self {
            Performance::Eco => 1.0, // nunca usa GPU (ver `allows_gpu`)
            Performance::Balanced => 0.20,
            Performance::Max => 0.10,
        }
    }

    pub fn allows_gpu(self) -> bool {
        !matches!(self, Performance::Eco)
    }
}

impl std::str::FromStr for Performance {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "eco" | "economico" | "econômico" => Ok(Performance::Eco),
            "balanced" | "equilibrado" | "auto" => Ok(Performance::Balanced),
            "max" | "maximo" | "máximo" => Ok(Performance::Max),
            _ => Err(format!("modo de desempenho inválido '{s}' (eco | balanced | max)")),
        }
    }
}

/// Resultado da sondagem de hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backend {
    /// Dispositivo preferido, em ordem de qualidade.
    pub preferred: Device,
    /// Todos os dispositivos detectados, do melhor para o pior (CPU sempre presente).
    pub available: Vec<Device>,
    /// Núcleos lógicos de CPU.
    pub cpu_threads: usize,
    /// Nome da GPU, se foi possível descobrir.
    pub gpu_name: Option<String>,
    /// VRAM total da GPU em MB, se foi possível descobrir (NVIDIA via nvidia-smi).
    pub gpu_vram_mb: Option<u64>,
    /// Política de uso do hardware.
    pub performance: Performance,
}

impl Backend {
    /// Sonda a máquina. Nunca falha: no pior caso devolve CPU.
    pub fn detect() -> Self {
        let mut available = Vec::new();
        let mut gpu_name = None;
        let mut gpu_vram_mb = None;

        if let Some((name, vram)) = probe_nvidia() {
            available.push(Device::Cuda);
            gpu_name = Some(name);
            gpu_vram_mb = vram;
        }
        if cfg!(target_os = "windows") && probe_directml() {
            available.push(Device::DirectMl);
        }
        if probe_vulkan() {
            available.push(Device::Vulkan);
        }
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            available.push(Device::Metal);
        }
        available.push(Device::Cpu);

        let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        Backend { preferred: available[0], available, cpu_threads, gpu_name, gpu_vram_mb, performance: Performance::default() }
    }

    /// Aplica uma política de desempenho. `Eco` remove as GPUs da lista.
    pub fn with_performance(mut self, p: Performance) -> Self {
        self.performance = p;
        if !p.allows_gpu() {
            self.available.retain(|d| !d.is_gpu());
            self.preferred = Device::Cpu;
        }
        self
    }

    /// Threads de trabalho segundo a política.
    pub fn work_threads(&self) -> usize {
        self.performance.threads(self.cpu_threads)
    }

    /// Decide se um modelo de `model_bytes` pode ir para a GPU: precisa de VRAM
    /// conhecida e suficiente com a margem da política. Sem informação de VRAM
    /// (AMD/Intel), deixa a engine tentar — ela cai para CPU se falhar ao alocar.
    pub fn gpu_fits(&self, model_bytes: u64) -> GpuFit {
        if !self.preferred.is_gpu() {
            return GpuFit::NoGpu;
        }
        match self.gpu_vram_mb {
            None => GpuFit::Unknown,
            Some(vram) => {
                let needed = (model_bytes as f64 * (1.0 + self.performance.vram_margin())) / (1024.0 * 1024.0);
                if (vram as f64) >= needed {
                    GpuFit::Fits { vram_mb: vram, needed_mb: needed as u64 }
                } else {
                    GpuFit::TooSmall { vram_mb: vram, needed_mb: needed as u64 }
                }
            }
        }
    }

    /// Força CPU (útil para `--cpu` na CLI ou testes).
    pub fn cpu_only() -> Self {
        Backend {
            preferred: Device::Cpu,
            available: vec![Device::Cpu],
            cpu_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            gpu_name: None,
            gpu_vram_mb: None,
            performance: Performance::default(),
        }
    }

    pub fn is_gpu(&self) -> bool {
        self.preferred.is_gpu()
    }

    pub fn has(&self, d: Device) -> bool {
        self.available.contains(&d)
    }

    /// Threads de CPU a dar para engines paralelas quando a GPU já está ocupada
    /// com outra etapa (deixa folga para o sistema).
    pub fn cpu_threads_for_side_work(&self) -> usize {
        (self.work_threads().saturating_sub(1)).max(1)
    }
}

/// Resultado de `Backend::gpu_fits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuFit {
    NoGpu,
    Unknown,
    Fits { vram_mb: u64, needed_mb: u64 },
    TooSmall { vram_mb: u64, needed_mb: u64 },
}

// --- sondas -----------------------------------------------------------------

fn probe_nvidia() -> Option<(String, Option<u64>)> {
    // nvidia-smi é a forma mais confiável e barata de saber se há driver NVIDIA,
    // e ainda informa a VRAM total.
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    if name.is_empty() {
        return None;
    }
    let vram = parts.next().and_then(|v| v.parse::<u64>().ok());
    Some((name, vram))
}

fn probe_directml() -> bool {
    // DirectML vem com o Windows 10 1903+; a DLL fica em System32.
    if let Ok(root) = std::env::var("SystemRoot") {
        return Path::new(&root).join("System32").join("DirectML.dll").exists();
    }
    false
}

fn probe_vulkan() -> bool {
    if cfg!(target_os = "windows") {
        std::env::var("SystemRoot").map(|r| Path::new(&r).join("System32").join("vulkan-1.dll").exists()).unwrap_or(false)
    } else if cfg!(target_os = "linux") {
        ["/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib64/libvulkan.so.1", "/usr/lib/libvulkan.so.1"]
            .iter()
            .any(|p| Path::new(p).exists())
            || which::which("vulkaninfo").is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_present() {
        let b = Backend::detect();
        assert!(b.has(Device::Cpu));
        assert_eq!(*b.available.last().unwrap(), Device::Cpu);
        assert!(b.cpu_threads >= 1);
    }

    #[test]
    fn cpu_only_is_not_gpu() {
        assert!(!Backend::cpu_only().is_gpu());
    }

    #[test]
    fn performance_threads_leave_headroom() {
        assert_eq!(Performance::Balanced.threads(8), 7);
        assert_eq!(Performance::Eco.threads(8), 4);
        assert_eq!(Performance::Max.threads(8), 8);
        assert_eq!(Performance::Balanced.threads(1), 1);
    }

    #[test]
    fn gpu_fit_uses_margin() {
        let mut b = Backend::cpu_only();
        b.preferred = Device::Cuda;
        b.available.insert(0, Device::Cuda);
        b.gpu_vram_mb = Some(2048);
        // medium ggml ≈ 1.5 GB: com 20 % de margem precisa de ~1.8 GB → cabe em 2 GB
        assert!(matches!(b.gpu_fits(1_500 * 1024 * 1024), GpuFit::Fits { .. }));
        // large-v3 ≈ 3 GB → não cabe
        assert!(matches!(b.gpu_fits(3_000 * 1024 * 1024), GpuFit::TooSmall { .. }));
        // Eco tira a GPU
        let eco = b.with_performance(Performance::Eco);
        assert_eq!(eco.gpu_fits(10), GpuFit::NoGpu);
    }
}
