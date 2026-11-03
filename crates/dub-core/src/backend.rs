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
}

impl Backend {
    /// Sonda a máquina. Nunca falha: no pior caso devolve CPU.
    pub fn detect() -> Self {
        let mut available = Vec::new();
        let mut gpu_name = None;

        if let Some(name) = probe_nvidia() {
            available.push(Device::Cuda);
            gpu_name = Some(name);
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

        Backend { preferred: available[0], available, cpu_threads, gpu_name }
    }

    /// Força CPU (útil para `--cpu` na CLI ou testes).
    pub fn cpu_only() -> Self {
        Backend {
            preferred: Device::Cpu,
            available: vec![Device::Cpu],
            cpu_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            gpu_name: None,
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
        (self.cpu_threads.saturating_sub(2)).max(1)
    }
}

// --- sondas -----------------------------------------------------------------

fn probe_nvidia() -> Option<String> {
    // nvidia-smi é a forma mais confiável e barata de saber se há driver NVIDIA.
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
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
        std::env::var("SystemRoot")
            .map(|r| Path::new(&r).join("System32").join("vulkan-1.dll").exists())
            .unwrap_or(false)
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
}
