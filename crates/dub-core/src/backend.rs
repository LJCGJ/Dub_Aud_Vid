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

/// Como repartir o trabalho entre CPU e GPU. É a escolha do usuário; `Auto`
/// decide por hardware (ver `Backend::device_for`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceMode {
    /// GPU dedicada: tudo que puder na GPU. GPU integrada (Intel/AMD APU):
    /// transcrição na CPU e voz na GPU, porque a iGPU perde da CPU no whisper.
    #[default]
    Auto,
    /// Só CPU.
    Cpu,
    /// Tudo que tiver backend compilado vai para a GPU, mesmo integrada.
    Gpu,
    /// Divide: transcrição na CPU, síntese de voz na GPU (as duas trabalham em paralelo).
    Split,
}

impl std::str::FromStr for DeviceMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "automatico" | "automático" => Ok(DeviceMode::Auto),
            "cpu" => Ok(DeviceMode::Cpu),
            "gpu" => Ok(DeviceMode::Gpu),
            "split" | "dividir" | "dividido" | "misto" => Ok(DeviceMode::Split),
            _ => Err(format!("modo de dispositivo inválido '{s}' (auto | cpu | gpu | split)")),
        }
    }
}

impl fmt::Display for DeviceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DeviceMode::Auto => "auto",
            DeviceMode::Cpu => "cpu",
            DeviceMode::Gpu => "gpu",
            DeviceMode::Split => "split",
        })
    }
}

/// Fabricante da GPU principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuKind {
    Nvidia,
    Amd,
    Intel,
    Apple,
    #[default]
    Unknown,
}

/// Papel de uma engine no pipeline, para decidir o dispositivo de cada uma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Transcribe,
    Translate,
    Synthesize,
    Separate,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::Transcribe => "transcrição (whisper)",
            Role::Translate => "tradução",
            Role::Synthesize => "voz (TTS)",
            Role::Separate => "separação voz/fundo",
        }
    }
}

/// Uma linha do plano de dispositivos: papel, dispositivo e o porquê.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePlan {
    pub role: Role,
    pub device: Device,
    pub reason: String,
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
    /// VRAM dedicada da GPU em MB, se foi possível descobrir (nvidia-smi ou DXGI).
    /// GPU integrada não tem VRAM própria: fica `None` e `gpu_shared_mb` diz o
    /// quanto de RAM ela pode tomar emprestado.
    pub gpu_vram_mb: Option<u64>,
    /// Memória de sistema compartilhável com a GPU, em MB (DXGI, Windows).
    #[serde(default)]
    pub gpu_shared_mb: Option<u64>,
    /// Fabricante da GPU principal.
    #[serde(default)]
    pub gpu_kind: GpuKind,
    /// GPU integrada ao processador (divide a RAM com o sistema).
    #[serde(default)]
    pub gpu_integrated: bool,
    /// RAM total e disponível em MB, quando o sistema informa.
    #[serde(default)]
    pub ram_total_mb: Option<u64>,
    #[serde(default)]
    pub ram_free_mb: Option<u64>,
    /// Política de uso do hardware.
    pub performance: Performance,
    /// Como repartir CPU/GPU.
    #[serde(default)]
    pub mode: DeviceMode,
    /// Se este build tem backend de GPU para transcrição / síntese (o runtime
    /// informa pelas features compiladas; sem isso o plano prometeria GPU e a
    /// engine cairia em CPU em silêncio).
    #[serde(default = "yes")]
    pub gpu_transcribe_compiled: bool,
    #[serde(default = "yes")]
    pub gpu_synthesize_compiled: bool,
}

fn yes() -> bool {
    true
}

impl Backend {
    /// Sonda a máquina. Nunca falha: no pior caso devolve CPU.
    pub fn detect() -> Self {
        let mut available = Vec::new();
        let mut gpu_name = None;
        let mut gpu_vram_mb = None;
        let mut gpu_shared_mb = None;
        let mut gpu_kind = GpuKind::Unknown;
        let mut gpu_integrated = false;

        if let Some((name, vram)) = probe_nvidia() {
            available.push(Device::Cuda);
            gpu_name = Some(name);
            gpu_vram_mb = vram;
            gpu_kind = GpuKind::Nvidia;
        }
        // Qualquer fabricante: DXGI no Windows, sysfs no Linux.
        if let Some(g) = probe_gpu_generic() {
            if gpu_name.is_none() {
                gpu_name = Some(g.name);
                gpu_kind = g.kind;
                gpu_vram_mb = g.dedicated_mb.filter(|&m| m > 0);
            }
            gpu_shared_mb = g.shared_mb;
            gpu_integrated = g.integrated;
        }
        if gpu_kind == GpuKind::Unknown && cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            gpu_kind = GpuKind::Apple;
            gpu_integrated = true;
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
        // Loader do Vulkan/DirectML presente não quer dizer placa presente
        // (contêineres e VMs têm o loader sem GPU). Sem um nome de GPU vindo
        // de uma sonda real, fica só a CPU — senão o plano prometeria GPU e a
        // engine cairia em CPU em silêncio.
        if gpu_name.is_none() && gpu_kind != GpuKind::Apple {
            available.retain(|d| !d.is_gpu());
        }
        available.push(Device::Cpu);

        let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let (ram_total_mb, ram_free_mb) = probe_ram();

        Backend {
            preferred: available[0],
            available,
            cpu_threads,
            gpu_name,
            gpu_vram_mb,
            gpu_shared_mb,
            gpu_kind,
            gpu_integrated,
            ram_total_mb,
            ram_free_mb,
            performance: Performance::default(),
            mode: DeviceMode::default(),
            gpu_transcribe_compiled: true,
            gpu_synthesize_compiled: true,
        }
    }

    /// Informa quais etapas têm backend de GPU neste build.
    pub fn with_compiled_gpu(mut self, transcribe: bool, synthesize: bool) -> Self {
        self.gpu_transcribe_compiled = transcribe;
        self.gpu_synthesize_compiled = synthesize;
        self
    }

    /// Define como repartir CPU/GPU. `Cpu` remove as GPUs da lista.
    pub fn with_mode(mut self, mode: DeviceMode) -> Self {
        self.mode = mode;
        if mode == DeviceMode::Cpu {
            self.available.retain(|d| !d.is_gpu());
            self.preferred = Device::Cpu;
        }
        self
    }

    /// GPU que seria usada, se houver uma e a política permitir.
    pub fn gpu_device(&self) -> Option<Device> {
        self.available.iter().copied().find(|d| d.is_gpu()).filter(|_| self.performance.allows_gpu())
    }

    /// Dispositivo para cada papel segundo o modo escolhido. Tradução fica
    /// sempre na CPU (CTranslate2 int8 é rápido ali e roda em paralelo com a
    /// síntese na GPU).
    pub fn device_for(&self, role: Role) -> Device {
        self.plan_for(role).device
    }

    /// Como `device_for`, com a justificativa (para mostrar ao usuário).
    pub fn plan_for(&self, role: Role) -> DevicePlan {
        let gpu = self.gpu_device();
        let compiled = match role {
            Role::Transcribe => self.gpu_transcribe_compiled,
            Role::Synthesize => self.gpu_synthesize_compiled,
            _ => true,
        };
        // Apple Silicon é "integrada" mas com memória unificada rápida: Metal
        // vence a CPU no whisper. A regra de poupar a iGPU é para Intel/AMD.
        let weak_igpu = self.gpu_integrated && self.gpu_kind != GpuKind::Apple;
        let (device, reason) = match (role, self.mode, gpu) {
            (Role::Translate, _, _) => (Device::Cpu, "tradução é leve na CPU e roda em paralelo com a voz".to_string()),
            (_, _, None) if !self.performance.allows_gpu() => (Device::Cpu, "modo Eco não usa GPU".to_string()),
            (_, _, None) => (Device::Cpu, "nenhuma GPU utilizável".to_string()),
            (_, DeviceMode::Cpu, _) => (Device::Cpu, "modo CPU escolhido".to_string()),
            // Separação (MDX-Net) é pesada em memória; em iGPU ela estoura a RAM
            // compartilhada e o DirectML derruba o processo. CPU, em qualquer modo.
            (Role::Separate, _, Some(_)) if weak_igpu => {
                (Device::Cpu, "separação pesada em memória; GPU integrada dividiria a RAM e pode estourar".to_string())
            }
            (_, _, Some(_)) if !compiled => (Device::Cpu, "este build não tem backend de GPU para esta etapa".to_string()),
            (_, DeviceMode::Gpu, Some(g)) => (g, "modo GPU escolhido".to_string()),
            (Role::Transcribe, DeviceMode::Split, _) => (Device::Cpu, "modo dividido: transcrição na CPU".to_string()),
            (_, DeviceMode::Split, Some(g)) => (g, "modo dividido: voz na GPU".to_string()),
            (Role::Transcribe, DeviceMode::Auto, Some(_)) if weak_igpu => {
                (Device::Cpu, "GPU integrada divide a RAM e perde da CPU no whisper; a voz fica com a GPU".to_string())
            }
            (_, DeviceMode::Auto, Some(g)) => (g, "GPU disponível".to_string()),
        };
        DevicePlan { role, device, reason }
    }

    /// Plano completo, na ordem do pipeline.
    pub fn plan(&self) -> Vec<DevicePlan> {
        [Role::Separate, Role::Transcribe, Role::Translate, Role::Synthesize].into_iter().map(|r| self.plan_for(r)).collect()
    }

    /// RAM que sobra depois de carregar `model_bytes` na CPU, em MB, se a RAM
    /// for conhecida. Negativo = não cabe com folga.
    pub fn ram_headroom_mb(&self, model_bytes: u64) -> Option<i64> {
        let free = self.ram_free_mb? as i64;
        // Modelos ggml ocupam ~1,3× o arquivo em memória de trabalho; deixamos
        // ainda 1 GB para o sistema não começar a usar swap.
        let needed = (model_bytes as f64 * 1.3 / (1024.0 * 1024.0)) as i64 + 1024;
        Some(free - needed)
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
        // GPU integrada não tem VRAM própria: o que conta é a memória que ela
        // pode tomar emprestada da RAM (e, nesse caso, a RAM livre de verdade).
        let budget = if self.gpu_integrated {
            match (self.gpu_shared_mb, self.ram_free_mb) {
                (Some(shared), Some(free)) => Some(shared.min(free)),
                (Some(shared), None) => Some(shared),
                (None, free) => free,
            }
        } else {
            self.gpu_vram_mb
        };
        match budget {
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
            gpu_shared_mb: None,
            gpu_kind: GpuKind::Unknown,
            gpu_integrated: false,
            ram_total_mb: None,
            ram_free_mb: None,
            performance: Performance::default(),
            mode: DeviceMode::Cpu,
            gpu_transcribe_compiled: true,
            gpu_synthesize_compiled: true,
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

/// GPU vista pelo sistema (qualquer fabricante).
#[allow(dead_code)] // nem toda plataforma tem sonda
struct GenericGpu {
    name: String,
    kind: GpuKind,
    dedicated_mb: Option<u64>,
    shared_mb: Option<u64>,
    integrated: bool,
}

#[allow(dead_code)]
fn vendor_kind(vendor_id: u32) -> GpuKind {
    match vendor_id {
        0x10DE => GpuKind::Nvidia,
        0x1002 | 0x1022 => GpuKind::Amd,
        0x8086 => GpuKind::Intel,
        0x106B => GpuKind::Apple,
        _ => GpuKind::Unknown,
    }
}

/// Integrada = sem VRAM própria de verdade. Intel é integrada salvo as Arc
/// dedicadas, que têm vários GB dedicados; APUs AMD idem.
#[allow(dead_code)]
fn looks_integrated(kind: GpuKind, dedicated_mb: u64) -> bool {
    match kind {
        GpuKind::Nvidia => false,
        GpuKind::Apple => true,
        GpuKind::Intel | GpuKind::Amd | GpuKind::Unknown => dedicated_mb < 1536,
    }
}

#[cfg(target_os = "windows")]
fn probe_gpu_generic() -> Option<GenericGpu> {
    use windows::Win32::Graphics::Dxgi::*;
    // SAFETY: chamadas COM padrão do DXGI; os objetos são liberados no drop.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<GenericGpu> = None;
        let mut i = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let Ok(desc) = adapter.GetDesc1() else { continue };
            if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue; // "Microsoft Basic Render Driver"
            }
            let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').trim().to_string();
            if name.is_empty() {
                continue;
            }
            let kind = vendor_kind(desc.VendorId);
            let dedicated_mb = (desc.DedicatedVideoMemory / (1024 * 1024)) as u64;
            let shared_mb = (desc.SharedSystemMemory / (1024 * 1024)) as u64;
            let g = GenericGpu {
                name,
                kind,
                dedicated_mb: Some(dedicated_mb),
                shared_mb: Some(shared_mb),
                integrated: looks_integrated(kind, dedicated_mb),
            };
            // Com iGPU + dedicada na mesma máquina, a dedicada manda.
            let better = match &best {
                None => true,
                Some(b) => (!g.integrated && b.integrated) || g.dedicated_mb > b.dedicated_mb,
            };
            if better {
                best = Some(g);
            }
        }
        best
    }
}

#[cfg(target_os = "linux")]
fn probe_gpu_generic() -> Option<GenericGpu> {
    // /sys/class/drm/card*/device/{vendor,device}: barato e sem dependências.
    let mut best: Option<GenericGpu> = None;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = entry.path().join("device");
        let Ok(vendor) = std::fs::read_to_string(dev.join("vendor")) else { continue };
        let vendor_id = u32::from_str_radix(vendor.trim().trim_start_matches("0x"), 16).unwrap_or(0);
        let kind = vendor_kind(vendor_id);
        if kind == GpuKind::Unknown {
            continue;
        }
        // VRAM dedicada (amdgpu expõe mem_info_vram_total; i915/xe não).
        let dedicated_mb = std::fs::read_to_string(dev.join("mem_info_vram_total"))
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|b| b / (1024 * 1024));
        let integrated = match kind {
            GpuKind::Intel => dedicated_mb.map(|m| m < 1536).unwrap_or(true),
            k => looks_integrated(k, dedicated_mb.unwrap_or(0)),
        };
        let g = GenericGpu { name: format!("{kind:?} ({name})"), kind, dedicated_mb, shared_mb: None, integrated };
        let better = match &best {
            None => true,
            Some(b) => !g.integrated && b.integrated,
        };
        if better {
            best = Some(g);
        }
    }
    best
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn probe_gpu_generic() -> Option<GenericGpu> {
    None
}

/// (total, disponível) em MB.
#[cfg(target_os = "windows")]
fn probe_ram() -> (Option<u64>, Option<u64>) {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut st = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    // SAFETY: struct inicializada com o tamanho correto, como a API exige.
    if unsafe { GlobalMemoryStatusEx(&mut st) }.is_ok() {
        (Some(st.ullTotalPhys / (1024 * 1024)), Some(st.ullAvailPhys / (1024 * 1024)))
    } else {
        (None, None)
    }
}

#[cfg(target_os = "linux")]
fn probe_ram() -> (Option<u64>, Option<u64>) {
    let Ok(info) = std::fs::read_to_string("/proc/meminfo") else { return (None, None) };
    let get = |key: &str| {
        info.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb / 1024)
    };
    (get("MemTotal:"), get("MemAvailable:"))
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn probe_ram() -> (Option<u64>, Option<u64>) {
    let total = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .map(|b| b / (1024 * 1024));
    (total, None)
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

    fn with_gpu(integrated: bool) -> Backend {
        let mut b = Backend::cpu_only();
        b.mode = DeviceMode::Auto;
        b.available = vec![Device::Vulkan, Device::Cpu];
        b.preferred = Device::Vulkan;
        b.gpu_integrated = integrated;
        b
    }

    #[test]
    fn auto_splits_on_integrated_gpu() {
        let b = with_gpu(true);
        assert_eq!(b.device_for(Role::Transcribe), Device::Cpu);
        assert_eq!(b.device_for(Role::Synthesize), Device::Vulkan);
        assert_eq!(b.device_for(Role::Translate), Device::Cpu);
    }

    #[test]
    fn separation_never_on_integrated_gpu() {
        assert_eq!(with_gpu(true).device_for(Role::Separate), Device::Cpu);
        assert_eq!(with_gpu(true).with_mode(DeviceMode::Gpu).device_for(Role::Separate), Device::Cpu);
        assert_eq!(with_gpu(false).device_for(Role::Separate), Device::Vulkan);
    }

    #[test]
    fn integrated_gpu_fit_uses_shared_memory() {
        let mut b = with_gpu(true);
        b.gpu_vram_mb = Some(128);
        b.gpu_shared_mb = Some(8_000);
        b.ram_free_mb = Some(4_000);
        assert!(matches!(b.gpu_fits(500 * 1024 * 1024), GpuFit::Fits { .. }));
    }

    #[test]
    fn auto_uses_discrete_gpu_everywhere() {
        let b = with_gpu(false);
        assert_eq!(b.device_for(Role::Transcribe), Device::Vulkan);
        assert_eq!(b.device_for(Role::Synthesize), Device::Vulkan);
    }

    #[test]
    fn apple_silicon_keeps_whisper_on_gpu() {
        let mut b = with_gpu(true);
        b.gpu_kind = GpuKind::Apple;
        b.available = vec![Device::Metal, Device::Cpu];
        assert_eq!(b.device_for(Role::Transcribe), Device::Metal);
    }

    #[test]
    fn plan_respects_compiled_backends() {
        let b = with_gpu(false).with_compiled_gpu(false, true);
        assert_eq!(b.device_for(Role::Transcribe), Device::Cpu);
        assert_eq!(b.device_for(Role::Synthesize), Device::Vulkan);
    }

    #[test]
    fn explicit_modes_win() {
        assert_eq!(with_gpu(true).with_mode(DeviceMode::Gpu).device_for(Role::Transcribe), Device::Vulkan);
        assert_eq!(with_gpu(false).with_mode(DeviceMode::Split).device_for(Role::Transcribe), Device::Cpu);
        assert_eq!(with_gpu(false).with_mode(DeviceMode::Split).device_for(Role::Synthesize), Device::Vulkan);
        let cpu = with_gpu(false).with_mode(DeviceMode::Cpu);
        assert_eq!(cpu.device_for(Role::Synthesize), Device::Cpu);
        assert!(!cpu.is_gpu());
        // Eco vence qualquer modo.
        assert_eq!(
            with_gpu(false).with_performance(Performance::Eco).with_mode(DeviceMode::Gpu).device_for(Role::Synthesize),
            Device::Cpu
        );
    }

    #[test]
    fn ram_headroom() {
        let mut b = Backend::cpu_only();
        b.ram_free_mb = Some(4096);
        // medium 1,5 GB → ~1,95 GB + 1 GB de folga = ~3 GB → sobra ~1 GB
        let h = b.ram_headroom_mb(1_500 * 1024 * 1024).unwrap();
        assert!(h > 900 && h < 1200, "{h}");
        b.ram_free_mb = Some(1024);
        assert!(b.ram_headroom_mb(1_500 * 1024 * 1024).unwrap() < 0);
    }
}
