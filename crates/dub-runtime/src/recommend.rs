//! Recomendações calculadas a partir do hardware detectado — não são fixas:
//! mudam com a GPU, a RAM livre e os núcleos da máquina onde o Dub roda.
//! O assistente da CLI e o app marcam "(recomendado)" com base nisto.

use dub_core::{Backend, DeviceMode, GpuKind, Performance};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation<T> {
    pub value: T,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendations {
    pub device: Recommendation<DeviceMode>,
    pub performance: Recommendation<Performance>,
    pub whisper_model: Recommendation<String>,
    pub voice: Recommendation<String>,
}

pub fn recommend(b: &Backend) -> Recommendations {
    let ram_free_gb = b.ram_free_mb.map(|m| m as f64 / 1024.0);
    let threads = b.cpu_threads;
    let discrete_gpu = b.gpu_device().is_some() && !b.gpu_integrated;
    let vram_gb = b.gpu_vram_mb.map(|m| m as f64 / 1024.0).unwrap_or(0.0);

    // Onde processar: o automático já aplica as regras por hardware.
    let device = Recommendation {
        value: DeviceMode::Auto,
        reason: if b.gpu_device().is_none() {
            "sem GPU utilizável: tudo na CPU".into()
        } else if b.gpu_integrated && b.gpu_kind != GpuKind::Apple {
            "GPU integrada: transcrição na CPU e voz na GPU".into()
        } else {
            "GPU dedicada: transcrição e voz na GPU".into()
        },
    };

    // Uso do hardware: Equilibrado salvo máquina apertada. Máximo nunca é
    // recomendado — é escolha de quem aceita a máquina no limite.
    let performance = match (ram_free_gb, threads) {
        (Some(r), _) if r < 3.0 => Recommendation {
            value: Performance::Eco,
            reason: format!("só {r:.1} GB de RAM livre: economiza para não entrar em swap"),
        },
        (_, t) if t <= 2 => Recommendation {
            value: Performance::Eco,
            reason: format!("{t} núcleos: deixa a máquina usável durante a dublagem"),
        },
        _ => Recommendation {
            value: Performance::Balanced,
            reason: "deixa um núcleo livre e a GPU com folga; a máquina continua respondendo".into(),
        },
    };

    // Modelo do whisper: pela RAM livre (CPU) ou VRAM (GPU dedicada).
    let whisper_model = if discrete_gpu && vram_gb >= 4.0 {
        Recommendation {
            value: "large-v3-turbo-q5_0".into(),
            reason: format!("GPU dedicada com {vram_gb:.0} GB: large-v3-turbo quantizado cabe e é rápido"),
        }
    } else {
        match ram_free_gb {
            Some(r) if r >= 3.5 && threads >= 6 => Recommendation {
                value: "medium".into(),
                reason: format!("{r:.1} GB livres e {threads} núcleos: medium cabe com folga"),
            },
            Some(r) if r >= 2.0 => Recommendation {
                value: "small".into(),
                reason: format!("{r:.1} GB livres: small mantém a máquina confortável"),
            },
            Some(r) => Recommendation { value: "base.en".into(), reason: format!("{r:.1} GB livres: só cabe base") },
            None => Recommendation { value: "medium".into(), reason: "RAM desconhecida: medium (padrão)".into() },
        }
    };

    let voice = Recommendation {
        value: crate::voices::AUTO_VOICE.into(),
        reason: "escolhe voz masculina ou feminina conforme a voz original de cada trecho".into(),
    };

    Recommendations { device, performance, whisper_model, voice }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_ram_recommends_eco_and_small() {
        let mut b = Backend::cpu_only();
        b.cpu_threads = 8;
        b.ram_free_mb = Some(2_500);
        let r = recommend(&b);
        assert_eq!(r.performance.value, Performance::Eco);
        assert_eq!(r.whisper_model.value, "small");
    }

    #[test]
    fn comfortable_machine_recommends_balanced_medium() {
        let mut b = Backend::cpu_only();
        b.cpu_threads = 8;
        b.ram_free_mb = Some(6_000);
        let r = recommend(&b);
        assert_eq!(r.performance.value, Performance::Balanced);
        assert_eq!(r.whisper_model.value, "medium");
        assert_eq!(r.device.value, DeviceMode::Auto);
    }
}
