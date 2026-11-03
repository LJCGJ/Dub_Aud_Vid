//! Encaixe temporal: a tradução em português costuma ser 15–30% mais longa que
//! o inglês. Esta etapa decide **quanto acelerar** cada fala e **quanto pode
//! transbordar** para o silêncio seguinte sem atropelar a próxima fala.
//!
//! Estratégia (em ordem):
//! 1. Se a dublagem já cabe na janela, nada a fazer.
//! 2. Usa a folga até o início do próximo segmento (`gap`), até `max_overflow`.
//! 3. Se ainda não cabe, calcula o fator de velocidade, limitado a `max_speed`.
//! 4. Se mesmo no `max_speed` não cabe, aceita o transbordo (melhor atropelar
//!    um pouco do que ficar ininteligível).

use crate::segment::Segment;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    /// Velocidade máxima aceitável (1.35 ainda soa natural; acima disso fica "corrido").
    pub max_speed: f32,
    /// Velocidade mínima (para falas curtas em janelas longas, não desacelera além disso).
    pub min_speed: f32,
    /// Quanto do silêncio seguinte pode ser usado, em segundos.
    pub max_overflow: f64,
    /// Margem a deixar antes da próxima fala, em segundos.
    pub safety_gap: f64,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self { max_speed: 1.35, min_speed: 0.9, max_overflow: 1.5, safety_gap: 0.15 }
    }
}

/// Decisão de encaixe para um segmento.
#[derive(Debug, Clone, PartialEq)]
pub struct FitPlan {
    /// Fator a aplicar ao áudio sintetizado (>1 = mais rápido).
    pub speed: f32,
    /// Janela efetiva disponível (duração do segmento + folga usada).
    pub window: f64,
    /// Quanto vai transbordar além da janela mesmo após acelerar (0 se coube).
    pub overflow: f64,
}

/// Calcula o plano de encaixe para o segmento `i` dado a duração da dublagem.
///
/// `next_start` é o início do próximo segmento (None no último).
pub fn plan_fit(seg: &Segment, dubbed_len: f64, next_start: Option<f64>, cfg: &TimingConfig) -> FitPlan {
    let base = seg.duration();

    // Folga até a próxima fala, respeitando margem e limite.
    let gap = next_start.map(|ns| (ns - seg.end - cfg.safety_gap).max(0.0)).unwrap_or(cfg.max_overflow).min(cfg.max_overflow);

    let window = base + gap;

    if dubbed_len <= 0.0 || window <= 0.0 {
        return FitPlan { speed: 1.0, window, overflow: 0.0 };
    }

    // Cabe na janela base sem acelerar? Talvez até possa desacelerar um pouco.
    if dubbed_len <= base {
        let slow = (dubbed_len / base) as f32;
        let speed = slow.max(cfg.min_speed).min(1.0);
        return FitPlan { speed, window, overflow: 0.0 };
    }

    // Cabe usando a folga?
    if dubbed_len <= window {
        return FitPlan { speed: 1.0, window, overflow: 0.0 };
    }

    // Precisa acelerar.
    let needed = (dubbed_len / window) as f32;
    let speed = needed.min(cfg.max_speed);
    let final_len = dubbed_len / speed as f64;
    let overflow = (final_len - window).max(0.0);

    FitPlan { speed, window, overflow }
}

/// Aplica `plan_fit` a toda a lista, preenchendo `speed_factor`.
/// Devolve os planos na mesma ordem.
pub fn plan_all(segments: &mut [Segment], dubbed_lens: &[f64], cfg: &TimingConfig) -> Vec<FitPlan> {
    let n = segments.len();
    let mut plans = Vec::with_capacity(n);
    for i in 0..n {
        let next_start = segments.get(i + 1).map(|s| s.start);
        let plan = plan_fit(&segments[i], dubbed_lens.get(i).copied().unwrap_or(0.0), next_start, cfg);
        segments[i].speed_factor = plan.speed;
        plans.push(plan);
    }
    plans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64) -> Segment {
        Segment::new(0, start, end, "x")
    }

    #[test]
    fn fits_without_change() {
        let p = plan_fit(&seg(0.0, 2.0), 2.0, Some(5.0), &TimingConfig::default());
        assert_eq!(p.speed, 1.0);
        assert_eq!(p.overflow, 0.0);
    }

    #[test]
    fn uses_gap_before_speeding() {
        // dub 2.5s, janela 2s, próxima fala em 4s → folga 1.85 → cabe sem acelerar
        let p = plan_fit(&seg(0.0, 2.0), 2.5, Some(4.0), &TimingConfig::default());
        assert_eq!(p.speed, 1.0);
        assert_eq!(p.overflow, 0.0);
    }

    #[test]
    fn speeds_up_when_needed() {
        // dub 3s, janela 2s, próxima fala logo em 2.2 → folga 0.05 → precisa acelerar
        let p = plan_fit(&seg(0.0, 2.0), 3.0, Some(2.2), &TimingConfig::default());
        assert!(p.speed > 1.3 && p.speed <= 1.35);
        assert!(p.overflow > 0.0); // mesmo no máximo, transborda um pouco
    }

    #[test]
    fn slows_short_speech_a_little() {
        let p = plan_fit(&seg(0.0, 2.0), 1.0, Some(5.0), &TimingConfig::default());
        assert_eq!(p.speed, 0.9); // limitado pelo min_speed
    }
}
