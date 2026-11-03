#include "timestretch.h"

#include <algorithm>
#include <cmath>

namespace dub {
namespace {

// Janela de Hann para o overlap-add. Com hop = window/2 as janelas somam 1.
std::vector<float> hann(std::size_t n) {
    std::vector<float> w(n);
    const double two_pi = 6.283185307179586;
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(two_pi * static_cast<double>(i) / static_cast<double>(n)));
    }
    return w;
}

// Correlação cruzada normalizada entre dois trechos de `n` amostras.
// Devolve valor em [-1, 1]; maior = formas de onda mais parecidas.
float similarity(const float* a, const float* b, std::size_t n) {
    double dot = 0.0, ea = 0.0, eb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        ea += static_cast<double>(a[i]) * a[i];
        eb += static_cast<double>(b[i]) * b[i];
    }
    const double denom = std::sqrt(ea * eb);
    if (denom < 1e-12) return 0.0f;
    return static_cast<float>(dot / denom);
}

}  // namespace

std::vector<float> wsola_stretch(const float* input, std::size_t len, std::uint32_t sample_rate,
                                 float speed, const StretchParams& params) {
    if (len == 0 || speed <= 0.0f) return {};
    if (std::fabs(speed - 1.0f) < 1e-4f) return std::vector<float>(input, input + len);

    const std::size_t window = params.window ? params.window : static_cast<std::size_t>(sample_rate * 0.025);
    const std::size_t search = params.search ? params.search : static_cast<std::size_t>(sample_rate * 0.0075);
    if (len < window * 2) {
        // Sinal mais curto que duas janelas: reamostragem linear simples
        // (muda o tom, mas em <50 ms ninguém percebe).
        const std::size_t out_len = static_cast<std::size_t>(std::lround(len / speed));
        std::vector<float> out(out_len);
        for (std::size_t i = 0; i < out_len; ++i) {
            const double pos = i * static_cast<double>(speed);
            const std::size_t i0 = std::min(static_cast<std::size_t>(pos), len - 1);
            const std::size_t i1 = std::min(i0 + 1, len - 1);
            const float frac = static_cast<float>(pos - i0);
            out[i] = input[i0] + (input[i1] - input[i0]) * frac;
        }
        return out;
    }

    const std::size_t hop_out = window / 2;                                        // hop de síntese (fixo)
    const double hop_in = hop_out * static_cast<double>(speed);                     // hop de análise (escalado)
    const std::size_t out_len = static_cast<std::size_t>(std::lround(len / speed));
    const std::vector<float> win = hann(window);

    std::vector<float> out(out_len + window, 0.0f);
    std::vector<float> norm(out_len + window, 0.0f);  // soma das janelas, para normalizar

    // Primeiro frame: copia direto.
    std::size_t out_pos = 0;
    std::size_t in_pos = 0;  // início do último frame usado na entrada
    for (std::size_t i = 0; i < window && i < len; ++i) {
        out[i] += input[i] * win[i];
        norm[i] += win[i];
    }

    // O "frame natural" seguinte na entrada é o que viria se não houvesse
    // esticamento: in_pos + hop_out. É com ele que o próximo frame precisa se
    // parecer para a sobreposição ficar contínua.
    std::size_t frame = 1;
    while (true) {
        out_pos = frame * hop_out;
        if (out_pos + window > out_len + hop_out) break;

        // Posição alvo na entrada (onde o frame cairia pelo fator de velocidade).
        const double target_d = frame * hop_in;
        if (target_d + window + search >= len) break;
        const std::size_t target = static_cast<std::size_t>(target_d);

        // Trecho de referência: continuação natural do último frame colado.
        const std::size_t natural = in_pos + hop_out;
        if (natural + window > len) break;
        const float* ref = input + natural;

        // Busca em [target - search, target + search] o deslocamento mais parecido com `ref`.
        const std::size_t lo = target > search ? target - search : 0;
        const std::size_t hi = std::min(target + search, len - window);
        std::size_t best = target;
        float best_sim = -2.0f;
        for (std::size_t cand = lo; cand <= hi; ++cand) {
            const float s = similarity(ref, input + cand, window);
            if (s > best_sim) {
                best_sim = s;
                best = cand;
            }
        }

        // Overlap-add do frame escolhido.
        for (std::size_t i = 0; i < window; ++i) {
            out[out_pos + i] += input[best + i] * win[i];
            norm[out_pos + i] += win[i];
        }
        in_pos = best;
        ++frame;
    }

    // Normaliza onde a soma das janelas não deu exatamente 1 (bordas).
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (norm[i] > 1e-3f) out[i] /= norm[i];
    }
    out.resize(out_len);
    return out;
}

std::vector<float> wsola_stretch(const std::vector<float>& input, std::uint32_t sample_rate, float speed) {
    return wsola_stretch(input.data(), input.size(), sample_rate, speed, StretchParams{});
}

}  // namespace dub
