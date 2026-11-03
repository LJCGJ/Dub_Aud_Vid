// Time-stretch de áudio por WSOLA (Waveform Similarity Overlap-Add).
//
// Altera a duração de um sinal mono preservando o tom. É o algoritmo clássico
// para fala: simples, rápido, sem FFT, e com qualidade boa para fatores entre
// 0.7x e 1.5x — exatamente a faixa que o pipeline de dublagem usa.
//
// Referência: Verhelst & Roelands, "An overlap-add technique based on waveform
// similarity (WSOLA) for high quality time-scale modification of speech", 1993.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dub {

struct StretchParams {
    // Tamanho da janela de análise/síntese em amostras. ~20–30 ms para fala.
    std::size_t window = 0;  // 0 = automático (25 ms no sample rate dado)
    // Raio de busca por similaridade em amostras. ~5–10 ms.
    std::size_t search = 0;  // 0 = automático (7.5 ms)
};

// Estica/comprime `input` pelo fator `speed` (>1 encurta). Devolve o sinal
// resultante. `sample_rate` só é usado para os valores automáticos de janela.
std::vector<float> wsola_stretch(const float* input, std::size_t len, std::uint32_t sample_rate,
                                 float speed, const StretchParams& params);

// Versão conveniente com parâmetros automáticos.
std::vector<float> wsola_stretch(const std::vector<float>& input, std::uint32_t sample_rate, float speed);

}  // namespace dub
