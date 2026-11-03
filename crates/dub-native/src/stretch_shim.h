// Shim entre a assinatura C++ pura (native/timestretch) e os tipos do cxx.
#pragma once

#include <cstdint>

#include "rust/cxx.h"
#include "timestretch.h"

namespace dub {
namespace bridge {

inline rust::Vec<float> wsola(rust::Slice<const float> input, std::uint32_t sample_rate, float speed) {
    const std::vector<float> out =
        dub::wsola_stretch(input.data(), input.size(), sample_rate, speed, dub::StretchParams{});
    rust::Vec<float> result;
    result.reserve(out.size());
    for (float v : out) result.push_back(v);
    return result;
}

}  // namespace bridge
}  // namespace dub
