# Toolchain "portátil" para o whisper.cpp/ggml nas builds de release (CI).
#
# Por padrão o ggml compila com GGML_NATIVE=ON (-march=native), otimizando para
# a CPU da máquina que compila. No GitHub Actions isso gera um binário com
# AVX-512 que dá "instrução ilegal" em PCs comuns. Aqui fixamos o nível AVX2
# (Intel Haswell 2013+ / AMD Excavator e Zen) como piso da release.
#
# Uso: CMAKE_TOOLCHAIN_FILE=<caminho deste arquivo>  (o whisper-rs-sys repassa
# variáveis CMAKE_* ao CMake). Builds locais (cargo build no seu PC) não usam
# isto e continuam otimizadas para a sua própria CPU.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_AVX    ON  CACHE BOOL "" FORCE)
set(GGML_AVX2   ON  CACHE BOOL "" FORCE)
set(GGML_FMA    ON  CACHE BOOL "" FORCE)
set(GGML_F16C   ON  CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)
set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)
