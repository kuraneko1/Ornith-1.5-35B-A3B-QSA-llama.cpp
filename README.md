# Ornith-1.5-35B-A3B QSA for llama.cpp

Qwen Sparse Attention (QSA) runtime for **ornith-ai/Ornith-1.5-35B-A3B**
(Qwen3.5-MoE, 40 layers = 30 GDN + 10 full attention).

This repository is a **source distribution**: build it for your own GPU and toolkit.
No prebuilt binaries are provided, and a build made for one machine should not be
reused on another.

The upstream llama.cpp README is kept as [README-upstream.md](README-upstream.md).

## What this fork adds

The base branch loads QSA indexer tensors and can attend through a full-KV mask. This
fork makes the selection actually sparse:

- **Block-sparse decode and prefill**: every query keeps its own top-k pools plus its
  own tail block, and a per-cell mask marks the visible 2048 cells. The CUDA kernel
  reads those cells straight from the cache, so attention cost follows the budget, not
  the context length. (`LLAMA_QSA_SPARSE=1`)
- **Indexed lightning indexer**: instead of gathering all pooled keys every token, the
  indexer kernel reads the cache rows through `pool_cells`. (`LLAMA_QSA_IDXDIRECT=1`)
- **Shared decode mask**: the decode mask depends only on the position, so it is built
  once per graph and shared by the 10 QSA layers.

The standalone patch against `e98d01d` is [qsa-ornith.patch](qsa-ornith.patch).
Without the environment variables the fork behaves like the base branch.

## Requirements

- A CUDA build of llama.cpp. The block-sparse QSA kernels are CUDA-only at this commit.
- CMake >= 3.20, Ninja, a C++17 compiler, CUDA Toolkit 12.x (tested with 12.9).
- The QSA GGUF (Q8_0 backbone + FP32 indexers) and its `mmproj`.

Other GPUs are expected to work when built for their architecture, but only V100 has
been measured so far. See [docs/V100.md](docs/V100.md) for that configuration.

## Build for your machine

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j --target llama-server
```

Set the architectures explicitly when you need a reproducible build:

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="native"
```

Use the CUDA compiler and toolkit of your own environment (`CUDACXX`, `CMAKE_CUDA_COMPILER`,
`CUDAToolkit_ROOT`). Do not reuse the V100 build flags from `docs/V100.md` on a different GPU.

## Run

```sh
LLAMA_QSA_SPARSE=1 LLAMA_QSA_IDXDIRECT=1 \
  build/bin/llama-server -m Ornith-1.5-35B-Q8_0-QSA-mm1.gguf \
  --mmproj mmproj-Ornith-1.5-35B-BF16.gguf \
  -c 262144 -ngl 99 --flash-attn on -ctk f16 -ctv f16 \
  --jinja --reasoning-format deepseek -a ornith-qsa
```

- Lower `-c` as needed for your VRAM. `262144` needs roughly 44 GB of KV+weights on this
  model at F16 KV.
- `--split-mode layer --tensor-split 1,1` is only for multi-GPU machines; omit it on one GPU.
- `-ngl 99` offloads all repeating layers; the token embedding stays on the host by design.

## Environment flags

| flag | default | effect |
|---|---|---|
| `LLAMA_QSA_SPARSE` | off | block-sparse prefill/decode instead of the full-KV mask |
| `LLAMA_QSA_IDXDIRECT` | off | indexer reads cache rows directly (no pool gather) |
| `LLAMA_QSA_TOP_K` | GGUF value | diagnostic override of the token budget (2048) |

## Model

Weights are kept in a private Hugging Face repository:
`higa2515/Ornith-1.5-35B-A3B-QSA` (Q8_0 GGUF, mmproj, distilled indexer checkpoint).
They are not redistributed here. Base model terms apply.

## Measured example

Only the V100 configuration is measured so far. See [docs/V100.md](docs/V100.md) for the
hardware, build flags, 0–262k curve, and the comparison with the mask/gather build.

## License

llama.cpp and ggml are MIT; see [LICENSE](LICENSE). QSA method: Qwen3.8-Next report.
Base model weights and terms belong to ornith-ai/Ornith-1.5-35B-A3B.
