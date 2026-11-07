# Ornith-1.5-35B-A3B QSA for llama.cpp

Qwen Sparse Attention (QSA) runtime for **ornith-ai/Ornith-1.5-35B-A3B**
(Qwen3.5-MoE, 40 layers = 30 GDN + 10 full attention), built on
`tvall43/llama.cpp` @ `e98d01d` (branch `qwen35moe-qsa`) with the additions below.

The upstream llama.cpp README is kept as [README-upstream.md](README-upstream.md).

## What this fork adds

The base branch already loads the QSA indexer tensors and can attend through a
full-KV mask. This fork makes the selection actually sparse:

- **Block-sparse decode and prefill**: every query keeps its own top-k pools plus its
  own tail block; a per-cell mask marks the visible 2048 cells. The CUDA kernel reads
  those cells straight from the cache, so attention cost follows the budget instead of
  the context length. (`LLAMA_QSA_SPARSE=1`)
- **Indexed lightning indexer**: instead of gathering all pooled keys every token, the
  indexer kernel reads the cache rows through `pool_cells`. Removes a 16.8 MB/layer/token
  copy at 262k. (`LLAMA_QSA_IDXDIRECT=1`)
- **Shared decode mask**: the decode mask depends only on the position, so it is built
  once per graph and shared by the 10 QSA layers instead of rebuilt per layer.

The standalone patch against `e98d01d` is [qsa-ornith.patch](qsa-ornith.patch).
Default behaviour is unchanged unless the environment variables are set.

## Build

```sh
cmake -B build-cuda12 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="70;86" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON
cmake --build build-cuda12 -j8 --target llama-server
```

## Run

```sh
LLAMA_QSA_SPARSE=1 LLAMA_QSA_IDXDIRECT=1 \
  build-cuda12/bin/llama-server -m Ornith-1.5-35B-Q8_0-QSA-mm1.gguf \
  --mmproj mmproj-Ornith-1.5-35B-BF16.gguf -c 262144 -ngl 99 \
  --split-mode layer --tensor-split 1,1 --flash-attn on -ctk f16 -ctv f16 \
  --jinja --reasoning-format deepseek -a ornith-qsa
```

Weights: private HF repo `higa2515/Ornith-1.5-35B-A3B-QSA`
(Q8_0 GGUF with 50 FP32 indexer tensors, plus the distilled `qsa_indexers` checkpoint).

## Measured (2× V100 32GB, Q8_0, F16 KV, 4 tasks × 2 repeats)

| context | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 4,096 | 791.1 | 86.3 |
| 16,384 | 762.6 | 84.9 |
| 65,536 | 737.4 | 82.5 |
| 131,072 | 726.6 | 80.0 |
| 261,120 | 709.6 | 75.9 |

For comparison, the mask-path / gather-indexer build of the same model measured
339.3 prefill and 68.3 decode at 261,120 tokens. Decode at 4k falls only 12% by 262k
in this fork. Single-machine numbers; not a guarantee.

## Differences from the 27B QSA fork

This fork targets `qwen35moe` (MoE + GDN) rather than the dense Qwen3.5 27B. The 27B
fork's MTP/cache features and block-refresh cache are not ported; MTP stays dense and
the sparse path assumes a single sequence with cell == position. Prefill and decode
both use the block-sparse kernel; no full-KV mask is built for decode.

## License

llama.cpp and ggml are MIT; see [LICENSE](LICENSE). QSA method: Qwen3.8-Next report.
Base model weights and terms belong to ornith-ai/Ornith-1.5-35B-A3B.
