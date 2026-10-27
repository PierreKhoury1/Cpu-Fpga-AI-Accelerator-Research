# Roofline Theory for LLM Inference and the Non-GPU Accelerator Landscape (state of the world: October 2026)

Scope note: Section A (questions A1-A5) gives the quantitative theory with explicit formulas and worked numbers for Llama-3-class 8B / 70B and DeepSeek-V3/R1. Section B (questions B1-B5) surveys non-GPU accelerators and cheaper-than-hardware strategies. All derived numbers are labelled "derived" and show the formula so the reader can recompute with their own model/hardware figures. Prices in USD; each figure is dated where the source dates it.

---

## A1. Roofline model, arithmetic intensity of decode vs prefill, and the bandwidth-bound decode rule

### Takeaway
Single-stream decode is a matrix-vector product with arithmetic intensity of only ~1-4 FLOP/byte (depending on weight precision), far below the ridge point of any accelerator (A100 ~208, H100 ~295 FLOP/byte), so tokens/s at batch 1 is simply memory bandwidth divided by the bytes of active weights (plus KV cache); prefill is a matrix-matrix product with intensity proportional to prompt length and is compute-bound.

### Cited Findings
- The roofline model (Williams, Waterman, Patterson, CACM 2009) bounds attainable performance by min(peak FLOP/s, arithmetic intensity x peak bandwidth); the "ridge point" is peak FLOP/s / peak bandwidth — [Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures"](https://dl.acm.org/doi/10.1145/1498765.1498785)
- A matrix-vector multiply (m x n matrix) costs 2mn FLOPs; a matrix-matrix multiply (m x n times n x p) costs 2mnp FLOPs; a full forward pass costs about 2P FLOPs per token for P parameters — [kipply, "Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/)
- A100: 312e12 FLOP/s peak and 1.5e12 bytes/s memory bandwidth, giving a compute-to-memory ratio of 208 FLOPs per byte; "below the ratio (about 208 tokens) you are memory-bandwidth bound; above it, you are FLOP bound" — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- kipply's small-batch (memory-bound) per-step latency: compute ~ 2 x P x (bytes per parameter) / (N x memory bandwidth) on N accelerators; comms ~ 4 x n_layers x 8 us (a latency term, not a bandwidth term) — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- Pope et al. (Google, 2022): "An N-parameter decoder-only model requires 2N matmul FLOPs in the forward pass per token"; weights and KV cache "need to be transferred from HBM to the compute cores of the chip once per forward pass (prefill or decode step)" which they call "memory time"; "at small batch sizes and sequence lengths, the time to load weights dominates. At larger batch sizes and sequence lengths (e.g. 2048+ tokens with batch size 512+), the time to load the KV cache dominates" — [Pope et al., "Efficiently Scaling Transformer Inference", arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Pope et al. report 29 ms/token low-batch latency with int8 weights on PaLM 540B over 64 TPU v4 chips and 76% MFU for large-batch prefill; the minimum-latency point was 28.5 ms/token with int8 weights at batch 64 — [Pope et al.](https://arxiv.org/abs/2211.05102)
- Databricks: "Generating the first token is typically compute-bound, while subsequent decoding is memory-bound"; it defines Model Bandwidth Utilization (MBU) = (model parameter bytes + KV cache bytes) / TPOT / peak bandwidth, with an example of a 7B 16-bit model moving 14 GB in 14 ms (~1 TB/s) = 50% MBU on a 2 TB/s part — [Databricks, "LLM Inference Performance Engineering: Best Practices"](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Databricks measured ~60% MBU at batch size 1 on 2x H100-80GB and ~55% on 4x A100-40GB for Llama-2 70B; H100-80GB has 2.15x the memory bandwidth of A100-40GB and gave 36% lower latency at batch 1 — [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Horace He's first-principles framing: deep-learning time is bandwidth, compute or overhead bound, and most elementwise/low-reuse ops are bandwidth bound — [Horace He, "Making Deep Learning Go Brrrr From First Principles"](https://horace.io/brrr_intro.html)
- A 2025 study found that LLM inference remains memory-bound even at large batch sizes on GPUs ("DRAM bandwidth is the limiting factor in large-batch regimes") — [Mind the Memory Gap, arXiv:2503.08311](https://arxiv.org/html/2503.08311v2)
- A worked example from an LLM-serving guide: a 70B FP16 model on an H100 SXM5 (3.35 TB/s) needs roughly 42 ms per token just to stream the weights at peak bandwidth — [KodeKloud llm-d notes](https://notes.kodekloud.com/docs/llm-d-A-Deep-Dive/Inside-a-Single-Request/The-Speed-Limit/page.md)

### Inferences
- Formulas (derived from the cited definitions):
  - Weight bytes W = N_active x b, where b = bytes per weight (BF16 2.0; INT8 1.0; 4-bit GGUF/AWQ-style ~0.56 i.e. ~4.5 bits/weight including scales/zeros).
  - Decode arithmetic intensity at batch 1 (mat-vec): AI_1 = 2mn / (mn x b) = 2/b FLOP per byte -> BF16: 1; INT8: 2; 4-bit: ~3.6-4 (and 4-bit needs dequant compute on top).
  - Batch B tokens per step (mat-mat, activations negligible): AI_B ~ 2B/b. Compute-bound when AI_B > ridge R = FLOPS_peak / BW_peak, i.e. when B > R x b / 2.
  - Decode step time t_step >= max( (W + KV_read) / BW_mem , 2 x N_active x B / FLOPS ) + comms. Tokens/s per user = 1/t_step; aggregate tokens/s = B/t_step.
  - Batch-1 rule: tokens/s <= BW_mem / (W + L x KV_bytes_per_token), i.e. BW_needed = (W + L x KV_tok) x target_tok_s.
- Ridge points (derived from vendor peak specs): H100 SXM = 989 TFLOPS BF16 dense / 3.35 TB/s = ~295 FLOP/byte (FP8 dense 1,979 TFLOPS -> ~590); A100 = 312/1.555 = ~200 (kipply rounds to 208); a DDR5 server CPU using the assignment's assumption of ~50 TFLOPS BF16 (AMX) and ~500 GB/s = ~100 FLOP/byte (for 12ch DDR5-6400 at 614 GB/s theoretical it is ~80); Tenstorrent Blackhole p150 = 664 TFLOPS (block-FP8, vendor) / 512 GB/s = ~1,300 FLOP/byte, i.e. extremely compute-rich and bandwidth-poor (needs ~650 tokens in flight at FP8 to be compute-bound).
- Batch needed to become compute-bound B* = R x b/2: H100 BF16 ~295, INT8 ~148, 4-bit ~74 (ignoring dequant cost); DDR5 CPU with R~100: BF16 ~100, INT8 ~50, 4-bit ~25. A CPU therefore leaves the bandwidth-bound regime at much smaller batch than a GPU, after which its (much lower) FLOP rate caps aggregate throughput.
- AI at batch 64 (ignoring KV): BF16 64, INT8 128, 4-bit ~256 FLOP/byte -> on H100 BF16 batch 64 is still memory-bound; on a ~100 FLOP/byte CPU, 4-bit at batch 64 is compute-bound and aggregate tok/s caps at FLOPS/(2 x N_active): with 50 TFLOPS real that is ~355 tok/s aggregate for a 70B model, ~3,100 tok/s for 8B.
- Measured MBU for well-tuned GPU serving at batch 1 is 50-60% (Databricks), and 41-50% for the newer Tenstorrent cards (see B1), so real tokens/s should be budgeted at roughly half the peak-bandwidth bound unless the kernel stack is proven.

### Gaps
- Apple does not publish M3 Ultra GPU FLOPS in a form I could cite, so its ridge point is not given; its bandwidth is treated in A2.
- Achievable (STREAM) fraction of theoretical DDR5 bandwidth for 12-channel Turin / Xeon 6 was not found in a citable form (MEGWARE/Red Hat pages have charts, not text); I use 70-75% as a working assumption below and flag it.

---

## A2. Worked numbers: bytes per token, bandwidth for 10/30/100 tok/s, and what DDR5 implies

### Takeaway
At batch 1, Llama 3 70B at 4-bit needs ~400 GB/s for 10 tok/s and ~4 TB/s for 100 tok/s; a single DDR5 socket (~614-845 GB/s theoretical, ~450-600 GB/s achievable) therefore tops out around 10-15 tok/s on 70B-Q4 and ~20-30 tok/s on DeepSeek-V3/R1-Q4, which is consistent with measured M3 Ultra (819 GB/s) results of 16-20 tok/s on R1-Q4.

### Cited Findings
- Llama 3 model sizes and GQA configuration (8B: 32 layers, d_model 4096, 32 heads, 8 KV heads; 70B: 80 layers, d_model 8192, 64 heads, 8 KV heads; head dim 128) — [Llama 3 Herd of Models, arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- DeepSeek-V3: 671B total parameters, 37B activated per token, 61 layers, Multi-head Latent Attention with KV compression dim d_c = 512 and decoupled RoPE key dim 64, native FP8 weights — [DeepSeek-V3 Technical Report, arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Llama-2 70B is 140 GB in FP16; KV cache at batch 1, seq 1024 is 0.312 GiB in FP16 (0.156 GiB INT8); batch 16 needs ~5 GiB — [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- H100 SXM5 memory bandwidth 3.35 TB/s; 70B FP16 weights take ~42 ms/token to stream at peak — [KodeKloud](https://notes.kodekloud.com/docs/llm-d-A-Deep-Dive/Inside-a-Single-Request/The-Speed-Limit/page.md)
- AMD EPYC Turin (5th gen) has 12 memory channels up to DDR5-6400, 1 DIMM per channel — [Lenovo Press LP2283](https://lenovopress.lenovo.com/lp2283-balanced-memory-configurations-with-5th-generation-amd-epyc-processors); [Chips and Cheese, Turin launch](https://chipsandcheese.com/p/amds-turin-5th-gen-epyc-launched)
- Intel Xeon 6 (Granite Rapids) is the first platform to support MRDIMM (MCR DIMM) at 8800 MT/s host-side, 12 channels on the 6900P — [ServeTheHome, Xeon 6900P](https://servethehome.com/welcome-back-intel-xeon-6900p-reasserts-intel-server-leadership/2); [TechRadar / Micron MRDIMM](https://www.techradar.com/pro/microns-data-center-mrdimms-double-the-bandwidth-of-existing-ddr5-dimms-promise-enhanced-ai-and-hpc-performance-with-xeon-6-granite-rapid-cpus)
- Apple M3 Ultra Mac Studio with 512 GB unified memory and "more than 800 GB/s" of bandwidth runs DeepSeek R1 671B 4-bit entirely in memory; reviewers report 17-18 tok/s drawing under 200 W (March 2025) — [TechRadar](https://www.techradar.com/pro/apple-mac-studio-m3-ultra-workstation-can-run-deepseek-r1-671b-ai-model-entirely-in-memory-using-less-than-200w-reviewer-finds); [The Register, Mar 5 2025](https://theregister.com/2025/03/05/apple_m3_ultra_mac_studio)
- Other M3 Ultra R1-Q4 measurements: MLX 4-bit 19.17 tok/s vs GGUF 15.78 tok/s; published tests range 16.5-19.9 tok/s — [dev.to (R1-0528 on M3 Ultra)](https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4); [QbitAI, Mar 2025](https://www.qbitai.com/2025/03/263949.html)
- ktransformers (Tsinghua, Feb 2025): DeepSeek-R1 Q4_K_M on one 24 GB GPU + Xeon host uses 14 GB VRAM and 382 GB DRAM; decode 8.73 tok/s (32 cores) rising to 13.69 tok/s (6-expert selection, v0.3 preview); prefill 54 -> 255 tok/s with the AMX MoE kernel and 286 tok/s with 6-expert selection — [ktransformers DeepSeek-R1/V3 tutorial](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/en/DeepseekR1_V3_tutorial.md); [TechNode, Feb 17 2025](https://technode.com/2025/02/17/tsinghua-universitys-ktransformers-enables-full-powered-deepseek-r1-with-low-cost-graphics-card/)
- Epoch AI's frontier model: serving speed for a dense Transformer scales roughly with the inverse square root of parameter count and with the cube root of memory bandwidth of the GPUs used — [Epoch AI, "Inference Economics of Language Models"](https://epoch.ai/blog/inference-economics-of-language-models)

### Inferences
- Bytes of active weights per token (derived, W = N x b; 4-bit taken as 4.5 bits/weight):
  | Model | Params (active) | BF16 | INT8/FP8 | 4-bit |
  |---|---|---|---|---|
  | Llama 3 8B | 8.03B | 16.1 GB | 8.0 GB | ~4.5 GB |
  | Llama 3 70B | 70.6B | 141 GB | 70.6 GB | ~40 GB |
  | DeepSeek-V3/R1 (per token) | 37B active of 671B | 74 GB | 37 GB (native FP8) | ~21 GB |
  | DeepSeek-V3/R1 (resident) | 671B | 1,342 GB | 671 GB | ~377 GB (GGUF Q4_K_M files are ~400 GB) |
  Caveat for MoE: 37B is the per-token active set at batch 1; as batch grows, different tokens hit different experts, so bytes read per step rise toward the full 671B until reuse saturates.
- Bandwidth required at batch 1 (derived, BW = W x tok/s, ignoring KV cache):
  | Model / precision | GB per token | 10 tok/s | 30 tok/s | 100 tok/s |
  |---|---|---|---|---|
  | 8B 4-bit | 4.5 | 45 GB/s | 135 GB/s | 450 GB/s |
  | 8B INT8 | 8.0 | 80 | 240 | 800 |
  | 8B BF16 | 16.1 | 161 | 483 | 1,610 |
  | 70B 4-bit | 40 | 400 | 1,200 | 4,000 |
  | 70B INT8 | 70.6 | 706 | 2,118 | 7,060 |
  | 70B BF16 | 141 | 1,410 | 4,230 | 14,100 |
  | DS-V3 4-bit (37B active) | 21 | 210 | 630 | 2,100 |
  | DS-V3 FP8 | 37 | 370 | 1,110 | 3,700 |
  | DS-V3 BF16 | 74 | 740 | 2,220 | 7,400 |
- Theoretical batch-1 ceilings at 100% MBU (derived): H100 (3,350 GB/s): 8B-BF16 209 tok/s, 8B-Q4 744, 70B-Q4 84, 70B-INT8 47, 70B-BF16 24 (needs 2 GPUs for capacity). One DDR5-6400 12-channel socket (614 GB/s theoretical; ~450 GB/s at an assumed ~73% achievable): 8B-Q4 136 / 100, 8B-BF16 38 / 28, 70B-Q4 15 / 11, 70B-INT8 8.7 / 6.4, DS-V3-Q4 29 / 21. Xeon 6900P with MRDIMM-8800 (845 GB/s theoretical): 70B-Q4 21, DS-V3-Q4 40. Apple M3 Ultra (819 GB/s): 70B-Q4 20, DS-V3-Q4 39 theoretical vs 16-20 measured, i.e. ~40-50% MBU.
- Cross-check of CPU numbers against measurement: ktransformers' 13.7 tok/s decode on R1-Q4 implies ~21 GB x 13.7 = ~290 GB/s effective, consistent with a dual-socket DDR5 host achieving roughly 40-50% of theoretical bandwidth in a real MoE kernel.
- DDR5 implication: HBM3/3e parts deliver 3.35-8 TB/s per package; a DDR5 socket delivers 0.6-0.85 TB/s theoretical, so a single-socket DDR5 design is 4-10x behind one GPU on batch-1 tokens/s for the same model and precision. Reaching H100-class batch-1 speed on 70B-Q4 (~84 tok/s) would require ~4 TB/s, i.e. 5-7 sockets' worth of DDR5 bandwidth that are all read in parallel on each token (see A5 for why this requires tensor parallelism, not pipelining).
- Prefill is the opposite problem on CPUs: a 4k prompt at 1 s needs ~66 TFLOPS (8B), ~303 TFLOPS (DS-V3) or ~578 TFLOPS (70B) at 100% MFU (A3), which a DDR5 CPU cannot supply; this is why ktransformers' CPU+GPU hybrid keeps prefill on AMX/GPU and why a CPU+FPGA design must budget prefill FLOPs separately from decode bandwidth.

### Gaps
- No citable full-socket STREAM measurement for 12-channel Turin or Xeon 6 (MRDIMM) was found; 614 GB/s and 845 GB/s are theoretical (channels x 8 bytes x MT/s). The 73% achievable assumption is mine.
- Apple's exact 819 GB/s spec was not confirmed in the fetched sources (they say ">800 GB/s"); 819 GB/s is Apple's published figure to the best of my knowledge and should be verified on apple.com.
- 4-bit GGUF file sizes (e.g. ~4.9 GB for 8B Q4_K_M, ~400 GB for R1 Q4_K_M) are from memory of common distributions, not from a fetched source.

---

## A3. Compute side: FLOPs per token, prefill TOPS, batching and the ridge point

### Takeaway
FLOPs per token are ~2 x active parameters (16 GFLOP for 8B, 141 GFLOP for 70B, 74 GFLOP for DeepSeek-V3), so prefilling a 4k-token prompt in one second needs ~66 / ~578 / ~303 TFLOPS at 100% MFU (double that at a realistic 50%); batching multiplies arithmetic intensity by the number of tokens in flight, moving decode toward the compute roof once B exceeds ridge x bytes-per-weight / 2.

### Cited Findings
- "An N-parameter decoder-only model requires 2N matmul FLOPs in the forward pass per token seen because each matmul performs one multiplication and one addition per pair of input token and parameter values" (citing Kaplan et al. 2020); attention matmul FLOPs are "much smaller" for large models but attention dominates memory traffic at long context because the KV cache is per-sequence — [Pope et al.](https://arxiv.org/abs/2211.05102)
- Prefill "can run in parallel over L_input" tokens in one forward pass; decode "must run sequentially over L_gen" — [Pope et al.](https://arxiv.org/abs/2211.05102)
- Pope et al.'s weight-gathered layouts reach 76% MFU for prefill at large batch; batch size in tokens (sequences x sequence length) is the quantity that sets the regime — [Pope et al.](https://arxiv.org/abs/2211.05102)
- kipply: "if we're flops bound, making the batch size larger doesn't mean anything is getting faster"; worked Gopher-260B-on-16-GPU example ~22 ms/token at small batch vs ~53 ms/token at batch 512 — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- Databricks' static-batching example: MPT-7B on one A100 goes from 0.9 requests/s at batch 1 to 12.5 requests/s at batch 64 (14x throughput for 64x batch, showing the move toward the compute roof) — [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Continuous batching is standard in production servers (vLLM-style scheduling); Red Hat/Neural Magic's post-acquisition strategy is built around vLLM — [Neural Magic DeepSparse repo notice](https://github.com/neuralmagic/deepsparse)

### Inferences
- FLOPs per token (derived, 2 x N_active): Llama 3 8B 16.1 GFLOP; Llama 3 70B 141 GFLOP; DeepSeek-V3 ~74 GFLOP (37B active). Attention adds ~4 x n_layers x d_model x context FLOPs per token; at 4k context on 70B that is ~4 x 80 x 8192 x 4096 = ~10.7 GFLOP at the end of the prompt (~5.4 GFLOP averaged over prefill), i.e. <5% of the weight matmuls and ignorable at this context length.
- Prefill of a 4,096-token prompt in 1 s (derived, 2 x N x 4096): 8B -> 66 TFLOP/s; 70B -> 578 TFLOP/s; DS-V3 -> 303 TFLOP/s. At 50% MFU: 132 / 1,157 / 606 TFLOPS. One H100 (989 TFLOPS BF16 dense) prefills a 4k prompt for 70B in ~0.6 s at 100% MFU, ~1.2 s at 50%; a CPU at an assumed 50 TFLOPS needs ~12-23 s for the same prompt, and ~1.3-2.6 s for 8B.
- Decode throughput roof when compute-bound: aggregate tok/s_max = FLOPS_achieved / (2 x N_active). On H100 BF16 dense that is ~7,000 tok/s for 70B and ~61,000 tok/s for 8B; on a 50-TFLOPS CPU ~355 tok/s (70B) and ~3,100 tok/s (8B). Since the bandwidth roof at batch B is B x BW/W, the crossover batch is B* = R x b/2 as in A1.
- Design rule for a bandwidth-starved, compute-rich part (e.g. Tenstorrent p150, R~1,300) the right strategy is huge batch; for a bandwidth-starved, compute-poor part (DDR5 CPU, R~80-100) neither regime is attractive for large dense models: batch 1 is bandwidth-capped at ~10 tok/s on 70B-Q4, and batch >25 is compute-capped at a few hundred aggregate tok/s. The CPU sweet spot is small models (8B and below) at moderate batch, or MoE models whose per-token active set is small.

### Gaps
- Real AMX/AVX-512 sustained BF16/INT8 throughput for current Xeon 6 / EPYC Turin SKUs in LLM kernels was not found in a citable source; the 50 TFLOPS figure is the assignment's working assumption, not a measurement.

---

## A4. KV cache: bytes per token per layer (Llama 70B GQA, DeepSeek MLA) and long-context decode bandwidth

### Takeaway
With GQA (8 KV heads, head dim 128) Llama 3 70B stores 4 KiB per token per layer, 320 KiB per token across 80 layers in BF16 (42 GB at 128k context), and DeepSeek-V3's MLA compresses this to ~70 KB per token; at long context or large batch the KV cache, not the weights, dominates the bytes read per decode step.

### Cited Findings
- KV cache bytes per token = 2 (K and V) x bytes per value x n_layers x n_heads x d_head; the 52B example (64 layers, d_model 8192) stores 4 x 64 x 8192 = ~2.1 MB per token; KV-cache FLOPs per token are 2 x 2 x n_layers x d_model^2, exactly 1/6 of a full forward pass — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- KV cache total = batch x seqlen x (d_model / n_heads) x n_layers x 2 (K,V) x 2 bytes (FP16) x n_kv_heads; Llama-2 70B at batch 1, seq 1024 = 0.312 GiB FP16 — [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Llama 3 8B/70B use 8 KV heads (GQA) with 32/80 layers, head dim 128 — [Llama 3 paper](https://arxiv.org/abs/2407.21783)
- DeepSeek-V3 MLA: per-token KV is the compressed latent (d_c = 512) plus the decoupled RoPE key (64), over 61 layers — [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)
- Pope et al.: for PaLM 540B at batch 512 and context 2048 "the KV cache totals 3TB, which is 3 times the size of the model's parameters. The on-chip memory needs to load this KV cache from off-chip memory once for every token generated during which the computational core of the chip is essentially idle"; multiquery attention sharded over batch "enables up to 32x larger context lengths" — [Pope et al.](https://arxiv.org/abs/2211.05102)
- Multi-query attention was introduced precisely because "incremental Transformer inference is limited by the memory bandwidth needed to reload the keys and values tensors" — [Shazeer 2019, arXiv:1911.02150](https://ar5iv.arxiv.org/html/1911.02150)
- KV-cache quantization (INT8/FP8/4-bit), MQA/GQA and shorter context are the standard levers to cut decode bytes — [UbiOps guide](https://ubiops.com/benchmark-and-optimize-llm-inference-performance/)

### Inferences
- Per-token KV (derived, 2 x n_layers x n_kv_heads x d_head x bytes):
  - Llama 3 8B BF16: 2 x 32 x 8 x 128 x 2 = 131,072 B = 128 KiB/token (4 KiB/token/layer).
  - Llama 3 70B BF16: 2 x 80 x 8 x 128 x 2 = 327,680 B = 320 KiB/token (4 KiB/token/layer); this reproduces Databricks' 0.312 GiB per 1,024 tokens exactly. Without GQA (64 KV heads) it would be 2.5 MiB/token, 8x more.
  - DeepSeek-V3 MLA BF16: (512 + 64) x 61 x 2 = 70,272 B = ~70 KB/token, 4.7x smaller than Llama 70B despite 9.5x the parameters.
  - 4-bit KV cache cuts these by 4x (80 KiB/token for 70B).
- KV totals for 70B BF16: 8k context 2.6 GB; 32k 10.5 GB; 128k 42 GB (already ~30% of the BF16 weights and larger than the 4-bit weights).
- Long-context decode bandwidth (derived, bytes per step = W + L x KV_tok): 70B INT8 (70.6 GB) at 32k context reads 81 GB per token (+15%), at 128k reads 113 GB (+60%); 70B 4-bit (40 GB) with BF16 KV at 128k reads 82 GB per token, so KV reads double the time per token. For a DDR5 design with ~450 GB/s this takes 70B-Q4 from ~11 tok/s (short context) to ~5.5 tok/s at 128k unless KV is quantized.
- Batch amplifies this: at batch B with distinct sequences the KV read scales as B x L while weights are read once, so KV dominates when B x L x KV_tok > W; for 70B BF16 that is B x L > 440k tokens (e.g. batch 64 x 7k context). This is the regime Pope et al. describe where the compute core is idle waiting for KV.
- Attention compute at batch B does not amortize over the batch the way weight matmuls do (each sequence has its own KV), so the attention portion of decode stays bandwidth-bound regardless of batch; only weight matmuls move toward the compute roof.

### Gaps
- None material; all numbers are derived from published model configs.

---

## A5. Multi-CPU designs: why aggregate bandwidth only scales under tensor parallelism, and the communication floor

### Takeaway
Pipeline-parallel sharding across n CPUs adds capacity but not batch-1 speed (each token still streams the whole model serially); only tensor parallelism makes n sockets read their shards concurrently, and it costs two all-reduces per layer per step, whose latency term (160 all-reduces per token for an 80-layer model) sets a hard floor of tens of tokens/s on commodity interconnects regardless of memory bandwidth.

### Cited Findings
- Megatron-LM tensor parallelism: each Transformer layer's MLP and self-attention are split so that the forward pass needs one all-reduce after the attention block and one after the MLP block (two per layer); the "trick" of partitioning the first matmul by output axis avoids communication between the two matmuls of a block — [Shoeybi et al., Megatron-LM, arXiv:1909.08053](https://arxiv.org/abs/1909.08053); the same trick is cited by Pope et al. ("there is a 'trick' (Shoeybi et al., 2019) to avoid any cross-chip communication between the matmuls")
- Pope et al.: "Both the weight loading part of the memory time and the non-attention compute time are proportional to the model size and inversely proportional to the number of chips. However, for a given partitioning layout, the time needed for chip-to-chip communication decreases less quickly (or not at all) with the number of chips used, so it becomes an increasingly important bottleneck as the chip count grows" — [Pope et al.](https://arxiv.org/abs/2211.05102)
- Pope et al. 1D weight-stationary feed-forward layer: the all-reduce is split into a reduce-scatter and an all-gather over tensors of shape B x L x E and the communication time is T_comm = 2BLE / network bandwidth (independent of chip count); 2D weight-stationary: T_comm = 8BLE / (sqrt(n_chips) x network bandwidth), which is better once sqrt(n_chips) > d_ff/d_model, i.e. n_chips > 16 when d_ff = 4 d_model; weight-gathered layouts have T_comm = 4E x sqrt(BLF / n_chips) / network bandwidth and win only at large batch — [Pope et al.](https://arxiv.org/abs/2211.05102)
- kipply's small-batch comms term is a pure latency term, ~4 x n_layers x 8 us per decode step on TPU ICI (two all-reduces per layer, each a reduce-scatter plus all-gather), and the per-chip d_model should stay above ~1024 to keep FLOPs-per-byte-of-comms healthy; the batch-512 example still had ~11% of its cost in latency — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- kipply's "FLOPs per byte of comms" for A100 = 312e12 / 300e9 = 1,040, versus 208 FLOPs per byte of HBM — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- Pope et al. combine partitioning with int8 weights to reach 29 ms/token on 64 TPU v4 chips for 540B, and note that at low latency targets int8 gives nearly 2x because the regime is weight-loading bound — [Pope et al.](https://arxiv.org/abs/2211.05102)
- 2026 arXiv work on CPU decode explicitly co-designs model structure for pipeline-friendly CPU decoding (title only; not read): "Pipeline-Native Transformers: Co-Designing Model Architecture and CPU Inference for Bandwidth-Efficient Autoregressive Decode" — [arXiv:2608.23841](https://arxiv.org/pdf/2608.23841); related 2026 work: "Cache-Resident LLM Inference in GB-Scale Last-Level Caches" — [arXiv:2606.25353](https://arxiv.org/pdf/2606.25353); "The Economics of AI Decoding Chips: Rebalancing Compute, Capacity, and Bandwidth" — [arXiv:2607.13068](https://arxiv.org/pdf/2607.13068)

### Inferences
- Communication-floor formula for a tensor-parallel decode step over n devices (derived from Megatron + Pope + kipply):
  t_step >= W/(n x BW_mem) + (KV term) + 2 x n_layers x [ alpha + (2(n-1)/n) x B x E x b / BW_link ]
  where alpha = per-collective latency, B = sequences in the batch (tokens per step), E = d_model, b = bytes per activation element, BW_link = per-device injection bandwidth; the factor 2(n-1)/n is ring reduce-scatter + all-gather traffic. Pipeline parallelism instead sends only B x E x b per stage boundary, (n-1) hops per token, but t_step = sum of stage times = W/BW_mem at batch 1, so PP gives no batch-1 speedup; its throughput scales with n only when >= n micro-batches are in flight.
- Batch-1 numbers for Llama 3 70B (E = 8192, BF16 activations -> 16 KB per all-reduce, 160 all-reduces per token): with alpha = 5 us (NVLink/ICI class) comms = 0.8 ms/token; alpha = 20 us (RDMA/RoCE) = 3.2 ms; alpha = 100-200 us (kernel TCP over Ethernet) = 16-32 ms, which alone caps the system at ~30-60 tok/s even with infinite memory bandwidth. For a 4-socket DDR5 tensor-parallel box on 70B-Q4 (weights 40 GB, ~450 GB/s per socket): memory term 22 ms; with RDMA-class alpha the total is ~25 ms (~40 tok/s, 3.6x one socket); with TCP-class alpha it is ~46 ms (~22 tok/s, only 2x one socket). Low-latency, user-space (or FPGA-synchronized) collectives are therefore the enabling technology for a multi-CPU design, exactly the problem an FPGA interconnect could target.
- Volume term at batch 64 for the same 4-socket system: 160 x 1.5 x 64 x 16 KB = ~246 MB per step; at 100 GbE (12.5 GB/s) that is ~20 ms, comparable to the 22 ms memory term; at 400 GbE / PCIe-5 x16 class (~50 GB/s) ~5 ms. Batch therefore also raises the interconnect bandwidth requirement linearly, so the "cheap CPUs" plan needs either a fast fabric or small batches.
- Expert parallelism (MoE) is the friendlier multi-socket layout: each socket holds a subset of experts and only routed activations move (all-to-all), and the resident model (671 GB FP8) is what forces multi-socket in the first place; this is the layout ktransformers and the M3 Ultra results exploit, and it is where a multi-socket DDR5 design is most defensible.
- Scaling check against the DDR5 ceilings in A2: matching one H100's ~84 tok/s batch-1 on 70B-Q4 needs ~4 TB/s read concurrently, i.e. 8-9 sockets at ~450 GB/s achievable, with 160 sub-20-us all-reduces per token across all of them; aggregate socket cost, power and the comms floor make this unattractive versus one GPU, so the design must win on capacity (large MoE), on $/token at modest batch, or on a niche, not on batch-1 speed for dense 70B.

### Gaps
- No published measurement of all-reduce latency across CPU sockets over CXL, PCIe or Ethernet for LLM decode was found; the alpha values above are illustrative and should be measured on the target fabric.
- The 2026 arXiv papers listed were identified by title in search results only; their contents were not read.

---

## B1. Tenstorrent (Wormhole, Blackhole, QuietBox, Galaxy): prices, measured performance, software maturity

### Takeaway
Tenstorrent is the one vendor that sells accelerator cards retail at GPU-like prices ($999 Wormhole n150s / Blackhole p100a; $1,399 Blackhole p150; $11,999 QuietBox with 4x p150, Oct 2026 store), but every card is bandwidth-poor (448-512 GB/s GDDR6) and independent testing (The Register, Nov 2025) found a single p150 performing about like an NVIDIA DGX Spark at ~41-50% MBU, with an immature, non-standard software stack.

### Cited Findings
- Blackhole store listing (Oct 2026): p100a $999 (28 GB GDDR6, 448 GB/s, 120 Tensix cores, 664 TFLOPS BLOCKFP8, 300 W, active cooling); p150a $1,399 and p150b $1,399 (32 GB GDDR6, 512 GB/s, 120 Tensix cores, 664 TFLOPS BLOCKFP8, 300 W, 4x QSFP-DD 800G ports); "in stock, ships in 2 weeks" — [tenstorrent.com/hardware/blackhole](https://tenstorrent.com/hardware/blackhole)
- Wormhole store listing (Oct 2026): n150s $999 (passive), n150d $1,099 (active, up to 160 W), n300s $1,399 (passive), n300d $1,449 (active, up to 300 W) — [tenstorrent.com/hardware/wormhole](https://tenstorrent.com/hardware/wormhole)
- Tenstorrent downgraded Blackhole p150 from 140 to 120 Tensix cores via firmware and ships 120-core cards going forward, claiming a 1-2% performance drop for existing users — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop)
- The Register's QuietBox review (Nov 27 2025): $11,999 system with 4x p150 (32 GB / 512 GB/s each, ~2 TB/s aggregate), 512 GB host DDR5; single-card Llama 3.1 8B decode reached only ~half of theoretical; Llama 3.3 70B hit ~41% of peak memory-bandwidth utilization; 1 -> 2 cards cut latency ~25% and raised throughput 36% at batch 32; 2 -> 4 cards added 27%; four cards served 1.78x the requests of one; a single p150 "performed almost identically to an NVIDIA DGX Spark"; the reviewer suspects the kernels were written for Wormhole, leaving 76 of 140 Tensix cores idle; accelerators draw ~1,200 W; Ollama and llama.cpp do not work; quantized formats are non-standard; verdict: "trapped in a software blackhole", better suited to developers than local-AI users — [The Register review](https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/)
- Vendor/partner-reported system results: 397.7 tok/s for an 8B MoE model on a QuietBox (4 chips) — [Medium (Arnis)](https://medium.com/@arnis.us/400-tokens-per-second-on-a-12-000-tenstorrent-quietbox-425aaf55bbeb); 350+ tok/s per user for DeepSeek R1 671B across 16 Galaxy units (512 chips) — [gpu.net blog, June 2026](https://blog.gpu.net/posts/2026/june/new-blog-june13/)
- A third-party analysis puts the Galaxy system at ~$110K — [Spheron, "Tenstorrent vs NVIDIA 2026"](https://www.spheron.network/blog/tenstorrent-vs-nvidia-open-source-ai-hardware/) (secondary; not confirmed on tenstorrent.com)
- Independent commentary notes the p150's 512 GB/s caps single-stream decode well below current NVIDIA consumer cards — [Hardware Corner](https://www.hardware-corner.net/tenstorrent-card-is-it-better-then-rtx/)

### Inferences
- Unit economics (derived from store prices): p150 = $2.73 per GB/s of bandwidth and $2.11 per block-FP8 TFLOPS (vendor peak); p100a = $2.23 per GB/s. Against a ~$30K H100 (3.35 TB/s, ~1,979 FP8 TFLOPS dense) that is ~$9/GB/s and ~$15/TFLOPS, so per unit of peak Tenstorrent is 3-7x cheaper, but at 41-50% achieved MBU and with 32 GB per card the batch-1 bound for 70B-Q4 is ~13 tok/s and the model does not even fit on one card.
- A p150's ridge point (~1,300 FLOP/byte) means it only pays off at very large batch (hundreds of tokens per step), i.e. as a throughput/serving part, not a local single-user part; this is the same regime problem a DDR5 design faces but from the opposite side (too much compute per byte rather than too little).
- Buyability and support: yes, individuals can buy today with 2-week lead times, and the stack is open source, but the Register review is the only independent measurement found and it reports kernels that under-use the new silicon; "software maturity" is the dominant risk for anyone building on it, and for a CPU+FPGA competitor it is the most direct price/performance comparator at the $1-12K tier.

### Gaps
- No independent single-card tokens/s table (model x batch) was found; the Register's numbers are in charts not reproduced in text.
- Galaxy pricing and Wormhole n150/n300 memory bandwidth are not on the vendor pages fetched (n150: 12 GB / n300: 24 GB GDDR6 per common spec sheets, unverified here).

---

## B2. Datacenter inference ASICs: Groq, Cerebras, SambaNova, d-Matrix, Etched, Positron, TPU, Trainium/Inferentia, Qualcomm, Gaudi 3, Chinese parts

### Takeaway
Every non-GPU datacenter inference chip that is actually shipping either keeps weights in SRAM (Groq, Cerebras, d-Matrix, partially SambaNova) to escape the DRAM bandwidth wall at batch 1, or is a hyperscaler captive part rented by the hour (TPU Ironwood $12.00/chip-hr on-demand, Trillium $2.70, Inferentia2 $0.76-12.98/hr); none is sold to individuals, and the ones that are cheaper per token than GPUs win on serving throughput, not on low-cost single-user hardware.

### Cited Findings
Groq
- In December 2025 Nvidia agreed to pay about $20B for a non-exclusive license to Groq's LPU technology ($17B cash in three installments through end-2026); founder Jonathan Ross, president Sunny Madra and ~80% of staff joined Nvidia; Groq remains a separate company; sources differ on whether the deal closed in Dec 2025 or Jan 2026 — [CNBC, Aug 24 2026](https://www.cnbc.com/2026/08/24/nvidia-says-groq-racks-will-be-online-this-year-after-20-billion-deal.html); [Jon Peddie Research](https://www.jonpeddie.com/news/nvidias-groq-tie-in/); [inferencechips.com Groq profile](https://www.inferencechips.com/companies/groq/)
- Nvidia unveiled the Groq 3 LPU at GTC March 2026: SRAM-based decode accelerator on Samsung 4nm, paired with Vera Rubin GPUs (GPUs do prefill, LPU does decode); the Groq 3 LPX rack was announced as in full production in August 2026; Nvidia claims 35x throughput per MW vs Blackwell NVL72 for trillion-parameter models (vendor claim) — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip); [CNBC](https://www.cnbc.com/2026/08/24/nvidia-says-groq-racks-will-be-online-this-year-after-20-billion-deal.html)
- GroqCloud Llama 3.3 70B pricing ~ $0.59 input / $0.79 output per Mtok and ~294 tok/s per Artificial Analysis (via secondary); Cerebras's own page quotes Groq at 403-750 tok/s — [Spheron, Groq vs Cerebras 2026](https://www.spheron.network/blog/groq-lpu-vs-cerebras-wse-cheapest-inference-2026/)
Cerebras
- WSE-3 has 900,000 AI cores and 44 GB on-die SRAM; vendor-reported Llama 3.1 70B at ~2,100 tok/s at batch 1; Llama 3.3 70B ~2,100-2,500 tok/s; API pricing ~$0.85/$1.20 per Mtok for Llama 3.3 70B — [Spheron, Cerebras vs H100 2026](https://www.spheron.network/blog/cerebras-vs-nvidia-h100-inference-2026/); [MorphLLM Cerebras pricing, Apr 2026](https://www.morphllm.com/cerebras-pricing)
- Cerebras IPO May 2026: priced at $185/share raising ~$5.55B; fully diluted valuation $56.4B at IPO price; post-pop valuations reported at $66B (Yahoo Finance) and ~$95B (another source), i.e. sources conflict — [Yahoo Finance](https://finance.yahoo.com/markets/stocks/articles/cerebras-raises-5-5b-kicking-163002320.html); [letsdatascience](https://letsdatascience.com/news/cerebras-prices-ipo-raises-555-billion-cf610a05)
SambaNova
- SN40L available via cloud and on-prem; SN50 broad shipments scheduled later in 2026; $1B Series F announced July 2026; claims 200 tok/s on DeepSeek-R1 "independently verified by Artificial Analysis" (vendor-adjacent) — [hashrateindex, Independent AI chip companies 2026](https://hashrateindex.com/blog/independent-ai-chip-companies-ai-asic-market-part-3/); [SambaNova vs Cerebras (vendor blog)](https://sambanova.ai/blog/sambanova-vs-cerebras)
d-Matrix
- Corsair entered full production June 9 2026 with volume shipments to qualified customers; SRAM "Performance Memory" plus up to 256 GB LPDDR5 "Capacity Memory" per card, TSMC N6; SRAM size reported as 2 GB (TechRadar) vs 4 GB at 300 TB/s (Spheron) - conflicting; company-reported 60,000 tok/s on Llama 3 8B from an 8-card system at 1 ms/token and 30,000 tok/s on Llama 3 70B per rack at 2 ms/token; positioned as a decode co-processor beside NVIDIA Blackwell; "a specialized enterprise platform rather than a retail accelerator" — [Converge Digest](https://convergedigest.com/d-matrix-ramps-corsair-ai-inference-platform-into-full-production/); [TechRadar](https://www.techradar.com/pro/tech-startup-proposes-a-novel-way-to-tackle-massive-llms-using-the-fastest-memory-available-to-mankind); [ServeTheHome Hot Chips 2025](https://www.servethehome.com/d-matrix-presents-corsair-an-in-memory-computing-architecture-for-inference-at-hot-chips-2025/)
Etched
- Sohu is a transformer-only ASIC; Etched claims ~500,000 tok/s on Llama 70B from one 8-chip server vs 23,000 for an 8x H100 server (its own extrapolation from 2024 MLPerf); $500M round Jan 2026 at $5B valuation; one source reports $1B of booked (not revenue) orders; a mid-2026 review says no independent benchmarks exist, the chip has not shipped to customers and it cannot run MoE models; another source says first hardware was delivered to Jane Street - conflicting — [TechRadar](https://www.techradar.com/pro/transformers-would-take-over-the-world-little-known-ai-startup-bet-on-asics-and-claims-it-can-trounce-nvidias-flagship-gpu-the-b200-by-almost-10-times); [fast.io Etched review 2026](https://fast.io/resources/etched-ai-review-2026/); [hashrateindex](https://hashrateindex.com/blog/independent-ai-chip-companies-ai-asic-market-part-3/)
Positron
- First-gen Atlas "fabricated by Intel in the US" is shipping, with a 50+-rack deployment at Oracle Cloud; Series B $230M (Feb 2026, >$1B valuation; investors incl. QIA, Arm); Series C $875M (Sept 2026, $5B post-money; co-led by NEA, Atreides, Valor, SemiAnalysis Capital); next-gen Asimov tapes out on TSMC N3P end-2026, production 2H 2027, with 2 TB of memory per accelerator / 8 TB per Titan system (one source says >2,304 GB) — [DCD](https://www.datacenterdynamics.com/en/news/positron-ai-raises-875m-at-a-5bn-valuation-to-support-develop-of-next-generation-hardware/); [HPCwire Series B](https://www.hpcwire.com/off-the-wire/positron-ai-raises-230m-series-b-at-over-1b-valuation-to-scale-energy-efficient-ai-inference/); [Dealroom](https://dealroom.co/news/150061-positron-raises-875m-series-c-at-5b-valuation-for-inference-chips/)
Google TPU (cloud $/chip-hour)
- Ironwood (TPU7x) on-demand $12.00/chip-hr (us-central1), $13.20 (europe-west2); flex-start $6.00; 1-yr $8.40; 3-yr $5.40 (pricing published ~Sept 8 2026, ~3 months after GA); Trillium (v6e) $2.70/chip-hr on-demand, $1.89 1-yr; a $1.60/hr Ironwood rate reported for Anthropic is a negotiated rate, not list — [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing) (page fetch returned title only; figures via [shattered.io](https://shattered.io/google-ironwood-tpu-pricing-vs-nvidia-2026/), [devzero v6e](https://www.devzero.io/instances/gcp/tpu-v6e-1), [Spheron](https://www.spheron.network/blog/google-tpu-v7-ironwood-vs-nvidia-b200-inference-cost/))
AWS Inferentia2 / Trainium2 / Trainium3
- Inf2 on-demand: inf2.xlarge (1 chip, 32 GB HBM) $0.76/hr; inf2.8xlarge $1.97; inf2.24xlarge (6 chips, 192 GB) $6.49; inf2.48xlarge (12 chips, 384 GB) $12.98/hr; 3-yr RI $0.30-$5.19; instance totals 2.3 PFLOPS and 9.8 TB/s; AWS claims "up to 40% better price performance than other comparable Amazon EC2 instances" (no GPU named) — [AWS Inf2 page](https://aws.amazon.com/ec2/instance-types/inf2/)
- Trn2: trn2.3xlarge (1 chip, 96 GB) and trn2.48xlarge (16 chips, 1.5 TB HBM, 20.8 PFLOPS FP8, 46 TB/s); price not shown on the page; AWS claims 30-40% better price-performance than P5e/P5en GPU instances — [AWS Trn2 page](https://aws.amazon.com/ec2/instance-types/trn2/)
- Trainium3 per-chip pricing is not published; ~$1.80/chip-hr is a third-party estimate (instance price / NeuronCore count) — [Spheron Trainium3 2026](https://www.spheron.network/blog/aws-trainium-3-vs-nvidia-h200-b200-llm-training-inference-2026/); trn1.32xlarge listed at $21.50/hr on-demand (June 2026) — [usage.ai](https://www.usage.ai/blogs/aws/ai-ml-cost/trainium-vs-gpu-instances/)
Qualcomm Cloud AI 100 Ultra
- 870 INT8 TOPS (1,044 FP8 TFLOPS), 128 GB LPDDR4x, 150 W; reseller prices $8,500-$11,000 per card, $9,775 liquid-cooled; AI 100 Pro listed at $3,999; Cirrascale rents 8x Ultra at $4,699/month ($3,759 annual); one cloud lists $5.15-6.44/GPU-hr reserved — [megagridsupply](https://megagridsupply.com/ai-hardware/qualcomm-cloud-ai-100-ultra); [itctshop](https://itctshop.com/product/qualcomm-cloud-ai100-ultra/); [Cirrascale pricing](https://www.cirrascale.com/pricing); [cloud-gpus.com](https://cloud-gpus.com/qualcomm-ai-100-ultra/)
Intel Gaudi 3
- Intel's 2024 list: 8-chip baseboard kit $125,000 (~$15,600/chip), reported as ~$16K vs H100 at $30K+; cloud from ~$2.30/hr; Intel projected inference 50% better than H100 on average (2024 vendor projection); a third-party 2026 benchmark blog reports a 10-20% performance gap but 30-45% lower $/Mtok for short-context — [TweakTown](https://www.tweaktown.com/news/98977/intel-discounting-new-gaudi-3-ai-accelerator-16k-against-nvidia-h100-gpu-for-30k/index.html); [Introl Gaudi 3 guide](https://introl.com/blog/intel-gaudi-3-deployment-guide-cost-effective-h100-alternative); [clusterbid 2026](https://clusterbid.com/blog/intel-gaudi-3-llm-inference-2026-benchmarks-roi-vs-h100-h200)
Huawei / Moore Threads / Cambricon
- Ascend 910C reportedly ~60% of H100 inference performance in DeepSeek's tests (Feb 2025, secondhand); Ascend 950PR entered mass production April 2026, Huawei claims 2.87x H20 inference at ~1/4 the cost (vendor), SMIC N+3; US export controls restrict acquisition outside China — [Gigazine](https://gigazine.net/gsc_news/en/20250207-huawei-ascend-910c-inference-performance-nvidia); [Spheron Ascend 950](https://www.spheron.network/blog/huawei-ascend-950-vs-nvidia-b300-b200-llm-inference-2026/); [The Register, Jul 2025](https://www.theregister.com/2025/07/29/huawei_rackscale_boogeyman/)
- Moore Threads MTT S5000: up to 1,000 TFLOPS dense (vendor via TrendForce); H1 2026 revenue 1.736B yuan (+147%); profitable in Q1 2026, loss-making for 2025; Cambricon 2025 revenue 6.497B yuan and profitable; no prices or independent LLM benchmarks found — [iTiger](https://www.itiger.com/news/1107356186); [TrendForce, Apr 2026](https://www.trendforce.com/news/2026/04/28/news-moore-threads-reportedly-turns-profitable-in-1q26-but-remains-loss-making-in-2025-as-rd-hits-rmb-1-3b/)

### Inferences
- Cost per unit (derived from cited prices): Qualcomm AI 100 Ultra ~$10-13 per INT8 TOPS at reseller prices, but its LPDDR4x memory (bandwidth not published on these pages; order of 0.5 TB/s) makes it a batch-throughput part like Tenstorrent, not a batch-1 part. Gaudi 3 at ~$15.6K list is ~half an H100 for comparable HBM bandwidth class (3.7 TB/s per Intel spec, unverified here), the cheapest $/GB/s among HBM parts, but it is sold through OEMs in 8-chip kits.
- Renting beats owning at small scale: Inferentia2 at $0.76/hr for a 32 GB HBM chip, Trillium at $2.70/chip-hr and Ironwood at $12/chip-hr (or $5.40 committed) put datacenter-class bandwidth on an hourly meter; a self-built DDR5 box must beat these $/hr figures at equal tokens/s, which from A2 means beating ~$0.76/hr for a part that streams a 70B-INT8 model at ~10+ tok/s with 32 GB... (it cannot hold 70B-INT8 on one chip; use 2-3 chips at ~$2/hr).
- The SRAM pattern is the physics lesson: Groq (now Nvidia's Groq 3 LPX), Cerebras and d-Matrix all accept tiny per-chip capacity (tens of MB to a few GB SRAM) and huge chip counts to get 10-100x the bytes/s of HBM, which is why they alone reach 1,000-2,500 tok/s per user; their $/token advantage exists only at high utilization across many users, so none of them is a cheaper single-user box.
- Buyability for an individual/small company (derived): Tenstorrent (yes, retail), Qualcomm AI 100 (grey-market resellers), Gaudi 3 (OEM servers), TPU/Trainium/Inferentia (rent only), Groq/Cerebras/SambaNova/d-Matrix/Etched/Positron (API or enterprise sales only), Huawei/Cambricon/Moore Threads (China only). For a CPU+FPGA builder, the realistic retail comparators are Tenstorrent cards, Apple M3 Ultra, DGX Spark-class boxes and consumer GPUs.

### Gaps
- Google's pricing page could not be parsed in this session (title only), so v5e and v5p per-chip-hour prices are not cited; the Ironwood/Trillium figures rely on secondary reports of the same page.
- Trn2/Trn3 on-demand prices are not on AWS's instance page; only third-party estimates were found.
- No independent (non-vendor) tokens/s measurement was found for SambaNova SN40L, d-Matrix Corsair, Etched Sohu or Positron Atlas; Cerebras/Groq figures come via Artificial Analysis as quoted by secondary blogs.
- Whether Positron's Atlas is built on Altera/Intel Agilex FPGAs (as widely reported at its 2024 launch) was not confirmed by the sources fetched; it is directly relevant to a CPU+FPGA proposal and should be verified on positron.ai.
- Cerebras CS-3 system price and WSE-3 on-chip bandwidth were not found in a citable source.

---

## B3. Edge NPUs, defunct/struggling startups, processing-in-memory, and RISC-V vector CPUs

### Takeaway
Edge NPUs (Hailo-10H, Axelera Metis) are cheap per TOPS ($0.70-$1.20/TOPS) but have only single-digit GB/s of memory bandwidth, so they produce single-digit tok/s on 7B models; the 2025 shake-out (Untether, Esperanto, Rain) shows that pure inference-chip startups without a software moat die, and commercially purchasable processing-in-memory for LLMs does not yet exist.

### Cited Findings
Edge NPUs
- Hailo-10H: 40 TOPS INT4 / 20 TOPS INT8 at ~2.5 W; LLM token rates "sit in the single digits" and "the 2.5 GB/s memory bandwidth is the choke point"; Hailo claimed up to 10 tok/s on Llama2-7B; Raspberry Pi AI HAT+ 2 kit listed at $414 — [botmonster review](https://botmonster.com/hardware/hailo-10-ai-accelerator-edge-inference-review/); [CNX Software](https://www.cnx-software.com/news/mini-PC/page/25/?amp=1)
- Axelera Metis: 214 TOPS INT8 at 15 TOPS/W; M.2 card launched (2022) at $149 ("$0.70 per TOPS"); a 2025 guide lists the M.2 card from $259 (unavailable) and an M.2 MAX with up to 16 GB memory for LLM/VLM with no price — [BusinessWire, Dec 2022](https://www.businesswire.com/news/home/20221213005652/de); [eesel Axelera pricing 2025](https://www.eesel.ai/es/blog/axelera-ai-pricing)
Shake-out
- Untether AI stopped supplying/supporting products in June 2025; engineering team acquihired by AMD; AMD did not acquire the technology; bankruptcy filing shows $128M owed to creditors; causes cited: failed fundraising and a late pivot to generative AI — [EE Times](https://www.eetimes.com/untether-ai-shuts-down-engineering-team-joins-amd/); [BetaKit bankruptcy](https://betakit.com/untether-ai-files-for-bankruptcy-following-amd-acquihire/); [zach.be analysis](https://www.zach.be/p/why-did-untether-ai-fail)
- Esperanto wound down its silicon business in 2025, cut HQ headcount ~90%, and sought to sell or license its RISC-V ET-SoC technology; CEO cited hiring raids with 2-4x compensation — [EE Times](https://www.eetimes.com/ai-startup-esperanto-winds-down-silicon-business/)
- Rain AI: OpenAI bought patents rather than the company after acquisition talks failed (reported 2025); Rain reportedly "effectively halted operations" after most staff left; OpenAI's 2019 $51M non-binding LOI is older context — [DigitalToday](https://www.digitaltoday.co.kr/en/view/90637/openai-buys-rain-ai-patents-to-speed-in-house-ai-chip-strategy); [The Decoder](https://the-decoder.com/openai-to-buy-neuromorphic-ai-chips-for-51-million/)
- Mythic (analog compute-in-memory): "ran out of runway" in late 2022, raised $13M in 2023; a CB Insights page shows a $125M raise on 12/17/2025 and an "Acquired" tag with NXP among investors, but the page mixes Kinara (acquired by NXP) and Mythic data and is unreliable — [The Register, Nov 2022](https://www.theregister.com/2022/11/09/mythic_analog_ai_chips/?td=rt-3a); [Edge AI & Vision (2023 raise)](https://edge-ai-vision.com/category/provider/mythic); [CB Insights compare page](https://www.cbinsights.com/compare/deep-vision1-vs-isocline-engineering-corp)
Processing-in-memory
- UPMEM's PIM-DIMM "was not intended for deep-learning applications, thus the peak bandwidth and throughput can hardly meet the requirements of LLM inference" — [LLM Inference Unveiled: Survey and Roofline Model Insights, arXiv:2402.16363](https://arxiv.org/pdf/2402.16363)
- SK hynix AiM: GDDR6-based PIM with ~1 TFLOPS BF16 and 1 TB/s per chip; no LLM results reported in the survey; described as datacenter-only, not consumer-purchasable (community source) — [arXiv:2402.16363](https://arxiv.org/pdf/2402.16363); [DEV Community](https://dev.to/plasmon_imp/if-memory-could-compute-would-we-still-need-gpus-4ccb)
- Samsung/SK hynix mobile PIM designs store two copies of weights; PIM-SHERPA (Mar 2026) removes 47.8-49.7% of that overhead in software — [arXiv:2603.09216](https://arxiv.org/pdf/2603.09216); related: CD-PIM, an LPDDR5-based PIM for low-batch edge LLMs — [arXiv:2601.12298](https://arxiv.org/pdf/2601.12298); CHIME DIMM-PIM for long-context attention — [arXiv:2504.17584](https://arxiv.org/pdf/2504.17584)
RISC-V vector CPUs
- Alibaba DAMO/T-Head XuanTie C950 (Mar 2026): RISC-V server CPU for inference, 3.2 GHz, 5 nm, >3x C920, said to run Qwen/DeepSeek inference with in-house acceleration engines; T-Head's accelerator line is the Zhenwu 810E — [Analytics India Mag](https://analyticsindiamag.com/ai-news/alibaba-unveils-new-chip-as-ai-agent-race-intensifies); [The Standard HK](https://www.thestandard.com.hk/innovation/article/327535/Alibaba-unveils-next-gen-chip-for-agentic-AI-company)
- SiFive Intelligence X390 (2nd gen family Sept 2025): dual vector ALUs and doubled vector length for 4x vector compute, VCIX custom-instruction interface for attached accelerators — [SiFive X390](https://www.sifive.com/cores/intelligence-x390); [The Register, Sept 2025](https://theregister.com/2025/09/08/sifive_ai_cpu_cores)
- Ventana Veyron V2 (announced Nov 2023): 15-wide OoO, 3.6 GHz, 4 nm, up to 192 cores, 512-bit vectors, AI matrix extensions, offered as chiplets/IP; no 2025-2026 shipping news found — [Design & Reuse](https://design-reuse.com/news/news55175.html); [StorageReview](https://www.storagereview.com/news/ventana-veyron-v2-risc-v-processor-announced)

### Inferences
- Edge NPUs are the clearest demonstration of the roofline: Hailo-10H's 20-40 TOPS over 2.5 GB/s gives a ridge of ~8,000-16,000 FLOP/byte, so an LLM at batch 1 uses <0.1% of its compute; a 7B 4-bit model (3.9 GB) at 2.5 GB/s is bounded at 0.6 tok/s from DRAM, so the 5-10 tok/s claims must rely on on-package memory/caching for smaller models. A cheap TOPS number is irrelevant for decode; $/GB/s is what matters.
- The 2025 failures cluster around chips that (a) had no software ecosystem (Untether's ImAIgine SDK was discontinued with the company), (b) targeted CNN-era int8 throughput rather than bandwidth-heavy LLM decode, and (c) could not fund re-spins; Esperanto in particular was a "1,000 RISC-V cores + vectors" design, the closest architectural cousin of a "lots of cheap CPUs" plan, and it did not find a market.
- PIM is the architecture that would most directly attack the DDR5 bandwidth problem (compute next to each DRAM bank), but as of Oct 2026 the only commercial parts (SK hynix AiM) are datacenter-only engagements and the academic work is still fixing basic layout/duplication issues; a small team cannot buy LLM-grade PIM DIMMs.
- RISC-V vector CPUs are being positioned (Alibaba C950, SiFive X390) as inference hosts with attached accelerators via custom instruction interfaces, which is conceptually the same "CPU plus tightly coupled accelerator" idea; none is sold retail and none publishes LLM tok/s.

### Gaps
- No reliable 2025-2026 status for Mythic; no evidence of a commercial Samsung LPDDR-PIM or Micron PIM product for LLMs; no Ventana shipping customers found.
- Hailo-10H and Axelera M.2 MAX standalone 2026 prices were not found (only kit/older prices).

---

## B4. Software and model-level strategies that are cheaper than new hardware (quantization, BitNet, speculative decoding, MoE, hybrid CPU+GPU, KV offload) and API price trends

### Takeaway
Each algorithmic lever attacks the bytes-per-token or steps-per-token term directly: 4-bit halves to quarters the bandwidth need, BitNet b1.58 cuts it ~10x versus BF16 (a synthetic 100B model decodes at 6.6 tok/s on an M2 Ultra), speculative decoding yields 2-3x per-user speed at no extra bandwidth per accepted token, MoE cuts active bytes 18x for DeepSeek, and ktransformers already runs R1 at ~14 tok/s on one GPU plus a DDR5 Xeon; meanwhile API prices for fixed capability fall ~10x/year, so any hardware must be judged against renting at $0.06-$1.20 per million tokens.

### Cited Findings
- BitNet b1.58 / bitnet.cpp: the paper reports 6.58 tok/s for a synthetic 100B configuration on a Mac Studio M2 Ultra (64 GB) and 1.70 tok/s on an Intel i7-13700H; speedups of 1.37-5.07x on ARM and 2.37-6.17x on x86 are measured vs llama.cpp FP16; the released trained checkpoint is 2B4T (GGUF ~1.19 GB); no trained 100B model exists — [Wavect review of the bitnet.cpp paper](https://wavect.io/blog/bitnet-cpu-llm-ram-requirements/); paper: [arXiv:2410.16144](https://arxiv.org/abs/2410.16144)
- Speculative decoding: a small draft model proposes several tokens that the target model verifies in one parallel forward pass, giving 2-3x decode speedups with identical output distribution — [Leviathan et al., arXiv:2211.17192](https://arxiv.org/abs/2211.17192); [Chen et al. (DeepMind), arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- d-Matrix cites Gimlet Labs cutting response time from 24 s to under 2 s by pairing Corsair with GPUs for speculative decoding (vendor-reported) — [Converge Digest](https://convergedigest.com/d-matrix-ramps-corsair-ai-inference-platform-into-full-production/)
- MoE: DeepSeek-V3 activates 37B of 671B parameters per token — [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)
- ktransformers hybrid: DeepSeek-R1 671B Q4_K_M on a 24 GB GPU (14 GB VRAM used) with 382 GB host DRAM; decode 8.73 -> 13.69 tok/s; prefill 54 -> 286 tok/s using Intel AMX MoE kernels (Feb 2025, v0.3 preview) — [ktransformers tutorial](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/en/DeepseekR1_V3_tutorial.md); [TechNode](https://technode.com/2025/02/17/tsinghua-universitys-ktransformers-enables-full-powered-deepseek-r1-with-low-cost-graphics-card/)
- Intel Xeon optimization work for SGLang (July 2025) targets CPU-side MoE/attention kernels — [LMSYS blog](https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization)
- a16z "LLMflation": cost for a fixed MMLU capability fell ~10x per year (2021-2024 data); GPT-3 at $60/Mtok (Nov 2021, MMLU 42) vs Llama 3.2 3B via Together at $0.06/Mtok, a 1,000x drop in 3 years — [a16z](https://www.a16z.com/llmflation-llm-inference-cost)
- Epoch AI measured 9x to 900x per year declines depending on the capability threshold — via [Tech Economics substack](https://techeconomics.substack.com/p/token-deflation-the-treadmill-killing)
- 2026 price cut example: OpenAI cut "GPT-5.6 Luna" by 80%, from $1.00 to $0.20 per million input tokens — [PYMNTS](https://www.pymnts.com/?p=4087603)
- Groq Llama 3.3 70B ~$0.59/$0.79 per Mtok; Cerebras ~$0.85/$1.20 — [Spheron](https://www.spheron.network/blog/groq-lpu-vs-cerebras-wse-cheapest-inference-2026/)
- Cost-per-token formula used in the industry: $/Mtok = (GPU hourly rate / utilization) / (tokens per second x 3,600) x 1,000,000, evaluated at production batch sizes — [SoftwareSeni inference economics](https://www.softwareseni.com/inference-economics-how-to-model-the-true-cost-per-token-across-gpu-architectures)
- SemiAnalysis frames token pricing as a trade-off between per-user speed and tokens per accelerator per second; NVIDIA cites SemiAnalysis InferenceX for a 5x $/Mtok reduction on GPT-OSS-120B within two months of Blackwell launch (vendor-cited) — [WEKA/SemiAnalysis conversation](https://www.weka.io/video/optimizing-ai-inference-economics-or-a-conversation-with-semianalysis); [NVIDIA perspectives](https://perspectives.nvidia.com/ai-infrastructure/total-cost-of-ownership/task/faq/memory-capacity-bandwidth-large-language-models-economics-datacenter/)

### Inferences
- Where each lever lands in the roofline (derived): quantization scales W (BF16 -> Q4 is 3.6x fewer bytes; BitNet 1.58-bit is ~10x fewer, 70B at 1.58 bits = ~14 GB, which would make 70B at ~30 tok/s feasible on one DDR5 socket if a trained 70B BitNet existed); speculative decoding scales the number of tokens produced per weight read (acceptance rate x draft length), so it is the only lever that raises batch-1 tok/s above BW/W on a bandwidth-bound machine; MoE scales N_active; KV quantization/offload scales the L x KV_tok term; batching scales throughput but not per-user latency.
- Build-vs-rent check (derived, assumptions stated): a $10K DDR5 server (3-year straight-line = $0.38/hr) drawing 500 W at $0.12/kWh ($0.06/hr) costs ~$0.44/hr; at ~450 GB/s it produces ~11 tok/s batch-1 on 70B-Q4, and with batch 8-16 perhaps ~60-100 tok/s aggregate before its compute roof (A3), i.e. ~0.2-0.36M tok/hr -> $1.2-2.2 per Mtok. Groq/Cerebras sell Llama 3.3 70B output at $0.79-$1.20 per Mtok and smaller models at a few cents, so the home-built box is 1.5-3x more expensive per token than the cheapest APIs at 100% utilization, before any software cost. The economic case for owning hardware therefore rests on privacy, data residency, availability or a workload the APIs do not serve, not on $/token.
- Because capability-adjusted API prices fall ~10x/yr, a hardware project with an 18-24 month build cycle is competing against prices 30-100x lower than today's at launch; the only hardware bets that have attracted 2026 capital (Positron $875M, Etched $500M, d-Matrix, Cerebras IPO) are datacenter throughput parts, not low-cost single-user devices.
- The ktransformers result is the most relevant existence proof for a CPU-centric design: a dual-socket DDR5 Xeon already reaches ~14 tok/s on the largest open MoE with a $300-1,000 GPU for attention/prefill; a CPU+FPGA design would need to beat that configuration's price (~$6-12K of CPU/DRAM in 2025 prices) or its tokens/s to be interesting.

### Gaps
- No 2026 measurement of bitnet.cpp on current Xeon 6/EPYC Turin; no trained BitNet model above ~2-8B parameters is available, so the 100B numbers are synthetic.
- Artificial Analysis provider pages were not fetched directly; provider tok/s and $/Mtok come via secondary blogs.
- 2026 DDR5 DRAM pricing (needed for $/GB/s of a DDR5 design) was not sourced; DRAM prices reportedly rose sharply in 2025-2026 but no citable figure was collected.

---

## B5. "Lots of cheap CPUs for AI" and CPU-inference-as-a-service: who tried it and how they fared

### Takeaway
The CPU-inference specialists either got absorbed into software companies and had their CPU engines deprecated (Neural Magic -> Red Hat, DeepSparse end-of-life June 2025), stayed niche with vendor-sponsored BERT-era benchmarks (Numenta NuPIC on Xeon), or wound down (Esperanto's 1,000-core RISC-V inference chip); none demonstrated that CPU bandwidth can carry generative decode at GPU-competitive cost.

### Cited Findings
- Red Hat completed its acquisition of Neural Magic on January 13, 2025 — [Network World](https://www.networkworld.com/article/3603635/red-hat-to-acquire-neural-magic.html); [HPCwire, Nov 2024](https://www.hpcwire.com/2024/11/12/red-hat-acquisition-charts-path-to-open-accelerated-ai/)
- Red Hat "ceased development and deprecated the community versions of DeepSparse (including DeepSparse Enterprise), SparseML, SparseZoo, and Sparsify on June 2, 2025"; the repo was archived June 3, 2025; focus "shifted to commercial and open-source offerings built around vLLM" — [neuralmagic/deepsparse GitHub](https://github.com/neuralmagic/deepsparse)
- DeepSparse's premise was sparsity-aware execution keeping data in CPU cache; it is still documented for YOLO on CPUs — [Ultralytics DeepSparse docs](https://docs.ultralytics.com/integrations/neural-magic)
- Numenta NuPIC: 123x BERT inference throughput on Intel Xeon (vendor, 2023), 62x from 4th-gen Xeon AMX over the prior generation; Intel's write-up claims BERT-Large on 5th-gen Xeon beats an NVIDIA A100 by up to 17x (Intel/Numenta co-promotion, Hot Chips 2023) — [BusinessWire, Jan 2023](https://www.businesswire.com/news/home/20230110005274/en); [Intel developer article](https://www.intel.com/content/www/us/en/developer/articles/technical/usher-in-a-new-era-of-accelerated-ai-on-cpus.html); [ServeTheHome](https://www.servethehome.com/?p=72988); [TechRadar](https://www.techradar.com/pro/a-tiny-startup-has-helped-intel-trounce-amd-and-nvidia-in-critical-ai-tests-is-it-game-over-already)
- Ampere Computing bought OnSpecta (2021) and claimed Altra could beat an NVIDIA T4 by ~4x on inference and that DLS gave ~4x on Altra vs ~2x on x86; no newer independent tests found — [The Next Platform, Aug 2021](https://www.nextplatform.com/ai/2021/08/03/ampere-computing-buys-an-ai-inference-performance-leap/1640774); [SDxCentral](https://www.sdxcentral.com/articles/news/amperes-ai-acquisition-puts-intel-nvidia-in-its-crosshairs/2021/08/)
- Esperanto (ET-SoC-1, ~1,000 RISC-V cores for inference) wound down its silicon business in 2025 — [EE Times](https://www.eetimes.com/ai-startup-esperanto-winds-down-silicon-business/)
- Alibaba's XuanTie C950 RISC-V server CPU (2026) is explicitly aimed at LLM inference "when paired with Alibaba's in-house acceleration engines" — [Analytics India Mag](https://analyticsindiamag.com/ai-news/alibaba-unveils-new-chip-as-ai-agent-race-intensifies)
- Academic/industry 2026 work continues on CPU decode (pipeline-native transformers; cache-resident inference in GB-scale LLC) — [arXiv:2608.23841](https://arxiv.org/pdf/2608.23841); [arXiv:2606.25353](https://arxiv.org/pdf/2606.25353)

### Inferences
- The pattern: every CPU-inference success story (Numenta, Neural Magic, Ampere/OnSpecta) was for encoder/CNN workloads where the model fits in cache or where sparsity shrinks the bytes moved; none produced a public benchmark of generative decode for 70B-class models at GPU-competitive $/token, which is what the roofline predicts (A2): a CPU socket has 4-10x less bandwidth than one GPU and the comparison does not improve with batching because the CPU's compute roof is reached early (A3).
- Red Hat's decision to kill DeepSparse in favor of vLLM (GPU-first) within five months of closing the acquisition is the strongest market signal that a CPU-only inference engine is not a business in 2025-2026.
- The surviving "CPU for AI" framing is hybrid and MoE-centric (ktransformers, Intel's SGLang Xeon work, Alibaba C950 + engines, Nvidia's own GPU-prefill + LPU-decode split): the CPU holds the large, sparsely activated parameter set in cheap DDR5 and an accelerator handles prefill/attention. That is the slot a low-cost CPU+FPGA design would most plausibly occupy, provided the FPGA delivers (a) sub-20-us collectives across sockets (A5) and (b) enough prefill FLOPs that 4k-token prompts do not take tens of seconds (A3).

### Gaps
- No public financials or customer counts for Numenta's NuPIC business, and no 2025-2026 updates were found.
- No "CPU inference as a service" startup with published LLM pricing was found; if such services exist they were not surfaced by the searches run.
