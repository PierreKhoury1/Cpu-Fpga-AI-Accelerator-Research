# CPU AI inference capability, memory bandwidth and cost (state of market: October 2026)

Scope note: all figures below are labelled **[peak]** (theoretical, derived from architecture × clock), **[measured]** (benchmark run by the cited party), or **[vendor claim]** (relative marketing multipliers). Prices are USD unless stated; "list" = vendor 1K-unit / RCP price, "street" = retail or reseller observation with date. Where I did my own arithmetic it is labelled **[my arithmetic]** and the assumptions are stated. Research was cut short by a shared web-search quota (no further searches possible after ~45 queries); the Gaps sections record what was not verified.

---

## KQ1. Peak matrix/vector AI throughput per socket (INT8 TOPS, BF16 TFLOPS) and the clock/core assumptions

### Takeaway
A single Intel Granite Rapids socket with AMX has a theoretical dense INT8 peak of roughly 0.5–0.8 POPS (128 cores × 2,048 int8 ops/cycle × 2.0–3.2 GHz), about 5–6× the per-socket peak of AMD's 192-core Turin (AVX-512 VNNI, ~110–180 TOPS) and far above Arm server cores or Apple's SME (~4 TOPS INT8 per P-cluster on M4). These are architectural ceilings at 100 % utilisation; no public source reports the sustained AMX clock under all-core AMX load, so real sustained peaks are lower.

### Cited Findings
- Intel AMX architectural peak: "2048 int8 operations per cycle (vs. 256 without AMX) and 1024 bfloat16 operations per cycle (vs. 64 without AMX)" per core, described as "peak architectural capability ... assuming 100 % CPU utilization" (Aug 2021, pre-production Sapphire Rapids) **[peak]** — [Intel Architecture Day 2021 claims](https://www.intel.com/ArchDay21claims); also reported by [The Next Platform](https://www.nextplatform.com/2021/08/19/with-amx-intel-adds-ai-ml-sparkle-to-sapphire-rapids/)
- Xeon 6980P (Granite Rapids-AP): 128 cores/256 threads, 2.0 GHz base, 3.2 GHz all-core turbo, 3.9 GHz max turbo, 500 W, 12 DDR5 channels, DDR5-6400 (1DPC) or MRDIMM-8800, list $17,800; 6972P 96c 2.4 GHz base $14,600; 6960P 72c 2.7 GHz base, 500 W, $13,750; 6952P 96c 2.1 GHz 400 W $11,400; 6732P 32c 3.8 GHz 350 W $5,295 (8 channels). Redwood Cove cores add AMX-FP16. Compute tiles on Intel 3 — [Wikipedia: Granite Rapids](https://en.wikipedia.org/wiki/Granite_Rapids)
- Xeon 6 E-core (Sierra Forest) 6780E: 144 cores/144 threads, 2.2/3.0 GHz, 330 W, RCP $8,513 (Feb 27 2025); 6766E 144c 250 W $7,692. The architecture lists only AVX-VNNI and AVX-IFMA (256-bit); AVX-512 and AMX are not listed — [Wikipedia: Sierra Forest](https://en.wikipedia.org/wiki/Sierra_Forest)
- Emerald Rapids 8592+: 1.9 GHz base, 2.9 GHz all-core turbo, 350 W, list $11,600; platform supports up to 8-channel DDR5-5600; AMX supported — [Wikipedia: Emerald Rapids](https://en.wikipedia.org/wiki/Emerald_Rapids)
- Sapphire Rapids: 8480+ list $10,710; 8490H 60c 1.9/2.9/3.5 GHz 350 W $17,000; 6454S 2.2/2.8 GHz 270 W $3,157; w9-3495X 56c 1.9 GHz 350 W, 8-ch DDR5-4800, $5,889 — [Wikipedia: Sapphire Rapids](https://en.wikipedia.org/wiki/Sapphire_Rapids)
- AMD Zen 5 (EPYC 9005): "512b datapath / 64b × (2 pipes × 2 ops/lane) = 32 FP64 FLOPS per clock"; AMD's own peak example: 96-core EPYC 9655 × 2.6 GHz base × 32 = 7.99 TFLOPS FP64 **[peak, base clock]** — [AMD blog](https://amd.com/en/blogs/2025/leadership-hpc-performance-with-5th-generation-amd.html)
- Zen 5 has two 512-bit FMA units = 64 FP32 FLOPS/cycle/core — [Chips and Cheese](https://chipsandcheese.com/p/zen-5s-avx-512-frequency-behavior)
- Measured instruction throughput (uops.info): `VPDPBUSD zmm` (INT8 VNNI, 64 MACs/instr) reciprocal throughput 0.50 cycles on Zen 5 (i.e., 2 per cycle), 1.00 measured on Zen 4, 0.50 on Emerald Rapids — [uops.info VPDPBUSD](https://uops.info/html-instr/VPDPBUSD_ZMM_ZMM_ZMM.html); `VDPBF16PS zmm` (BF16 dot, 32 MACs/instr) 0.50 on Zen 5, 1.00 measured on Zen 4, 2.00 on Emerald Rapids (4 µops) — [uops.info VDPBF16PS](https://uops.info/html-instr/VDPBF16PS_ZMM_ZMM_ZMM.html) **[measured]**
- EPYC 9965: 192 Zen 5c cores, 2.25 GHz base, 3.7 GHz boost, 384 MB L3, 500 W, 12 CCDs — [Wikipedia: Zen 5](https://en.wikipedia.org/wiki/Zen_5); 9755: 128c 2.7/4.1 GHz 500 W; 9575F: 64c 3.3/5.0 GHz 400 W; 9175F: 16c 4.2/5.0 GHz 320 W — [Wikipedia: Epyc](https://en.wikipedia.org/wiki/Epyc)
- Ryzen AI 300/400 mobile parts (incl. Strix Halo) use a 256-bit AVX-512 datapath, not the full 512-bit of desktop/server Zen 5 — [Wikipedia: Zen 5](https://en.wikipedia.org/wiki/Zen_5)
- NVIDIA Grace CPU Superchip: 144 Neoverse V2 cores (2 × 72); Grace C1: 72 cores — [NVIDIA](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)
- AWS Graviton4: 96 Neoverse V2 cores, 12 × DDR5-5600 — [WikiChip Fuse](https://fuse.wikichip.org/?p=7633)
- AmpereOne A192-32X: 192 custom Arm cores at 3.2 GHz, 8-ch DDR5-5200, 128 PCIe 5 lanes, list $5,555, rated 276 W — [Phoronix pricing](https://www.phoronix.com/review/ampereone-skus-pricing/2), [Phoronix review](https://www.phoronix.com/review/ampereone-a192-32x/12); AmpereOne M: up to 192 cores and 12 DDR5 channels — [Ampere](https://amperecomputing.com/blogs/5-ways-ampereone-m-enables-efficient-scalable-llm-inference). Chips and Cheese notes AmpereOne's core "is a poor fit if vector/FP performance is a priority" — [Chips and Cheese](https://chipsandcheese.com/p/llvms-ampere1b-commit)
- Apple M4 SME **[measured microbenchmarks, SC-W 2024]**: one P-core driving the SME unit reaches 2,009 FP32 GFLOPS (FMOPA), 2,010 BF16→FP32 GFLOPS (BFMOPA), 2,010 FP16→FP32, 4,017 INT8 GOPS (SMOPA); E-core cluster SME unit: 357 GFLOPS FP32, 715 INT8 GOPS; chip-wide "over 2.3 FP32 TFLOPS"; NEON FMLA per P-core only 113 FP32 GFLOPS; authors conclude M4 SME is "FP32-centric", with only a 2× INT8 speed-up, and note "Nvidia's 72-core Grace CPU has only 3.5× higher FP32 performance than M4's SME unit" — [Remke & Breuer, "Hello SME!", SC-W 2024 (PDF)](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b436/555400b436.pdf); independent LIBXSMM run on Mac mini M4: 1,833 FP32 GFLOPS for 512³ GEMM — [Uni Jena](https://scalable.uni-jena.de/research/2025/03/31/sme-kernels.html)
- NPUs **[peak, vendor]**: Intel Panther Lake NPU 5 "up to 50 TOPS (int8)" — [Wikipedia: Panther Lake](https://en.wikipedia.org/wiki/Panther_Lake_(microprocessor)); Lunar Lake NPU 4 = 48 TOPS ("exactly two more than Lunar Lake") — [The Register](https://theregister.com/2025/10/09/intel_panther_lake); AMD Ryzen AI Max+ 395 (Strix Halo) XDNA 2 NPU 50 TOPS — [Wikipedia: Zen 5](https://en.wikipedia.org/wiki/Zen_5); contradicted by The Register which says Strix Halo "boasts 60 TOPS" — [The Register](https://theregister.com/2025/10/09/intel_panther_lake)

### Inferences
- **Per-socket peak table [my arithmetic]** using the cited per-core rates and cited clocks (low = base clock, high = all-core/boost; actual AMX/AVX-512 sustained clocks are lower and unpublished):
  - Xeon 6980P (AMX): INT8 128 × 2,048 × 2.0–3.2 GHz = **524–839 TOPS**; BF16 **262–419 TFLOPS**.
  - Xeon 6972P (AMX): INT8 96 × 2,048 × 2.4 GHz = **472 TOPS** (base); BF16 236 TFLOPS.
  - Xeon 8592+ (Emerald Rapids, AMX): INT8 64 × 2,048 × 1.9–2.9 GHz = **249–380 TOPS**; BF16 124–190 TFLOPS.
  - Xeon 8480+ (Sapphire Rapids 56c, AMX): INT8 ≈ **230 TOPS** at 2.0 GHz; BF16 ≈ 115 TFLOPS.
  - EPYC 9965 (Zen 5c, AVX-512 VNNI 2×/cycle → 256 int8 ops/cycle/core; BF16 2×/cycle → 128 FLOP/cycle/core): INT8 192 × 256 × 2.25–3.7 GHz = **111–182 TOPS**; BF16 **55–91 TFLOPS**; FP32 28–45 TFLOPS.
  - EPYC 9755: INT8 128 × 256 × 2.7–4.1 GHz = **88–134 TOPS**; BF16 44–67 TFLOPS.
  - EPYC 9575F: INT8 64 × 256 × 3.3–5.0 GHz = **54–82 TOPS**; BF16 27–41 TFLOPS.
  - EPYC 9654 (Zen 4, VNNI/BF16 measured at 1×/cycle): INT8 96 × 128 × 2.4 GHz ≈ **29 TOPS**; BF16 ≈ 15 TFLOPS.
  - Apple M4 (base): ≈ 2.3 TFLOPS FP32/BF16 and ≈ 4–4.7 TOPS INT8 from SME (P-cluster 4.0 + E-cluster 0.7). Scaling to M4 Max/M5 Ultra depends on the number of P-clusters, which I could not source.
  - Strix Halo CPU side: 16 Zen 5 cores with a 256-bit datapath → roughly half the per-core AVX-512 rate of server Zen 5 (≈ 128 int8 ops/cycle/core) → ≈ 6–10 TOPS INT8 at 3.0–5.1 GHz, plus 50 TOPS NPU and the 40-CU iGPU.
- The INT8 AMX peak of one 6980P is ~4.7× that of one EPYC 9965 and ~130× that of the whole M4 SME unit; however KQ3 shows that decode-phase LLM throughput is bandwidth-bound and barely uses this compute, while prefill/batched inference and CNNs do.
- Sierra Forest (6780E) has no AMX and no AVX-512; its AI value is as a many-core AVX-VNNI part. No per-core VNNI rate for Crestmont E-cores was sourced.

### Gaps
- No primary source for sustained AMX frequency ("AMX license") on Granite Rapids; Intel does not publish it, so the 524–839 TOPS range is bracketed by base and all-core turbo only.
- No source retrieved for Neoverse V2/V3 per-core SDOT/I8MM throughput (needed for Graviton4/Grace INT8 peak); no SME-capable Arm server chip benchmarks found.
- Apple M5 / M5 Ultra SME throughput not measured publicly in the sources found; Apple does not publish Neural Engine TOPS on the Mac Studio spec page.
- Arrow Lake NPU TOPS (~13) not verified (search budget exhausted).

---

## KQ2. Memory bandwidth per socket: theoretical vs measured (STREAM/likwid/MLC)

### Takeaway
Theoretical per-socket bandwidth is 614 GB/s for 12-ch DDR5-6400 (Turin, Xeon 6900P), 845 GB/s with MRDIMM-8800 (Xeon 6900P only), 538 GB/s for Graviton4, 512 GB/s per Grace die (1,024 GB/s per Superchip), 819 GB/s (M3 Ultra) to 1.2 TB/s (M5 Ultra) for Apple, and 256 GB/s for Strix Halo. Measured efficiency ranges from ~99 % of theoretical on Turin reads (Chips and Cheese) down to 73–80 % on Genoa in llama.cpp-style load tests; dual-socket aggregate adds capacity but LLM decode does not scale with it.

### Cited Findings
- EPYC 9575F (Turin) with DDR5-6000 (6400 only on validated 1DPC systems): theoretical 576 GB/s; **[measured]** full-socket reads ≈ 99 % of theoretical (~570 GB/s), writes 435 GB/s, add 453 GB/s; single thread 52 GB/s read; 2DPC drops to DDR5-4400 — [Chips and Cheese](https://chipsandcheese.com/p/amds-turin-5th-gen-epyc-launched)
- STH's Turin platform ran DDR5-6400 RDIMMs at 6000 MT/s; AMD initially specified DDR5-6000 then validated 6400 on customer platforms; STREAM charts given only as images — [ServeTheHome](https://servethehome.com/amd-epyc-9005-turin-turns-transcendent-performance-solidigm-broadcom/5)
- likwid-bench load **[measured]**: 1× EPYC 9374F, 12× DDR5-4800: 361.5 of 460.8 GB/s (78 %); 2× EPYC 9175F, 16× DDR5-6400 (8 ch/socket): 378.3 of 409.6 GB/s per socket (92 %), 753.6 of 819.2 dual; 2× EPYC 9654, 24× DDR5-4800: 359.9 of 460.8 per socket (78 %), 679.2 of 921.6 dual (74 %) — [llama.cpp discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- Intel MLC **[measured]**: Threadripper PRO 7965WX ≈ 225–250 GB/s; dual EPYC Genoa ≈ 730 GB/s aggregate reads; EPYC 7532 Rome ≈ 160–173 GB/s — [Level1Techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826)
- Xeon 6900P: 12 channels at 6400 MT/s on the 500 W AP platform; SP (6700P) 8 channels — [Chips and Cheese interview with Intel's Ronak Singhal](https://chipsandcheese.com/p/chips-and-cheese-interviews-ronak-singhal)
- Intel claims MRDIMM-8800 gives 1.07–1.33× over DDR5-6400 across 8 workloads **[vendor claim]** — [Intel Xeon 6 performance index](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/intel-xeon-6); Intel also claims 1.31× STREAM; Phoronix's independent dual-6980P test found MRDIMM-8800 wins every synthetic bandwidth test, HPCG +33 %, AMG ~+30 %, but only ~8 % geomean across 75 benchmarks (2024) **[measured]** — [Phoronix](https://www.phoronix.com/review/intel-xeon6-mrdimm-ddr5)
- SGLang/Intel dual-6980P system: 24 × 64 GB MRDIMM "rated 12800 MT/s, running at 8800 MT/s"; their INT8 MoE kernels reach **1.45 TB/s effective across both sockets (85 % efficiency)** **[measured, vendor-authored]** — [LMSYS/Intel blog, Jul 2025](https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization)
- Intel's MLPerf v5.0 CPU system: 2 × Xeon 6980P, 24 slots/socket × 96 GB MRDIMM 8800 MT/s = 2,304 GB — [MLCommons results repo](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/systems/1-node-2S-GNR_128C.json)
- Intel memory roadmap (Jul 2026): Xeon 6 now supports 8000 MT/s DDR5; second-generation MRDIMM at 8800 MT/s for Xeon 6900P slated for Q1 2027; Intel had not provided 6700P/Xeon 6+ review samples — [Phoronix news](https://phoronix.com/news/Intel-More-Xeon-6-MRDIMMs)
- Graviton4: 12-ch DDR5-5600, theoretical 537.6 GB/s (5.6 GB/s per core), vs 307.2 GB/s on Graviton3 **[peak]** — [WikiChip](https://en.wikichip.org/wiki/annapurna%20labs/graviton/graviton4)
- NVIDIA Grace Superchip: LPDDR5X 240/480 GB at up to 1,024 GB/s, 960 GB at 768 GB/s; Grace C1 (72 cores): 120/240 GB at 512 GB/s, 480 GB at 384 GB/s; NVLink-C2C 900 GB/s between dies **[peak]** — [NVIDIA](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)
- Apple **[peak, vendor spec]**: M4 Max up to 546 GB/s, 128 GB — [Apple newsroom Oct 2024](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/); M3 Ultra 819 GB/s, up to 512 GB — [Apple support](https://support.apple.com/en-mk/122211); M5 Max 460 GB/s (32-core GPU) or 614 GB/s (40-core GPU, up to 128 GB); M5 Ultra 1.2 TB/s, 96/256/512 GB — [Apple Mac Studio specs](https://www.apple.com/mac-studio/specs/)
- Strix Halo (Ryzen AI Max+ 395): soldered LPDDR5X on a 256-bit bus — [Wikipedia: Zen 5](https://en.wikipedia.org/wiki/Zen_5); GMKtec EVO-X2 ships LPDDR5X-8000 — [Yahoo Tech](https://tech.yahoo.com/computing/articles/gmktec-evo-x2-workstation-pc-030300493.html); Framework's reference configuration footnote cites LPDDR5X-8533 — [Framework](https://frame.work/desktop)
- AmpereOne A4 vs E6 (OCI) STREAM Triad: 143 GB/s vs 53 GB/s per instance **[vendor]** — [Ampere brief](https://amperecomputing.com/briefs/llama-3-ai-inference-on-ampereone)
- Threadripper 9995WX reviewed on TRX50 with quad-channel DDR5 — [Phoronix](https://www.phoronix.com/review/amd-threadripper-9995wx-trx50)

### Inferences
- **Theoretical per-socket table [my arithmetic, 8 B × channels × MT/s]**: 12-ch DDR5-6400 = 614 GB/s (Turin at 6400; Xeon 6900P); 12-ch DDR5-6000 = 576 GB/s (typical Turin); 12-ch MRDIMM-8800 = 845 GB/s (Xeon 6900P); 8-ch DDR5-6400 = 410 GB/s (Xeon 6700P/E, Turin in 8-ch boards, TR PRO 9000 WRX90); 8-ch DDR5-5600 = 358 GB/s (Emerald Rapids); 8-ch DDR5-4800 = 307 GB/s (Sapphire Rapids); 8-ch DDR5-5200 = 333 GB/s (AmpereOne A192); 4-ch DDR5-6400 = 205 GB/s (TRX50 Threadripper 9000X); 256-bit LPDDR5X-8000 = 256 GB/s (Strix Halo).
- Realistic sustained LLM-decode bandwidth is 75–92 % of theoretical on x86 (likwid), ~99 % on Turin for pure reads; the SGLang team's 1.45 TB/s on dual MRDIMM-8800 6980P (85 % of 1.69 TB/s) is the highest documented CPU figure.
- Dual-socket doubles capacity and raw bandwidth, but KQ3 shows LLM decode scales 1.0–1.8× at best, so for bandwidth-bound workloads a second socket is mostly capacity, not speed.

### Gaps
- No independently measured STREAM number for a single Xeon 6980P with MRDIMM-8800 (Phoronix pages are bot-walled; STH gives charts only). Intel's 1.31× STREAM claim is marketing.
- No measured bandwidth for Graviton4, Grace, AmpereOne or Apple M3/M5 Ultra from the sources retrieved (Apple GPU-side sustained bandwidth is typically below the spec figure; not quantified here).
- Threadripper 9000 series channel counts/speeds (TRX50 4-ch vs WRX90 8-ch) inferred from the Phoronix title; not verified against AMD spec.

---

## KQ3. Measured LLM (and CNN) inference throughput on CPUs: llama.cpp, SGLang/IPEX (AMX), vLLM CPU, ktransformers; prefill vs decode; batch scaling

### Takeaway
Best documented CPU-only decode rates are ~15 tok/s for DeepSeek-R1-671B INT8 and ~13 tok/s for a 70B INT8 dense model on a dual Xeon 6980P with MRDIMM-8800 under SGLang+AMX (batch 1), with prefill ~400–900 tok/s; llama.cpp on the same box is 2–4× slower for decode and 6–14× slower for prefill. Commodity dense-70B Q4 decode is ~5 tok/s on Strix Halo, 5–10 tok/s on a 64-vCPU Graviton4 slice, 2.3 tok/s (F16) per Turin socket, and dual-socket scaling in llama.cpp is 1.0–1.8×. Batched serving on Xeon 6 (vLLM, Llama 3 8B BF16) delivers ~30 tok/s/user at ≤8 users and ~8 M output tokens/kWh at 128 users.

### Cited Findings
**Intel Xeon 6 with AMX (SGLang CPU backend), dual 6980P, 1.5 TB MRDIMM-8800, batch 1, 1024 in/1024 out [measured, Intel/LMSYS authors, Jul 2025]**
- DeepSeek-R1-671B INT8 (w8a8): TTFT 1,885 ms, TPOT 68.0 ms (≈ 543 tok/s prefill, ≈ 14.7 tok/s decode); llama.cpp Q8 on same box: TTFT 24,547 ms, TPOT 172 ms (≈ 42 tok/s prefill, ≈ 5.8 tok/s decode) → 13.0× / 2.5× speed-ups. DeepSeek-R1 FP8 (emulated): TTFT 2,235 ms, TPOT 77.7 ms (≈ 12.9 tok/s). Qwen3-235B-A22B INT8: TTFT 1,164 ms, TPOT 51.8 ms (≈ 19.3 tok/s decode; llama.cpp 4.7 tok/s). DeepSeek-R1-Distill-Llama-70B INT8: TTFT 2,638 ms, TPOT 76.5 ms (≈ 13.1 tok/s; llama.cpp ≈ 5.1 tok/s, TTFT 20.3 s). Llama-3.2-3B BF16 (1 socket): TTFT 268 ms, TPOT 17.0 ms (≈ 59 tok/s; llama.cpp 18 tok/s). Multi-NUMA overhead ~3 %; backend requires AMX; FP8 reaches 80–90 % of INT8 speed — [LMSYS blog](https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization) (tok/s conversions are my arithmetic from the stated latencies)

**ktransformers (CPU experts + 1 GPU), dual Xeon Gold 6454S (2 × 32 cores, Sapphire Rapids, AMX) + RTX 4090D 24 GB, DeepSeek-V3/R1 [measured, project-reported, early 2025]**
- V0.2 Q4_K_M, 500-token prompt: dual socket 8 experts: prefill 82.9 tok/s, decode 12.2 tok/s; 6 experts: 97.3 / 13.7; single socket 8 experts: 54.2 / 8.7; llama.cpp 2 × 32 cores: 10.3 / 4.5. V0.3 preview with AMX INT8 MoE kernel: prefill 255 tok/s (8 experts, 2K prompt), 287 tok/s (6 experts) = up to 27.8× llama.cpp prefill; decode unchanged (~13.7). Requires 1 TB DRAM dual-socket. "Collaborating with Intel to ... optimize for Xeon6/MRDIMM" — [ktransformers tutorial](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/en/DeepseekR1_V3_tutorial.md); coverage noted AMX dependency: "CPUs from other brands are not yet capable" — [TechNode](https://technode.com/2025/02/17/tsinghua-universitys-ktransformers-enables-full-powered-deepseek-r1-with-low-cost-graphics-card/)

**vLLM CPU backend on Xeon 6 (Lenovo, Jul 2026), Llama 3.1-8B BF16, vLLM 0.21+cpu [measured]**
- 6972P (96c, 500 W), 6960P (72c, 500 W), 6732P (32c, 350 W): at 1–8 users per-user throughput "approaching 30 tokens/s/user with sub-200 ms TTFT"; at 128 users TTFT 1.9–2.0 s (6972P/6960P) vs ~4.8 s (6732P) with per-user rate "toward single digits"; power efficiency: 6972P ≈ 7 M output tokens/kWh at 64 users, 8 M at 128 users; 6960P ≈ 5.7 M / 6.5 M; 6732P ≈ 3.2 M / 4 M — [Lenovo Press LP2463](https://lenovopress.lenovo.com/lp2463.pdf)

**llama.cpp on AMD EPYC (single vs dual socket) [measured, community, Feb 2025]**
- Llama-3.1-70B **F16** (131 GiB): 1× 9175F (16c, 8-ch DDR5-6400): pp512 21.5 tok/s, tg128 2.35 tok/s; dual 9175F: 39.3 / 4.30 (1.83× with a NUMA "warm-up" trick; 2.27 tok/s without it). 2× 9654 (96c, DDR5-4800): single socket 28.4 / 2.19, dual 48.0 / 3.97. 1× 9374F (NPS2): 12.9 / 1.25.
- Mixtral 8x22B Q8_0: 9175F single 15.3 / 6.80, dual 29.3 / 10.6; 9654 single 31.4 / 7.02, dual 55.9 / 10.2.
- DeepSeek R1 Q4_K_S: 9175F single socket pp 19.9 / tg **9.08**, dual 34.3 / 9.79 (decode +8 % only); 2× 9654 Q4_K_S 32.9/8.48 → 48.0/8.67; Q8_0 33.0/6.03 → 49.2/6.69. Root causes: NUMA-unaware weight placement and barrier/sync overhead on small matrices (small-matrix GEMM scales 105 % across sockets vs 163 % for large) — [llama.cpp discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- Threadripper PRO 7965WX (24c, 256 GB DDR5): DeepSeek R1 UD-Q2_K_XL (211 GiB) pp512 27.6, tg128 8.44 tok/s; Q4_K_M (377 GiB, exceeds RAM) 1.12 tok/s; EPYC 7532 Rome 512 GB DDR4-3200: Q2_K_XL tg 4.69 tok/s with `--numa distribute` — [Level1Techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826)
- Reported (secondary): dual EPYC + 768 GB DDR5 build "6–8 tokens per second" on R1 671B — [Notebookcheck](https://www.notebookcheck.net/Way-to-run-DeepSeek-s-671B-AI-model-without-expensive-GPUs-discovered.956421.0.html); single 9534 with 12-channel "7 tokens/sec", 9684X "close to 10 tok/s on Q8" (forum comments, unverified) — [Level1Techs p4](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826?page=4)
- Phoronix (Dec 2025), llama.cpp Llama-3.1-Tulu-3-8B-Q8_0: EPYC 9755 2P fastest at text generation, Xeon 6980P 2P fastest at prompt processing (numbers in images, not extracted) — [Phoronix](https://www.phoronix.com/review/xeon-6980p-epyc-9755-2025/4); Phoronix AMX study: Granite Rapids AMX gives "much better" prompt-processing on DeepSeek-R1-Distill-Llama-8B-Q8_0 — [Phoronix](https://www.phoronix.com/review/intel-xeon-6-granite-rapids-amx/5)

**Arm servers**
- Graviton4 c8g.16xlarge (64 vCPU), llama.cpp b3440, Llama 3 70B Q4_K_M, batch 1: "5–10 tokens per second" (conclusion says 10); Graviton3 cannot hold 70B; Llama 3 8B Q4_0 prompt encoding +14–26 % and generation +5–50 % vs Graviton3 — [Arm blog](https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/running-llama-3-70b-on-aws-graviton4)
- Graviton4 16 vCPU, Arcee AFM-4.5B: ≈ 25 tok/s F16, 40 Q8_0, 60 Q4_0 — [Arm Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/arcee-foundation-model-on-aws/07_evaluating_the_quantized_models/); r8g.8xlarge (32 vCPU) Virtuoso-Small: 17.2 FP16 / 28.5 Q8 / 44.5 Q4 tok/s — [AWS APN blog](https://aws.amazon.com/blogs/apn/running-genai-inference-with-aws-graviton-and-arcee-ai-models/); Llama-3.2-3B Q4_0 on c8g.16xlarge: +30.6 % throughput and +22 % latency vs Graviton3 (absolute values only in images) — [AWS Builder](https://builder.aws.com/content/2pViH36qEfqDYwv0kEysJ0kqDp4/small-language-models-slms-inference-with-llama-cpp-on-graviton4)
- AmpereOne A192-32X (192c), 8-bit models, SLA ≥ 10 tok/s per user: Llama 3.2 1B → 64 concurrent users; 3B → 32; Llama 3.1 8B → 8; Llama 3 70B and 405B "n/a" (cannot meet 10 tok/s); server draws 577 W under SPECint load **[vendor]** — [Ampere](https://amperecomputing.com/blogs/llms-bigger-is-not-always-better); Oracle A1 (Ampere Altra) Llama 3 8B up to 115 tok/s aggregate single node, llama.cpp **[vendor]** — [Oracle](https://blogs.oracle.com/ai-and-datascience/post/introducing-meta-llama-3-on-oci-ampere-a1)

**Apple M3 Ultra Mac Studio 512 GB [measured]**
- DeepSeek R1 671B 4-bit: ≈ 17–18 tok/s at 160–180 W (short context) — [AppleInsider, Mar 2025](https://appleinsider.com/articles/25/03/18/heavily-upgraded-m3-ultra-mac-studio-is-great-for-ai-projects); 15.78 tok/s Ollama/GGUF vs 19.17 tok/s MLX — [QbitAI](https://www.qbitai.com/2025/03/263949.html); with an ~11,000-token prompt: llama.cpp 76.8 tok/s prefill / **4.26 tok/s** decode vs MLX 4-bit 189 / 11.2 tok/s; MLX peak memory 422 GB; macOS VRAM cap must be raised (default 384 GB) — [dev.to](https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4)
- M4 Max vs M5 Max single test: 79.1 vs 88.5 tok/s (model unstated) — [video summary](https://videohighlight.com/v/XGe7ldwFLSE)

**AMD Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X-8000) llama.cpp, mostly Vulkan [measured, community 2025–26]**
- Llama 3.1 70B Q4_K_M (39.6 GiB): pp512 81.6 / tg128 **5.1 tok/s**; Qwen2.5 32B Q4_K_M: 242 / 11.3; Qwen2.5 14B Q4: 600 / 24.5; Llama 2 7B Q4_K_M: 1,074 / 47.3; Llama 3.2 3B Q8_0: 2,248 / 60.9; MoE: Qwen3 30B-A3B Q4_K_M 1,142 / 86.1; GPT-OSS 120B 120 / 53.4; Qwen3-235B-A22B Q3_K_M 101 / 17.2 — [visorcraft/strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf)
- Framework Desktop llama-bench: Hermes-4-70B Q4_K_M ROCm 121.9 / 4.6, Vulkan 76.5 / 5.26; gpt-oss-120B ~36 tg; Qwen3.6-35B-A3B Q8 53 tg; dense 27B Q8 ~7.7 tg; power "under 100 W" during MoE runs — [Level1Techs](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466)

**CNN / classic inference (MLPerf Inference v5.0, Apr 2025, closed division, CPU-only, 2× Xeon 6980P, 2.3 TB MRDIMM-8800, PyTorch) [measured, audited]**
- ResNet-50 Offline **45,530 samples/s**; Server 40,185 samples/s at 14.8 ms p99 — [resnet50 Offline](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/resnet50/Offline/performance/run_1/mlperf_log_summary.txt), [Server](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/resnet50/Server/performance/run_1/mlperf_log_summary.txt)
- RetinaNet Offline 713.8 samples/s — [retinanet](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/retinanet/Offline/performance/run_1/mlperf_log_summary.txt); DLRM-v2 99.9 Offline 18,686 samples/s — [dlrm-v2](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/dlrm-v2-99.9/Offline/performance/run_1/mlperf_log_summary.txt)
- GPT-J 6B (99): Offline 7.49 samples/s ≈ **517 tok/s**; Server 3.84 samples/s ≈ 265 tok/s at 17.1 s p99 — [gptj Offline](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/gptj-99/Offline/performance/run_1/mlperf_log_summary.txt), [Server](https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/closed/Intel/results/1-node-2S-GNR_128C/gptj-99/Server/performance/run_1/mlperf_log_summary.txt). Intel's v5.0 CPU submissions cover 3d-unet, dlrm-v2, gptj, resnet50, retinanet, rgat; **no Llama-2-70B CPU-only entry** — [results tree](https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/Intel/results/1-node-2S-GNR_128C)
- Intel: Xeon 6 P-core up to 1.9× generational gain in MLPerf v5.1 (Sep 9 2025) — [Intel](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-arc-pro-b-series-gpus-and-xeon-6-shine-in-mlperf-inference-v5-1.html); in v6.0 (Apr 1 2026) Intel is "the only server processor vendor to submit stand-alone CPU results" — [Intel](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-delivers-ai-performance-mlperf-inference-v6-0.html)

**Intel relative claims (vendor, IPEX, Nov 2024–2025)**: 2× 6980P MRDIMM-8800 BF16, 128/128 tokens, batch 16, p90 2nd-token < 100 ms: 1.38× (Llama-2-7B), 1.23× (Llama-3.1-8B), 1.28× (Phi-3.5-mini), 1.20× (GPT-J-6B) vs 2× EPYC 9965 (AMD's own ZenDNN 5.0 numbers); INT8 ResNet-50 2.17× vs 9755 / 1.79× vs 9965; BERT-Large 1.85×/1.57×; DLRM 1.87×/1.93×; 2× 6972P Llama-2-7B int8 chatbot 3.06× perf and 1.86× perf/W vs 2× EPYC 9654 — [Intel Xeon 6 performance index](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/intel-xeon-6)

### Inferences
- **Effective decode bandwidth [my arithmetic = active bytes/token × tok/s]**: Strix Halo 70B Q4 (39.6 GB) × 5.1 = 202 GB/s ≈ 79 % of 256 GB/s; single 9175F 70B F16 (131 GB) × 2.35 = 308 GB/s ≈ 81 % of likwid-measured 378 GB/s; dual-6980P SGLang DeepSeek-R1 INT8 (~37 GB active/token) × 14.7 ≈ 545 GB/s, only ~32 % of theoretical 1.69 TB/s — MoE routing, attention/MLA and NUMA overheads cost most of the headroom; M3 Ultra R1 4-bit (~21 GB active) × 17.5 ≈ 370 GB/s ≈ 45 % of 819 GB/s. Rule of thumb: single-stream CPU decode realises 30–80 % of nominal bandwidth, best for dense models.
- **Prefill is where AMX matters**: SGLang on dual 6980P prefill is ~540–880 tok/s vs 42–61 tok/s for llama.cpp (which uses little AMX); ktransformers' AMX kernel raised prefill 3.45× on Sapphire Rapids. Without AMX (EPYC, Arm, Apple/llama.cpp), prefill of 70B-class models is ~20–80 tok/s per socket — minutes for long prompts.
- **Batching on CPU scales aggregate throughput but not linearly**: Lenovo's 6972P moves from ~30 tok/s per user (≤ 8 users) to "single digits" per user at 128 users; the MLPerf GPT-J Offline number (517 tok/s on 2 sockets for a 6B model) is a reasonable upper bound for fully batched CPU LLM throughput per socket pair in 2025 software.
- Dual-socket llama.cpp decode scaling is ~1.0–1.1× for MoE (DeepSeek) and ~1.8× for dense only with a workaround, so "ganging sockets" under llama.cpp mostly buys capacity; SGLang's 3 % multi-NUMA overhead shows purpose-built tensor-parallel CPU runtimes can do far better, but only on AMX hardware.

### Gaps
- No absolute llama.cpp tokens/s for 6980P or EPYC 9965/9755 could be extracted (Phoronix/OpenBenchmarking are Cloudflare-walled to this agent; data exist at [OpenBenchmarking 2x9965 llama](https://openbenchmarking.org/result/2512117-PTS-LLAMABLE20) and [Phoronix Dec 2025](https://www.phoronix.com/review/xeon-6980p-epyc-9755-2025/4)).
- No llama.cpp numbers for NVIDIA Grace CPU-only or for a 192-core AmpereOne; Ampere publishes only SLA-concurrency charts.
- No measured Llama-2/3-70B CPU-only MLPerf result exists in v5.0 (Intel did not submit it); v5.1/v6.0 per-benchmark CPU scores not retrieved.
- No measured numbers for Intel IPEX-LLM INT4 on Xeon 6 beyond relative multipliers.

---

## KQ4. Prices (USD): CPUs, platforms, memory, Apple and Strix Halo systems

### Takeaway
Top server CPUs list at $12–18k (9755 $12,984; 9965 $14,813; 6980P $17,800) with street roughly $13–15k in mid-2026, while the real 2026 cost driver is DRAM: 64 GB DDR5 RDIMMs went from ~$255 (Q3 2025) to >$900 contract (Q1 2026) and $2,000+ retail (Aug 2026), so a 768 GB 12-channel socket's memory now costs more than the CPU. Cheapest high-bandwidth memory per dollar is the Mac Studio M5 Ultra ($5,499 for 96 GB at 1.2 TB/s) and Strix Halo mini-PCs (~$1,700–2,000 for 128 GB at 256 GB/s).

### Cited Findings
**Server CPUs (list)**
- Intel: 6980P $17,800 (also "Intel's most expensive CPU ... more than twice ... EPYC 9754 at $11,900" — [TechRadar](https://www.techradar.com/pro/intels-128-core-wonder-processor-is-also-its-most-expensive-cpu-right-now-xeon-6980p-costs-more-than-twice-its-amds-128-core-rival-the-epyc-9754)); 6979P $15,750; 6972P $14,600; 6960P $13,750; 6952P $11,400; 6732P $5,295; 6761P $6,570 — [Wikipedia: Granite Rapids](https://en.wikipedia.org/wiki/Granite_Rapids); 6780E $8,513 RCP (Feb 2025) — [Wikipedia: Sierra Forest](https://en.wikipedia.org/wiki/Sierra_Forest) (another listing gives $9,535 — [pchardware](https://pchardware.org/processors/intel-xeon-6780e/)); 8592+ $11,600, 8580 $10,710, 8558 $4,650 — [Wikipedia: Emerald Rapids](https://en.wikipedia.org/wiki/Emerald_Rapids); 8480+ $10,710, 6454S $3,157 — [Wikipedia: Sapphire Rapids](https://en.wikipedia.org/wiki/Sapphire_Rapids)
- AMD Turin: 9965 $14,813; 9845 $13,564; 9755 $12,984; 9655 $11,852 (9655P $10,811); 9575F $11,791; 9555 $9,826; 9355 $3,694; 9175F $4,256. Genoa: 9654 $11,805; 9554 $9,087; 9374F $4,860; 9334 $2,990 — [Wikipedia: Epyc](https://en.wikipedia.org/wiki/Epyc); Phoronix confirms "EPYC 9965 is at $14,813" vs AmpereOne A192-32X $5,555 — [Phoronix](https://www.phoronix.com/review/ampereone-skus-pricing/2)
- Street (mid-2026, implied from CPUBenchmark value = CPU Mark/price, Jul 14 2026, my back-calculation): 6980P ≈ $14,000; 9755 ≈ $13,000; 9965 ≈ $14,900; 9575F ≈ $7,700 — [CPUBenchmark 6980P vs 9755](https://www.cpubenchmark.net/compare/7220vs6702/Intel-Xeon-6980P-vs-AMD-EPYC-9755), [9965 vs 9575F](https://www.cpubenchmark.net/compare/6323vs6317/AMD-EPYC-9965-vs-AMD-EPYC-9575F); Newegg.ca lists 9575F at $8,999.99 (currency likely CAD) — [Newegg.ca](https://www.newegg.ca/amd-epyc-9575f-socket-sp5/p/N82E16819113867); grey-market Alibaba 9965 ≈ $4,200–5,955 (unverified) — [Alibaba](https://electronics.alibaba.com/product/amd-epyc-price)
- AmpereOne A192-32X: $5,555 list; Newegg listing $4,999 (out of stock) — [Phoronix](https://www.phoronix.com/review/ampereone-skus-pricing/2), [Newegg Ampere store](https://www.newegg.com/Ampere/BrandStore/ID-225172)
- NVIDIA Grace: no published list price — [NVIDIA](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/)

**Used / prior-gen EPYC (eBay, undated listings seen Oct 2026)**
- EPYC 9554 (64c Genoa): ~$1,750 (Dell vendor-locked) to ~$2,450; unlocked used ~$1,925; new unlocked from Shenzhen $2,119 — [eBay](https://ebay.com/p/20058044883), [eBay.de](https://www.ebay.de/itm/115850434209); EPYC 9654 (96c) new sealed $2,099 (CA seller); 9654 + 384 GB DDR5 bundle sold $4,700 — [eBay.de](https://www.ebay.de/itm/365559919448)
- Dual-Genoa home build estimate (2 × 9334 QS, 24 × 64 GB DDR5-5600, 4 × 2 TB NVMe) ≈ €5,000 in early 2025, "with current DDR5 prices ... expected to exceed 10k" — [Level1Techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826)

**Workstation**
- Threadripper 9980X (64c, 350 W): launch $4,999, $4,819 on Aug 14 2026, 12-month low $3,999 (Jul 9 2026) — [TechSpot](https://www.techspot.com/products/cpu/amd-ryzen-threadripper-9980x-32ghz-socket-str5.309476/); Threadripper PRO 9995WX (96c, 350 W): $11,699 at Amazon/Newegg — [TechRadar](https://www.techradar.com/pro/amd-threadripper-pro-9995wx-goes-on-sale-at-amazon-and-newegg-for-as-little-as-usd11699-you-can-get-the-worlds-fastest-cpu-but-i-dont-think-its-worth-it); all TR 9000 parts 350 W TDP, boost to 5.4 GHz — [Notebookcheck](https://www.notebookcheck.net/AMD-launches-Ryzen-Threadripper-9000-Shimada-Peak-series-led-by-the-96C-192T-350-W-Ryzen-Threadripper-Pro-9995WX.1021230.0.html)

**Apple**
- Mac Studio M3 Ultra (2025): 32-core CPU/80-core GPU $5,499 + $4,000 for 512 GB → $9,499 **[my sum]** — [Low End Mac](https://lowendmac.com/2025/mac-studio-early-2025/)
- Mac Studio M5 Ultra (announced Aug 25 2026, ships Sep 22; 512 GB config "late October"): from $5,499 (96 GB, 1 TB); 256 GB/16 TB $18,299; 512 GB price not announced at launch; M5 Max Mac Studio from $2,499 (460 GB/s; 36–128 GB), street $2,299 — [Macworld](https://macworld.com/article/3220024/apple-announces-the-m5-ultra-mac-studio-with-up-to-512gb-of-ram.html); a 256 GB/4 TB config quoted at $12,299 — [Guru3D](https://guru3d.com/story/apple-m5-ultra-mac-studio-launches-with-36core-cpu-80core-gpu-and-512gb-unified-memory/)

**Strix Halo mini-PCs (128 GB LPDDR5X)**
- Framework Desktop Ryzen AI Max+ 395 128 GB from $1,999 (launch, 2025) — [TechPowerUp forum](https://www.techpowerup.com/forums/posts/5488111); Framework's current page lists configs (incl. a new Max+ PRO 495 192 GB) without text prices — [Framework](https://frame.work/desktop)
- GMKtec EVO-X2 128 GB/2 TB: €1,799 final pre-sale price (May 2025), CNY 14,999 (~$2,066) in China — [Yahoo Tech](https://tech.yahoo.com/computing/articles/gmktec-evo-x2-workstation-pc-030300493.html); Micro Center sale $1,709 (128 GB) — [Level1Techs](https://forum.level1techs.com/t/gmktec-evo-x2-vs-framework-desktop/233420); EU retail €3,749–3,799, CHF 3,299 — [Galaxus](https://www.galaxus.ch/en/s1/producttype/pc-18/compare/gmktec-evo-x2-vs-hp-z2-mini-g1a-2501)
- HP Z2 Mini G1a 128 GB/1 TB: CHF 5,555 (no US price found) — [Galaxus](https://www.galaxus.ch/en/s1/producttype/pc-18/compare/gmktec-evo-x2-vs-hp-z2-mini-g1a-2501)

**DDR5 server memory (the 2026 shock)**
- 64 GB DDR5 RDIMM contract price: ~$255 (Q3 2025) → $450 (Q4 2025) → >$900 (Q1 2026) — [Network World citing Counterpoint](https://www.networkworld.com/article/4093752/server-memory-prices-could-double-by-2026-as-ai-demand-strains-supply.html); Counterpoint: $927 end-Q1 2026, expected ~$1,500 in Q3 2026 — [Counterpoint](https://counterpointresearch.com/de/reports/memory-price-tracker-april-2026); TrendForce expects +13–18 % QoQ in 3Q26 — [same](https://www.networkworld.com/article/4093752/server-memory-prices-could-double-by-2026-as-ai-demand-strains-supply.html)
- Retail (Aug 14 2026): new DDR5 ECC RDIMM $29.56–39.06/GB, median $35.70/GB; a 64 GB RDIMM ≈ $2,064 — [Capital & Compute](https://capitalandcompute.net/memory-prices/server/); live index (Oct 2026): DDR5 ECC median $44.33/GB (16 modules), best $29.37/GB; DDR4 ECC median $8.93/GB, 64 GB DDR4 from $6.23/GB — [DatacenterDisk](https://datacenterdisk.com/server-ram/price-index)
- Cloud alternative: AWS r8g.48xlarge (Graviton4, 192 vCPU, 1,536 GiB) $11.31/h on-demand, $3.88/h spot, $5.13/h 3-yr reserved — [Vantage](https://instances.vantage.sh/aws/ec2/r8g.48xlarge)

### Inferences
- **Illustrative single-socket bill of materials, Oct 2026 [my arithmetic]**: EPYC 9575F (~$7.7k street) + 12 × 64 GB DDR5-6400 RDIMM at $927–2,064 each = $11.1k–24.8k memory → **$19–33k** for ~570 GB/s and 768 GB, before board (~unverified, see Gaps), chassis, PSU. Memory is 60–75 % of the cost. The same socket in Q3 2025 would have cost ~$3k in memory.
- Used Genoa route: 9654 (~$2.1k) + 12 × 64 GB DDR5-4800 ($11–25k) → ~$13–27k for ~360 GB/s measured; the CPU is now almost irrelevant to cost.
- Dual 6980P with 1.5 TB MRDIMM-8800: CPUs ~$28–36k; MRDIMM prices not found but exceed RDIMM; realistic $60–100k+ per node is plausible but not sourced.
- **$ per GB/s of memory bandwidth (whole system) [my arithmetic]**: Strix Halo 128 GB mini-PC $1,709–1,999 → $6.7–7.8 per GB/s; Mac Studio M5 Ultra 96 GB $5,499 → $4.6 per GB/s (512 GB price unknown); M3 Ultra 512 GB $9,499 → $11.6 per GB/s; single Turin socket $19–33k → $33–58 per GB/s; AWS r8g.48xlarge at $11.31/h ≈ $99k/yr on-demand for ~1,075 GB/s theoretical (2 sockets).

### Gaps
- No verified prices for SP5/LGA-4710 motherboards, MRDIMM-8800 modules, or 96 GB RDIMMs (search quota exhausted); no current used-market prices for Sapphire/Emerald Rapids.
- No US list price for HP Z2 Mini G1a; Mac Studio M5 Ultra 512 GB price not yet published at launch.
- Street prices for 6980P/9965 are inferred from a benchmark site's value ratio, not retailer quotes.

---

## KQ5. Power draw per socket under AI load (perf-per-watt)

### Takeaway
Flagship sockets are 500 W TDP (6980P, 9965, 9755) and a loaded dual-socket Turin server draws 1.5–1.7 kW at the wall; Arm parts (AmpereOne 276 W, Grace Superchip 500 W incl. memory) and Apple (M3 Ultra ~160–180 W while running R1 at 17 tok/s) are far lower, and Strix Halo runs LLMs under ~100–140 W. Measured efficiency: Xeon 6972P at 128 concurrent users ≈ 8 M output tokens/kWh for Llama-3-8B BF16; Apple M3 Ultra ≈ 0.1 tok/s per watt on a 671B MoE at batch 1.

### Cited Findings
- TDPs: Xeon 6980P 500 W, 6960P 500 W, 6952P 400 W, 6732P 350 W — [Wikipedia: Granite Rapids](https://en.wikipedia.org/wiki/Granite_Rapids); 6780E 330 W, 6766E 250 W — [Wikipedia: Sierra Forest](https://en.wikipedia.org/wiki/Sierra_Forest); EPYC 9965/9755 500 W, 9575F/9655 400 W, 9555 360 W, 9355 280 W, 9175F 320 W — [Wikipedia: Epyc](https://en.wikipedia.org/wiki/Epyc); 8592+ 350 W — [Wikipedia: Emerald Rapids](https://en.wikipedia.org/wiki/Emerald_Rapids); AmpereOne A192-32X 274–276 W — [Phoronix](https://www.phoronix.com/review/ampereone-a192-32x/12); Grace Superchip "only 500 W for the CPU and memory", C1 140–250 W configurable — [NVIDIA](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/); Threadripper 9000 350 W — [Notebookcheck](https://www.notebookcheck.net/AMD-launches-Ryzen-Threadripper-9000-Shimada-Peak-series-led-by-the-96C-192T-350-W-Ryzen-Threadripper-Pro-9995WX.1021230.0.html); Strix Halo 45–120 W configurable — [Wikipedia: Zen 5](https://en.wikipedia.org/wiki/Zen_5); GMKtec EVO-X2 140 W total budget — [Yahoo Tech](https://tech.yahoo.com/computing/articles/gmktec-evo-x2-workstation-pc-030300493.html); Mac Studio (M5 generation) maximum continuous power 480 W — [Apple](https://www.apple.com/mac-studio/specs/)
- **[measured]** Dual-socket 500 W Turin server with 24 DIMMs, NICs, SSDs: ~1.5–1.7 kW; 1.2 kW PSUs are a challenge on 120 V — [ServeTheHome](https://servethehome.com/amd-epyc-9005-turin-turns-transcendent-performance-solidigm-broadcom/5)
- **[measured]** AmpereOne A192-32X server 577 W under SPECint load; comparison servers 8× L40S 3,097 W, 8× H100 6,297 W — [Ampere](https://amperecomputing.com/blogs/llms-bigger-is-not-always-better); Phoronix found EPYC 9965 has better performance and performance-per-watt than the 192-core AmpereOne — [Phoronix](https://phoronix.com/review/amd-epyc-9965-ampereone)
- **[measured]** Xeon 6 vLLM Llama-3-8B BF16: 6972P ≈ 7 M tokens/kWh (64 users), 8 M (128 users); 6960P 5.7 M / 6.5 M; 6732P 3.2 M / 4 M; "close to double the power efficiency" for 96c vs 32c "despite drawing more peak power per socket" — [Lenovo LP2463](https://lenovopress.lenovo.com/lp2463.pdf)
- **[measured]** M3 Ultra Mac Studio: DeepSeek R1 4-bit at 17–18 tok/s "drawing around 160 to 180 W" — [AppleInsider](https://appleinsider.com/articles/25/03/18/heavily-upgraded-m3-ultra-mac-studio-is-great-for-ai-projects); Strix Halo "under 100 W" during 35B-A3B runs — [Level1Techs](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466)
- **[vendor claim]** Intel: Xeon 6 up to 1.9× perf/W vs 8592+ at 40 % utilisation; 2× 6972P 1.86× perf/W vs 2× EPYC 9654 on Llama-2-7B INT8; 6760P 1.73× perf/W vs EPYC 9535 on ViT INT8 — [Intel performance index](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/intel-xeon-6)

### Inferences
- 8 M tokens/kWh (Lenovo, 6972P at 128 users) = **2.2 output tok/s per watt** at the system level for an 8B BF16 model — this is the best batched CPU efficiency figure found. M3 Ultra at 17.5 tok/s / 170 W ≈ 0.10 tok/s/W on a 671B MoE at batch 1; Strix Halo 70B Q4 at 5.1 tok/s / ~100 W ≈ 0.05 tok/s/W.
- A dual-flagship x86 node realistically budgets 1.5–1.7 kW; four such nodes "ganged" would be ~6–7 kW, comparable to one 8× H100 server's 6.3 kW (Ampere's figure).

### Gaps
- No measured socket-level power during AMX/AVX-512 LLM load for 6980P or 9965 (Phoronix power pages inaccessible).
- No power data for Graviton4 (AWS does not publish) or Grace CPU-only LLM runs.

---

## KQ6. Benchmarks comparing CPU-only inference with GPUs

### Takeaway
Independent, audited CPU-vs-GPU comparisons for LLMs are scarce: Intel is the only MLPerf CPU-only submitter and did not submit Llama-70B, and its public claims compare only against AMD CPUs or older Xeons. The vendor-neutral data available (Lenovo, llama.cpp community, Ampere's own SLA tables) consistently show CPUs serving ≤ 8B models to ~8–30 users or large MoE models at 10–20 tok/s single-stream, and failing ≥ 10 tok/s SLAs for dense 70B.

### Cited Findings
- Intel is "the only server processor vendor to submit stand-alone CPU results" in MLPerf Inference v6.0 (Apr 2026); over half of v6.0 submissions had Xeon hosts; Intel's GPU comparisons concern its own Arc Pro B-Series — [Intel newsroom](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-delivers-ai-performance-mlperf-inference-v6-0.html); STH on v5.0: Intel's Xeon 6900P/6700P submissions were the only CPU-only results — [ServeTheHome](https://www.servethehome.com/mlperf-inference-v5-0-released-amd-intel-nvidia/)
- MLPerf v5.0 CPU-only absolute results (2× 6980P): ResNet-50 45,530 samples/s Offline; GPT-J 6B ≈ 517 tok/s Offline — see KQ3 links. Signal65's v5.0 review lists "Intel Xeon 6980P CPU" among new hardware but gives GPU results only relatively (B200 ≈ 3× H200 on Llama-3.1-70B/405B) — [Signal65](https://signal65.com/research/ai/mlperf-inference-v5-0-new-workloads-new-hardware/)
- Lenovo (Jul 2026): "a substantial share of enterprise AI workloads ... utilize ... 7B–8B parameter range ... single-digit to low double-digit simultaneous users per node"; Xeon 6 can serve this tier "without a discrete accelerator"; at 128 users per-user throughput falls to single digits — [Lenovo LP2463](https://lenovopress.lenovo.com/lp2463.pdf)
- Ampere's rack-level framing: AmpereOne A192-32X server (577 W) vs 8× L40S (3,097 W) and 8× H100 (6,297 W) servers for sessions-per-12.5 kW-rack at ≥ 10 tok/s; H100 MIG assumed 7 sessions/GPU for small models; CPU supports 0 sessions of 70B/405B at the SLA **[vendor]** — [Ampere](https://amperecomputing.com/blogs/llms-bigger-is-not-always-better); Ampere claims a 2× price-performance advantage over an NVIDIA A10 bare-metal instance for Llama 3.1 8B on a 90-core A4 (vendor claim, seen in search snippet; exact page not confirmed) — [Ampere brief](https://amperecomputing.com/briefs/llama-3-ai-inference-on-ampereone)
- Lambda: GH200 (Grace + H100, CPU-offload) 4.33 tok/s vs 0.57 tok/s on H100 SXM with CPU offload for Llama-3.1-70B FP16 (vendor benchmark; illustrates CPU memory as spill capacity, not CPU compute) — [Lambda](https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics)
- Intel TCO claims (vendor assumptions: 4-yr, PUE 1.6, thinkmate pricing): 10 racks of 2× 6972P replace 18 racks of 2× EPYC 9654 for Llama-2-7B chatbot (57 % lower TCO) — [Intel performance index](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/intel-xeon-6)
- SGLang CPU backend authors position dual-6980P DeepSeek-R1 at "fractional cost" vs GPU deployments but give no dollar or GPU figures — [LMSYS blog](https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization)
- Cloud H100 rental fell to ~$1.49–3.90/h (2025) — [Introl](https://introl.com/blog/cost-per-token-llm-inference-optimization) (context for cost-per-token comparisons; GPU throughput numbers are covered by the GPU researcher)

### Inferences
- Comparing like-for-like at batch 1 on DeepSeek-R1-671B: dual 6980P + 1.5 TB MRDIMM (SGLang) ≈ 14.7 tok/s decode / 543 tok/s prefill; M3 Ultra 512 GB ≈ 17–19 tok/s decode (short context) / ~77–189 tok/s prefill; the Mac costs ~$9.5k vs an unpriced but almost certainly > $50k Xeon node — for single-user large-MoE inference the Apple box wins on $/tok and W/tok, while the Xeon node wins on prefill and multi-user capacity.
- "Ganging" CPUs to reach GPU-class throughput runs into three documented walls: (1) decode is bandwidth-bound and llama.cpp dual-socket scaling is ~1.0–1.8×; (2) each additional socket now carries $11–25k of DRAM at 2026 prices; (3) prefill-class compute (AMX ~0.5–0.8 POPS INT8 peak per 6980P) is only exploitable with AMX-aware runtimes (SGLang CPU, ktransformers, IPEX), none of which run on AMD/Arm.
- A tensor-parallel CPU runtime over several AMX sockets (SGLang's 3 % NUMA overhead suggests feasibility) would be the only path to GPU-like batched throughput; no public multi-node CPU LLM-serving benchmark was found.

### Gaps
- No independent head-to-head of a Xeon 6/EPYC 9005 node vs a single H100/H200/B200 on identical LLM workloads was found; MLPerf CPU and GPU entries differ in benchmarks (no CPU Llama-70B).
- SemiAnalysis / Chips and Cheese CPU-vs-GPU inference analyses were not retrieved (search quota).
- No cost-per-million-tokens figure for CPU serving from a neutral source; the only $/token-style data are Lenovo's tokens/kWh and AWS instance prices.
