# Prototype sketch: Intel Xeon hosts synced by an FPGA fabric, versus a GPU

*Design document, 27 October 2026. Every price and measurement below is dated and sourced in
`research_notes/FPGA synced CPUs as GPU alternative/`; figures the notes could not source are
marked **est.** The simulation numbers come from `tools/sim_xeon_fpga_cluster.py` and the full
tables are in `docs/sim_results.md`.*

## 0. The short version

The idea under test: an FPGA that synchronises several CPUs so that together they behave like a
GPU for neural-network inference, at lower cost than a GPU.

What the sketch and the simulation show:

| Question | Answer |
|---|---|
| What does an FPGA sync fabric actually fix? | The latency of the 160 all-reduces per generated token that tensor parallelism needs. On four Xeon 6980P hosts that is about **2.0 ms per token over 400G RDMA and about 1.0 ms with an FPGA tree reduction**. |
| What sets the token time? | Streaming the model weights out of DRAM. Four hosts each read 10 GB per token for a 70B 4-bit model, which at the measured 32-80 % of 614 GB/s takes **about 32 ms**. The FPGA cannot touch this. |
| Net effect of the FPGA | **+3 % tokens/s** over plain RDMA NICs, for +$7,000 per node and a multi-year RTL project. |
| Where the money goes | DDR5 RDIMMs are $29-44/GB in October 2026, so one 12-channel socket costs **$40-60 per GB/s** of bandwidth. A used RTX 3090 costs about **$2 per GB/s**. The design loses on the one metric that matters for decoding. |
| Best case for the CPU cluster | 70B 4-bit, batch 1: **21-41 tok/s at $136k** (Rev 2) versus 29-36 tok/s at $7k for two used RTX 4090s, 55-68 tok/s at $30k for one H100, 130-160 tok/s at $46k for one B200. |
| Cost per million generated tokens (70B, batch 16) | Xeon cluster **$6-12**; H100 $0.8-1.0; B200 $0.5-0.6; Mac Studio M5 Ultra $0.6-1.0; public API $0.32 (DeepInfra output price). |
| Is any FPGA path viable? | Only **Rev 3**, where an HBM FPGA does the math itself (31-59 tok/s at $88k in simulation). That is what Positron sells, and they are moving from the Agilex FPGA to an ASIC because the FPGA is the expensive part. |

The rest of this document is the build sheet behind those numbers, the design revisions
("reworks"), the simulation evidence, and a go/no-go test plan that costs almost nothing.

## 1. What a decode step is, in numbers

For one new token of a dense model every weight is read once. Per token:

| Model | Active weights read | FLOPs | KV cache read at 2k context |
|---|---|---|---|
| Llama-3.3-70B, 4-bit | 40 GB | 141 GFLOP | 0.7 GB |
| Llama-3.3-70B, INT8 (AMX-native) | 70 GB | 141 GFLOP | 0.7 GB |
| Llama-3.1-8B, INT8 | 8 GB | 16 GFLOP | 0.25 GB |
| DeepSeek-V3/R1, 4-bit (37B active of 671B) | 21 GB | 74 GFLOP | 0.07 GB |

So at batch 1 the token rate is bandwidth divided by bytes, and nothing else, until communication
or software overhead gets in the way:

```
tokens/s  <=  usable aggregate DRAM bandwidth / bytes per token
```

A Xeon 6980P socket with twelve DDR5-6400 channels has 614 GB/s theoretical. Measured end-to-end
decode on dual 6980P with SGLang's AMX backend reached 915 GB/s of a 1.69 TB/s MRDIMM pair (54 %)
on a dense 70B INT8 model and 544 GB/s (32 %) on DeepSeek-R1. That is the real-world range used in
the simulator. The CPU side also needs 2 x 80 = 160 all-reduces of a 16 KiB vector per token when
the model is split across hosts by tensor parallelism.

## 2. Build sheet, by revision

All prices USD, October 2026, from the research notes unless marked est. "Host share" is board,
chassis, PSU, boot SSD and a 25G NIC for a server node (est. $3,500).

### Rev 0: "two CPUs synced" is one dual-socket board

The first thing to notice is that Intel already sells the sync fabric for two CPUs. UPI 2.0 gives
48 GB/s per direction per link, with three to six links per socket, and about 200 ns remote-memory
latency. SGLang measured the all-reduce overhead between two 6980P sockets at 3 % of the token
time. No FPGA can beat an on-package coherent link.

| Qty | Part | Unit | Notes |
|---|---|---|---|
| 2 | Intel Xeon 6980P, 128 cores, AMX, 12-channel, 500 W | $14,000 street | list $17,800; 6972P (96c) $14,600 list and 6960P (72c) $13,750 list are near-equivalent for bandwidth |
| 24 | 16 GB DDR5-6400 ECC RDIMM (fills all 24 channels for full bandwidth, 384 GB total) | $560 | at $35/GB median; the cheapest way to get the whole 1.23 TB/s |
| 24 | *or* 32 GB MRDIMM-8800 (768 GB, 1.69 TB/s theoretical) | $1,440 est. | no public MRDIMM price; est. $45/GB; 2nd-gen MRDIMM slated Q1 2027 |
| 1 | Dual-socket Xeon 6900P 2U server (Supermicro X14 / Lenovo SR650 V4 class) | $5,000 est. | board price not found in the notes |
| | **Total, DDR5 build** | **~$47,000** | |
| | **Total, MRDIMM build** | **~$68,000** | |

Software: SGLang CPU backend (AMX INT8, measured), vLLM CPU backend, Intel xFasterTransformer
(oneCCL tensor parallel), llama.cpp as the control.

Measured anchors on exactly this hardware (LMSYS/Intel, July 2025, dual 6980P MRDIMM):
Llama-70B INT8 **13.1 tok/s** decode, DeepSeek-R1 INT8 **14.7 tok/s** decode, **543 tok/s**
prefill. Simulator band for the same box: 8-17 tok/s and 16-32 tok/s respectively.

### Rev 1: four hosts, off-the-shelf 400G RDMA fabric

Before building an FPGA, build the thing it would replace. Intel's own precedent: Qwen-72B on four
Xeon 8575C sockets at 140 ms/token (7 tok/s), and Llama-2-70B going from 249.7 ms on two sockets
to 87.7 ms on eight sockets across four machines (2.85x on 4x the hardware).

| Qty | Part | Unit | Notes |
|---|---|---|---|
| 4 | Single-socket Xeon 6980P host, 12 x 16 GB DDR5-6400 (192 GB), host share | $24,220 | |
| 4 | 400G RoCE NIC, ConnectX-7 class | $2,500 | adapter latency 0.5-1.5 us |
| 6 | 400G direct-attach cables (full mesh, no switch) | $300 est. | a switch adds 0.6-1.5 us per hop and ~$8-10k |
| 1 | Rack, PDU, 100 GbE management | $9,000 est. | |
| | **Total** | **~$116,000** | 2.9 kW |

Software: oneCCL ring all-reduce under xFasterTransformer or SGLang with TP=4.

Simulation, 70B 4-bit, batch 1: **20-39 tok/s** (p50 30), all-reduce share **6 %** of the step.

### Rev 2: the proposal. FPGA hardware all-reduce fabric between the hosts

Replace each NIC with an FPGA card that does the reduction in hardware over direct-attach optical
links, so the 160 collectives per token become 2 fixed hops (host to FPGA tree, tree back to host)
instead of a 2(N-1)-step ring.

| Qty | Part | Unit | Notes |
|---|---|---|---|
| 4 | Xeon 6980P host as in Rev 1 | $24,220 | |
| 4 | AMD Alveo V80: PCIe Gen5, 4 x QSFP56 200G ports, 32 GB HBM2e, 190 W | $9,495 list | AMD's current recommended Alveo; 4 ports allow a 4-node full mesh with no switch |
| 4 | *Intel-native alternative:* Altera Agilex 7 I-series PCIe card (BittWare IA-840f or Altera dev kit), 4 x QSFP-DD | $10,000 est. | no public price; Agilex 7 I/M-series R-tile has hard PCIe 5.0 x16 and CXL Type 1/2/3 IP |
| 6 | QSFP56 direct-attach cables | $400 est. | |
| | **Total hardware** | **~$136,000** | 3.2 kW |
| | **Non-recurring engineering** | **$0.5-1.5 M est.** | see below |

FPGA design (what has to be written):

1. Host DMA engine (XDMA/QDMA on AMD, or Altera's PCIe hard IP) with a doorbell and completion
   path. Measured floor for a PCIe DMA round trip is about 1 us; this is 2 of the 4 fixed
   latencies per collective and nothing in the FPGA can remove it.
2. A bf16/fp32 adder tree that reduces four 16 KiB vectors as they stream in (trivial logic).
3. A lightweight link layer over the QSFP ports (Microsoft Catapult's LTL is the precedent:
   2.88 us FPGA-to-FPGA round trip through a top-of-rack switch; direct links should be about
   1 us one-way).
4. A oneCCL or custom collective plug-in on the CPU side that posts the vector to the card and
   spins on the completion flag.

Engineering estimate: two FPGA engineers and one systems engineer for 9-15 months **(est.; the
notes found no month-level effort data; the closest academic precedent, ACCL+ at OSDI 2024, was a
multi-year research project and found its host-data path "comparable to software RDMA MPI").**

Simulation, 70B 4-bit, batch 1: **21-41 tok/s** (p50 30), all-reduce share **3 %**.
Per-token breakdown at the median: weight streaming 31.8 ms, all-reduces 1.0 ms (versus 2.0 ms in
Rev 1). The FPGA buys 0.9 ms per token, a 3 % gain, for about $20,000 of extra hardware per four
nodes plus the NRE.

**Rework 2b, CXL-attached FPGA.** Attach the FPGA as a CXL Type-2 device so the CPU writes the
vector into FPGA-mapped memory with plain stores and no DMA doorbell. Measured uncached
host-to-FPGA access on Agilex 7 is 0.4-0.7 us (versus 0.14-0.41 us for ASIC CXL devices and about
0.1 us for local DRAM). Simulation: 21-42 tok/s, all-reduce share 2 %. Gain over 2: under 1 %.

**Rework 2c, micro-batch overlap.** Split each batch into four micro-batches so the all-reduce of
one overlaps the compute of the next. Simulation: +3 % at batch 1, more at larger batches. This is
a software change and works equally well with RDMA NICs.

**Rework 2d, eight hosts.** Doubling to eight 6980P hosts doubles aggregate bandwidth: 39-74 tok/s
at $273k. The cost per token does not improve, because the DRAM cost scales with the sockets.

### Rev 3: the FPGA does the math (HBM FPGA as the engine)

The only FPGA architecture with a physics case: an FPGA with its own HBM holds a shard of the
weights and streams them at 820 GB/s, which is 1.3x a Xeon socket's theoretical bandwidth at a
third of the memory cost.

| Qty | Part | Unit | Notes |
|---|---|---|---|
| 4 | Altera Agilex 7 M-series HBM2e dev kit (DK-DEV-AGM039EA), 32 GB HBM2e, 820 GB/s, 88.5 INT8 TOPS theoretical | EUR 9,889 (~$10,700) | 33-week factory lead time; BittWare IA-860m is the production card (PCIe Gen5, CXL, 400G) with no public price |
| 4 | *or* AMD Alveo V80, 32 GB HBM2e, 819 GB/s, no AI engines | $9,495 | DSP-only INT8, est. ~20 TOPS |
| 4 | Host PCs | $1,500 | |
| 6 | QSFP direct-attach cables | $400 est. | |
| | **Total hardware** | **~$88,000** (Agilex) | 1.8 kW |
| | **Non-recurring engineering** | **$2-5 M est.** | a full transformer engine in RTL/HLS: FlightLLM, DFX and EdgeLLM are each multi-year academic efforts; Positron raised $875 M to do it commercially |

Simulation, 70B 4-bit sharded over four cards, batch 1: **31-59 tok/s** at $88k. The wide band is
the memory-bandwidth utilisation: published FPGA LLM engines (FlightLLM on U280/VHK158) reach about
35 %, Positron claims 93 % on the same Agilex-M silicon, and no independent measurement exists.
Even at the top of the band this is a used-GPU-pair result at twelve times the price.

## 3. Simulation evidence

Full tables for four models are in `docs/sim_results.md`. The headline rows, 70B 4-bit, 2,048-token
context, 3-year amortisation at 50 % utilisation:

| Configuration | Capex | tok/s batch 1 (p10-p90) | $/Mtok batch 1 | tok/s batch 16 | $/Mtok batch 16 |
|---|---|---|---|---|---|
| 2 x used RTX 4090, PCIe | $7,000 | 29-36 | $6.9-8.6 | no fit (KV) | |
| 1 x RTX PRO 6000 96 GB | $19,000 | 26-34 | $13.7-17.6 | 340-435 | $1.07-1.37 |
| 1 x H100 SXM | $30,000 | 55-68 | $10.4-12.9 | 706-867 | $0.81-1.00 |
| 1 x B200 | $46,000 | 130-160 | $6.7-8.2 | 1678-2060 | $0.52-0.64 |
| 1 x Intel Gaudi 3 | $20,600 | 56-70 | $7.5-9.4 | 721-898 | $0.58-0.73 |
| Mac Studio M5 Ultra 96 GB | $5,499 | 14-20 | $6.7-9.5 | 135-218 | $0.61-0.98 |
| Rev 0: 2 x Xeon 6980P, DDR5 | $48,000 | 11-21 | $53-106 | 136-272 | $4.2-8.3 |
| Rev 0: 2 x Xeon 6980P, MRDIMM | $70,000 | 14-29 | $55-109 | 185-361 | $4.4-8.5 |
| Rev 1: 4 x Xeon 6980P, 400G RDMA | $116,000 | 20-39 | $69-133 | 257-493 | $5.5-10.5 |
| Rev 2: 4 x Xeon 6980P, FPGA fabric | $136,000 | 21-41 | $78-153 | 263-512 | $6.2-12.0 |
| Rev 2: 8 x Xeon 6980P, FPGA fabric | $273,000 | 39-74 | $85-162 | 497-920 | $6.9-12.7 |
| Rev 3: 4 x Agilex 7 M HBM FPGAs | $88,000 | 31-59 | $34-66 | 383-724 | $2.8-5.3 |
| Anti-pattern: 8 x Xeon 6732P over 10 GbE | $93,000 | 14-21 | $108-161 | 113-138 | $16-20 |

For an 8B INT8 model the ratios are the same: one used RTX 3090 ($2,150) gives 65-85 tok/s at
$1.0-1.3 per million tokens; the dual-6980P box gives 51-100 tok/s at $11-22; the four-host FPGA
cluster gives 97-185 tok/s at $17-32.

How much to trust it:

- The simulator reproduces seven measured points (dual-6980P SGLang 70B INT8 and R1, RTX 5090 and
  3090 llama.cpp, M3 Ultra R1, Strix Halo 70B, DGX Spark 70B) inside or at the edge of its P10-P90
  band; run `python3 tools/sim_xeon_fpga_cluster.py --validate` to see them.
- It is deliberately optimistic for the CPU side: perfect load balance, no NUMA mistakes, no
  software stalls beyond a fixed 8-25 us per layer. Real multi-host CPU runs scale worse (Intel's
  own 8-socket run reached 71 % efficiency; llama.cpp RPC runs get slower with more nodes).
- The communication model is where the FPGA could surprise: if a card achieved a 0.3 us host hop
  (it cannot over PCIe; CXL measured 0.4-0.7 us) the all-reduce time would fall from 1.0 ms to
  about 0.4 ms per token. That is still 1-2 % of the step.

## 4. "Reworks": what would have to change for the idea to work

1. **Cheaper DRAM bandwidth.** The cluster's cost is dominated by filling 12 channels per socket at
   $35/GB. If RDIMM prices returned to the Q3-2025 level of about $4/GB, the Rev 0 box falls to
   about $35k and Rev 2 to about $110k, still 5-15x the GPU cost per token. The fabric does not
   enter this calculation at all.
2. **Higher memory-bandwidth utilisation in software.** Xeon decode sits at 32-54 % of theoretical;
   GPU serving stacks reach 70-85 %. Closing that gap is worth up to 1.5-2x and needs no hardware.
   This is the one lever worth pulling, and it is Intel's and the SGLang team's problem, not an
   FPGA's.
3. **AMX prefill efficiency.** Measured prefill on 2 x 6980P (388-543 tok/s on 70B-671B INT8)
   implies about 27 TFLOPS per socket, roughly 5 % of the 524 TOPS INT8 peak at base clock.
   Better kernels could give 3-6x prefill. Again software.
4. **Batching.** Batch 16 cuts the CPU cluster's cost per token by about 10x, but it does the same
   for GPUs; the ratio holds.
5. **A fabric that adds bandwidth, not just lower latency.** This is the CXL memory-pool direction
   (Enfabrica, which NVIDIA paid more than $900 M for), and its customers attach GPUs to the pool,
   not CPUs to each other.

## 5. Go/no-go test plan (cheapest path to a real measurement)

| Gate | Cost | What to run | Pass criterion | Simulator prediction |
|---|---|---|---|---|
| 0. Paper | $0 | `python3 tools/sim_xeon_fpga_cluster.py` with your own prices and a `--custom` scenario in `tools/inference_cost_model.py` | a CPU configuration under $10 per million tokens at batch 16 | none exists at 2026 DRAM prices |
| 1. One box, rented | ~$500-1,500 for a week | a dual-socket Xeon 6900P bare-metal server (any bare-metal provider; not priced in the notes), SGLang CPU backend, Llama-70B INT8 and 8B INT8, batch 1 and 16; record tok/s, prefill tok/s, and `perf` memory-bandwidth counters | end-to-end memory-bandwidth utilisation above 60 % | 32-54 % (measured 2025) |
| 2. Two hosts, RDMA | ~$5k-$50k (rent or buy) | two single-socket hosts with ConnectX-7 direct-attached, xFasterTransformer TP=2; measure all-reduce latency for 16 KiB with `oneCCL` benchmarks and the end-to-end token time | communication above 20 % of the token time (otherwise an FPGA has nothing to fix) | 6 % at 4 hosts |
| 3. FPGA latency only | ~$20k | two Alveo V80 (or Agilex 7 I-series) cards, a minimal DMA-reduce-DMA loop, no model | host-to-host all-reduce of 16 KiB under 2 us | 2-4 us (PCIe DMA floor about 1 us each way) |
| 4. Full Rev 2 | ~$136k + NRE | only if gates 2 and 3 both pass | beats Rev 1 by more than 25 % tok/s at batch 1 | +3 % |

The decision rule is that Rev 2 is funded only if gate 2 shows communication is a large share of
the token time and gate 3 shows the FPGA can cut it. The simulator says gate 2 fails by a factor of
three and gate 3 is marginal.

## 6. Intel-based builds that do make sense for cheap inference

- **Xeon 6 + one GPU (ktransformers pattern).** The CPU holds the MoE experts in DRAM (capacity),
  the GPU holds attention and the dense layers (bandwidth): DeepSeek-R1 at 12-14 tok/s decode and
  up to 287 tok/s prefill on dual Xeon 6454S plus one RTX 4090D (Feb 2025). The CPU contributes
  capacity, the accelerator contributes bandwidth. This is the proven CPU-plus-accelerator split.
- **Intel Gaudi 3.** 128 GB HBM2e, 3.7 TB/s, $15,625 list (2024; no 2026 price found):
  simulated $0.58-0.73 per million tokens on 70B at batch 16. This is Intel's own answer to the
  question and it is an accelerator with HBM, not a CPU cluster.
- **Intel Arc B580** for 8B-class models: 456 GB/s, 12 GB, $580-655 street; 31-40 tok/s on 8B
  INT8 at $1.4-1.7 per million tokens.
- **A single dual-socket Xeon 6 box (Rev 0)** is still the right tool when the model does not fit
  any affordable accelerator and privacy or data residency rules out an API: DeepSeek-R1 INT8 at
  14.7 tok/s with 768 GB of memory is a real capability, just not a cheap one.

## 7. Numbers the notes could not source (verify before buying)

- MRDIMM-8800 module prices (est. $45/GB here) and dual-socket Xeon 6900P board prices.
- Alveo V80 port configuration and PCIe lane split (4 x QSFP56 is from AMD's product page as
  recalled; confirm on the datasheet). Altera Agilex 7 I-series card prices (none public).
- Sustained AMX clock under all-core load (Intel does not publish it).
- Engineering effort for an FPGA collective engine or transformer engine (no month-level data).
- Any independent Positron benchmark (all figures are vendor claims).
- 2026 Gaudi 3 pricing; 2026 used-market prices for Sapphire/Emerald Rapids.
