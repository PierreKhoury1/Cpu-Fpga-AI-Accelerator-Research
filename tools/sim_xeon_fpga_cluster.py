#!/usr/bin/env python3
"""
sim_xeon_fpga_cluster.py - Monte-Carlo simulation of LLM decode/prefill on a cluster of
Intel Xeon hosts whose tensor-parallel all-reduces run over a fabric (UPI, RDMA NIC,
or an FPGA hardware all-reduce card), compared with GPU baselines.

Why a separate simulator from inference_cost_model.py?
  * It walks the transformer layer by layer (QKV, attention over KV, output proj,
    all-reduce, MLP up/gate, MLP down, all-reduce) instead of one lumped roofline.
  * It models the fabric path in detail: host DMA to the card, serialisation,
    reduction hops, DMA back, plus per-collective software overhead.
  * Every uncertain parameter is a RANGE and the simulator samples it, so the
    output is a P10 / P50 / P90 band, not a single optimistic number.
  * It supports overlap of communication with compute via micro-batching
    (the only real trick available to hide all-reduce latency in decode).

Nothing here needs more than the Python standard library.

Usage:
    python3 tools/sim_xeon_fpga_cluster.py                      # all scenarios, Llama-70B Q4
    python3 tools/sim_xeon_fpga_cluster.py --model llama8b-int8 --samples 2000
    python3 tools/sim_xeon_fpga_cluster.py --markdown > sim.md  # table for the report
    python3 tools/sim_xeon_fpga_cluster.py --json
    python3 tools/sim_xeon_fpga_cluster.py --explain "Rev 2: 4x Xeon 6980P, FPGA all-reduce fabric"

The preset numbers and their sources are listed in docs/prototype_intel_fpga_cluster.md
and in the research notes; dates matter, re-check them before buying anything.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

Range = Tuple[float, float]


def U(rng: random.Random, r: Range) -> float:
    """Uniform sample from an inclusive range; a (x, x) range is a constant."""
    lo, hi = r
    return lo if lo == hi else rng.uniform(lo, hi)


# --------------------------------------------------------------------------- #
# Hardware descriptions
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    name: str
    peak_mem_bw_gbs: float
    mem_bw_eff: Range                 # fraction of peak actually streamed by a GEMV kernel
    peak_int8_tops: float             # dense INT8 (AMX for Xeon, tensor cores for GPU)
    gemm_eff: Range                   # fraction of peak reached by batched GEMM
    gemv_sw_overhead_us: Range        # per-layer fixed software cost (thread fork/join, kernel launch)
    price_usd: float                  # node incl. memory, board, chassis share
    power_w: float
    mem_gb: float
    notes: str = ""


@dataclass
class Fabric:
    name: str
    kind: str                         # "none" | "upi" | "rdma" | "fpga" | "tcp"
    link_bw_gbs: float                # per direction, per node, usable
    host_io_lat_us: Range             # time to get a buffer from CPU memory onto the wire (DMA setup, PCIe)
    hop_lat_us: Range                 # per-hop latency inside the fabric (switch / FPGA pipeline / UPI)
    sw_overhead_us: Range             # software cost per collective (oneCCL/NCCL call, polling)
    hops_fn: str = "ring"             # "ring": N-1 hops; "tree": ceil(log2 N) hops (hardware reduce)
    price_per_node_usd: float = 0.0   # NIC or FPGA card + cables per node
    power_per_node_w: float = 0.0
    notes: str = ""


@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    bytes_per_param: float
    active_params_b: float
    moe_total_params_b: Optional[float] = None
    kv_bytes_per_token_per_layer: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        if self.kv_bytes_per_token_per_layer is None:
            self.kv_bytes_per_token_per_layer = 2 * self.n_kv_heads * self.head_dim * 2  # K and V, bf16

    # weight bytes per layer by block (dense approximations; MoE active-expert scaled)
    def attn_weight_bytes(self) -> float:
        q = self.hidden * self.n_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        return (q + kv + o) * self.bytes_per_param

    def mlp_weight_bytes(self) -> float:
        return 3 * self.hidden * self.intermediate * self.bytes_per_param

    def total_active_bytes(self) -> float:
        return self.active_params_b * 1e9 * self.bytes_per_param

    def total_resident_bytes(self) -> float:
        p = self.moe_total_params_b if self.moe_total_params_b else self.active_params_b
        return p * 1e9 * self.bytes_per_param

    def flops_per_token(self) -> float:
        return 2 * self.active_params_b * 1e9


@dataclass
class Scenario:
    name: str
    node: Node
    n_nodes: int
    fabric: Fabric
    microbatches: int = 1             # >1 lets comm of one micro-batch overlap compute of the next
    fixed_cost_usd: float = 0.0       # one-off: FPGA NRE share, switch, rack
    fixed_power_w: float = 0.0
    tag: str = ""


@dataclass
class Econ:
    years: float = 3.0
    utilisation: float = 0.5
    usd_per_kwh: float = 0.15
    hosting_usd_per_node_month: float = 0.0


# --------------------------------------------------------------------------- #
# One sampled decode step
# --------------------------------------------------------------------------- #
def sample_step(sc: Scenario, m: ModelSpec, batch: int, context: int, rng: random.Random) -> Dict[str, float]:
    n = sc.n_nodes
    nd, fb = sc.node, sc.fabric
    bw = nd.peak_mem_bw_gbs * 1e9 * U(rng, nd.mem_bw_eff)
    ops = nd.peak_int8_tops * 1e12 * U(rng, nd.gemm_eff)
    sw_layer = U(rng, nd.gemv_sw_overhead_us) * 1e-6

    # ---- per-layer compute/memory on one node (its 1/n shard) -------------- #
    # The dense per-block formulas are normalised so the layer total matches the model's
    # declared active-parameter count (keeps MoE / MLA models such as DeepSeek honest).
    formula_params = (m.attn_weight_bytes() + m.mlp_weight_bytes()) / m.bytes_per_param * m.layers \
        + m.vocab * m.hidden
    scale = m.active_params_b * 1e9 / formula_params
    attn_w = m.attn_weight_bytes() * scale / n
    mlp_w = m.mlp_weight_bytes() * scale / n
    kv_read = m.kv_bytes_per_token_per_layer * context * batch / n
    layer_bytes = attn_w + mlp_w + kv_read
    layer_flops = (2 * (m.attn_weight_bytes() + m.mlp_weight_bytes()) / m.bytes_per_param) * batch / n \
                  + 2 * 2 * m.n_heads * m.head_dim * context * batch / n   # attention scores+values
    t_layer_dev = max(layer_bytes / bw, layer_flops / ops) + sw_layer

    # ---- two all-reduces per layer over the fabric ------------------------- #
    t_comm_layer = 0.0
    if n > 1:
        act_bytes = batch * m.hidden * 2        # bf16 activations
        if fb.hops_fn == "tree":
            hops = math.ceil(math.log2(n))
            wire_bytes = act_bytes               # hardware reduce streams the vector once
        else:
            hops = n - 1
            wire_bytes = 2 * (n - 1) / n * act_bytes   # ring reduce-scatter + all-gather
        per_ar = (2 * U(rng, fb.host_io_lat_us) * 1e-6           # out and back across PCIe/DMA
                  + hops * U(rng, fb.hop_lat_us) * 1e-6
                  + wire_bytes / (fb.link_bw_gbs * 1e9)
                  + U(rng, fb.sw_overhead_us) * 1e-6)
        t_comm_layer = 2 * per_ar

    # ---- lm head / embeddings (sharded by vocab) --------------------------- #
    head_bytes = m.vocab * m.hidden * m.bytes_per_param * scale / n
    t_head = max(head_bytes / bw, 2 * m.vocab * m.hidden * scale * batch / n / ops)

    t_dev = m.layers * t_layer_dev + t_head
    t_comm = m.layers * t_comm_layer
    if sc.microbatches > 1 and n > 1:
        # comm of micro-batch i overlaps compute of micro-batch i+1 (per layer); residual = 1/k exposed
        k = sc.microbatches
        t_step = max(t_dev, t_comm) + min(t_dev, t_comm) / k
    else:
        t_step = t_dev + t_comm
    return {"t_dev": t_dev, "t_comm": t_comm, "t_step": t_step, "bw": bw, "ops": ops}


def sample_prefill(sc: Scenario, m: ModelSpec, prompt: int, rng: random.Random) -> float:
    n, nd, fb = sc.n_nodes, sc.node, sc.fabric
    ops = nd.peak_int8_tops * 1e12 * U(rng, nd.gemm_eff)
    bw = nd.peak_mem_bw_gbs * 1e9 * U(rng, nd.mem_bw_eff)
    t = max(m.flops_per_token() * prompt / (n * ops), m.total_active_bytes() / n / bw)
    if n > 1:
        act = prompt * m.hidden * 2
        wire = act if fb.hops_fn == "tree" else 2 * (n - 1) / n * act
        hops = math.ceil(math.log2(n)) if fb.hops_fn == "tree" else n - 1
        t += 2 * m.layers * (wire / (fb.link_bw_gbs * 1e9) + hops * U(rng, fb.hop_lat_us) * 1e-6
                             + 2 * U(rng, fb.host_io_lat_us) * 1e-6)
    return prompt / t


def capex(sc: Scenario) -> float:
    return sc.n_nodes * (sc.node.price_usd + sc.fabric.price_per_node_usd) + sc.fixed_cost_usd


def power(sc: Scenario) -> float:
    return sc.n_nodes * (sc.node.power_w + sc.fabric.power_per_node_w) + sc.fixed_power_w


def usd_per_mtok(sc: Scenario, tps: float, e: Econ) -> float:
    hours = e.years * 365 * 24
    tokens = tps * 3600 * hours * e.utilisation
    cost = capex(sc) + power(sc) / 1000 * hours * e.usd_per_kwh \
        + e.hosting_usd_per_node_month * sc.n_nodes * 12 * e.years
    return cost / tokens * 1e6


def fits(sc: Scenario, m: ModelSpec, context: int, batch: int) -> bool:
    need = m.total_resident_bytes() + m.kv_bytes_per_token_per_layer * m.layers * context * batch
    return need <= sc.node.mem_gb * 1e9 * sc.n_nodes


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def run(sc: Scenario, m: ModelSpec, batches: List[int], context: int, prompt: int,
        samples: int, e: Econ, seed: int) -> Dict:
    rng = random.Random(seed)
    out: Dict = {"scenario": sc.name, "tag": sc.tag, "n_nodes": sc.n_nodes,
                 "capex_usd": round(capex(sc)), "power_w": round(power(sc))}
    for b in batches:
        if not fits(sc, m, context, b):
            out[f"b{b}"] = None
            continue
        tps, comm_share = [], []
        for _ in range(samples):
            s = sample_step(sc, m, b, context, rng)
            tps.append(b / s["t_step"])
            comm_share.append(s["t_comm"] / (s["t_dev"] + s["t_comm"]))
        out[f"b{b}"] = {
            "tps_p10": pct(tps, .1), "tps_p50": pct(tps, .5), "tps_p90": pct(tps, .9),
            "usd_per_mtok_p50": usd_per_mtok(sc, pct(tps, .5), e),
            "usd_per_mtok_p10": usd_per_mtok(sc, pct(tps, .9), e),   # best case cost = p90 speed
            "usd_per_mtok_p90": usd_per_mtok(sc, pct(tps, .1), e),
            "comm_share_p50": pct(comm_share, .5),
        }
    pf = [sample_prefill(sc, m, prompt, rng) for _ in range(max(50, samples // 10))]
    out["prefill_tps_p50"] = pct(pf, .5)
    return out


# --------------------------------------------------------------------------- #
# Presets - every number is dated/sourced in research_notes/FPGA synced CPUs as GPU
# alternative/*.md and summarised in docs/prototype_intel_fpga_cluster.md.
# "est." marks a figure the notes could not source (no public price, unpublished spec).
# --------------------------------------------------------------------------- #
MODELS: Dict[str, ModelSpec] = {
    "llama70b-q4": ModelSpec("Llama-3.3-70B, 4-bit (~4.6 bpw)", 80, 8192, 28672, 64, 8, 128, 128256,
                             0.57, 70.6, notes="~40 GB weights; KV 4 KiB/token/layer bf16"),
    "llama70b-int8": ModelSpec("Llama-3.3-70B, INT8 (AMX-native)", 80, 8192, 28672, 64, 8, 128, 128256,
                               1.0, 70.6, notes="~70 GB weights; what SGLang-CPU runs on Xeon 6"),
    "llama8b-int8": ModelSpec("Llama-3.1-8B, INT8", 32, 4096, 14336, 32, 8, 128, 128256, 1.0, 8.0),
    "llama8b-q4": ModelSpec("Llama-3.1-8B, 4-bit", 32, 4096, 14336, 32, 8, 128, 128256, 0.57, 8.0),
    "deepseek-v3-int8": ModelSpec("DeepSeek-V3/R1 671B MoE, INT8", 61, 7168, 2048 * 8 + 18432, 128, 128, 128,
                                  129280, 1.0, 37.0, moe_total_params_b=671.0,
                                  kv_bytes_per_token_per_layer=(512 + 64) * 2,
                                  notes="37B active; ~671 GB resident at INT8 (needs >= 768 GB)"),
    "deepseek-v3-q4": ModelSpec("DeepSeek-V3/R1 671B MoE, 4-bit", 61, 7168, 2048 * 8 + 18432, 128, 128, 128,
                                129280, 0.57, 37.0, moe_total_params_b=671.0,
                                kv_bytes_per_token_per_layer=(512 + 64) * 2,
                                notes="37B active; ~380 GB resident"),
}

# Memory pricing used below (Oct 2026): DDR5 ECC RDIMM $29-44/GB, median ~$35/GB
# (capitalandcompute / datacenterdisk indices); MRDIMM-8800 has no public price, est. $45/GB.
DDR5_USD_PER_GB = 35.0
MRDIMM_USD_PER_GB = 45.0       # est.
HOST_SHARE_USD = 3500.0        # board, chassis, PSU, boot SSD, 25G NIC share for a server node (est.)
HOST_SHARE_W = 120.0

NODES: Dict[str, Node] = {
    # ---- Intel Xeon 6 nodes (AMX).  peak TOPS = cores x 2048 INT8 ops/cycle x base clock (Intel);
    #      sustained AMX clock is unpublished, and measured LLM prefill reaches only ~5-15 % of this
    #      peak (SGLang 2x6980P: 388-543 tok/s on 70B-671B INT8 = ~27 TFLOPS/socket; ktransformers
    #      AMX kernel ~7 %), hence the low gemm_eff range.  mem_bw_eff: SGLang's kernel-level figure is
    #      85 % (1.45 TB/s on dual MRDIMM-8800) but END-TO-END decode reaches 54 % (70B INT8: 70 GB in
    #      76.5 ms = 915 GB/s) and 32 % (R1 INT8: 37 GB in 68 ms), so the range is 0.32-0.80
    #      (dense models sit in the upper half, MoE models at the bottom).
    "xeon-6980p-mrdimm": Node("Xeon 6980P 128c + 12x32 GB MRDIMM-8800 (384 GB)", 845, (0.32, 0.80), 524,
                              (0.04, 0.15), (8, 25), 14000 + 384 * MRDIMM_USD_PER_GB + HOST_SHARE_USD,
                              500 + HOST_SHARE_W, 384,
                              "CPU list $17,800, street ~$14k; MRDIMM $/GB est."),
    "xeon-6980p-ddr5": Node("Xeon 6980P 128c + 12x16 GB DDR5-6400 (192 GB)", 614, (0.32, 0.80), 524,
                            (0.04, 0.15), (8, 25), 14000 + 192 * DDR5_USD_PER_GB + HOST_SHARE_USD,
                            500 + HOST_SHARE_W, 192,
                            "all 12 channels populated with the smallest RDIMMs = cheapest full bandwidth"),
    "xeon-6972p-ddr5": Node("Xeon 6972P 96c + 12x16 GB DDR5-6400 (192 GB)", 614, (0.32, 0.80), 472,
                            (0.04, 0.15), (8, 25), 12500 + 192 * DDR5_USD_PER_GB + HOST_SHARE_USD,
                            500 + HOST_SHARE_W, 192, "list $14,600"),
    "xeon-6732p-ddr5": Node("Xeon 6732P 32c + 8x16 GB DDR5-6400 (128 GB)", 410, (0.30, 0.70), 249,
                            (0.04, 0.15), (8, 25), 4500 + 128 * DDR5_USD_PER_GB + 2500, 350 + 90, 128,
                            "list $5,295; 8-channel 6700P platform; 32 cores may not saturate 8 channels"),
    # ---- GPU / accelerator baselines (Oct 2026 street price + host share).  mem_bw_eff is the
    #      measured llama.cpp MBU (5090 ~64 %, 4090 ~71 %, 3090 ~66 %); serving stacks reach ~80 %.
    "rtx-5090": Node("RTX 5090 32 GB (street $7.4-8.5k, Oct 2026)", 1792, (0.58, 0.80), 838, (0.5, 0.75),
                     (0.5, 2.0), 7800 + 1500, 575 + 150, 32, "MSRP $1,999; dense INT8 ~838 TOPS (1,676 sparse)"),
    "rtx-4090-used": Node("RTX 4090 24 GB (used $1.5-2.5k)", 1008, (0.60, 0.80), 660, (0.5, 0.75), (0.5, 2.0),
                          2000 + 1500, 450 + 150, 24),
    "rtx-3090-used": Node("RTX 3090 24 GB (used $0.7-1.2k)", 936, (0.55, 0.78), 284, (0.5, 0.75), (0.5, 2.0),
                          950 + 1200, 350 + 120, 24, "284 dense INT8 TOPS is the standard spec value"),
    "rtx-pro-6000": Node("RTX PRO 6000 Blackwell 96 GB ($16-18k)", 1792, (0.58, 0.80), 1000, (0.5, 0.75),
                         (0.5, 2.0), 17000 + 2000, 600 + 150, 96, "dense FP8 est. ~1 PFLOPS"),
    "arc-b580": Node("Intel Arc B580 12 GB (street $580-655)", 456, (0.55, 0.75), 233, (0.4, 0.7), (0.5, 3.0),
                     620 + 800, 190 + 100, 12, "Intel's cheapest useful GPU for 8B-class models"),
    "h100-sxm": Node("H100 SXM 80 GB (used $15-28k, new $25-40k)", 3350, (0.65, 0.85), 1979, (0.55, 0.8),
                     (0.3, 1.5), 25000 + 5000, 700 + 150, 80, "rental floor $1.99/h, median $3.3-3.7/h"),
    "b200": Node("B200 180 GB ($30-50k)", 8000, (0.65, 0.85), 4500, (0.55, 0.8), (0.3, 1.5), 40000 + 6000,
                 1000 + 150, 180, "rental $3.75/h floor, $8/h median"),
    "gaudi3": Node("Intel Gaudi 3 128 GB HBM2e ($15.6k list, 2024)", 3700, (0.6, 0.8), 1835, (0.5, 0.75),
                   (0.5, 2.0), 15625 + 5000, 900 + 150, 128, "no 2026 price found"),
    "m5-ultra-96": Node("Mac Studio M5 Ultra 96 GB ($5,499)", 1200, (0.45, 0.70), 60, (0.3, 0.6), (1, 4),
                        5499, 200, 96, "1.2 TB/s; 7B Q4 179 tok/s measured (~57 % MBU)"),
    "m3-ultra-512": Node("Mac Studio M3 Ultra 512 GB ($9,499)", 819, (0.40, 0.65), 56, (0.3, 0.6), (1, 4),
                         9499, 180, 512, "70B Q4 25-30 tok/s, R1 4-bit 17-19 tok/s measured"),
    "strix-halo": Node("Ryzen AI Max+ 395 mini PC 128 GB ($1.7-2.0k)", 256, (0.70, 0.85), 50, (0.3, 0.6),
                       (1, 4), 1900, 120, 128, "70B Q4 5.1 tok/s measured (~80 % MBU)"),
    "dgx-spark": Node("NVIDIA DGX Spark 128 GB ($4,699)", 273, (0.55, 0.75), 500, (0.4, 0.7), (0.5, 2.0),
                      4699, 240, 128, "70B Q4 4.4 tok/s measured"),
    "tt-p150": Node("Tenstorrent Blackhole p150 32 GB ($1,399)", 512, (0.40, 0.55), 664, (0.2, 0.5), (2, 8),
                    1399 + 1000, 300 + 120, 32, "41-50 % MBU measured; Ollama/llama.cpp do not run"),
    # ---- FPGA as the compute engine ----------------------------------------------------- #
    "alveo-v80": Node("AMD Alveo V80, 32 GB HBM2e 819 GB/s ($9,495)", 819, (0.35, 0.85), 20, (0.3, 0.6),
                      (1, 5), 9495 + 1500, 190 + 100, 32,
                      "no AI Engines; DSP INT8 est. ~20 TOPS; FlightLLM-class designs reach ~35 % MBU, Positron claims 93 %"),
    "agilex7m-hbm": Node("Altera Agilex 7 M-series HBM2e kit, 32 GB 820 GB/s (EUR 9,889)", 820, (0.35, 0.85),
                         88, (0.3, 0.6), (1, 5), 10700 + 1500, 150 + 100, 32,
                         "AI tensor blocks 88.5 INT8 TOPS theoretical; 33-week lead time; what Positron builds on"),
}

FABRICS: Dict[str, Fabric] = {
    "none": Fabric("single device", "none", 0, (0, 0), (0, 0), (0, 0)),
    # UPI 2.0: 48 GB/s per direction per link, 3-6 links; remote-memory ~200 ns; SGLang measured 3 %
    # all-reduce overhead on 2 sockets (calibration target).
    "upi": Fabric("UPI 2.0, dual-socket board (shared-memory all-reduce)", "upi", 150, (0.05, 0.1), (0.2, 0.5),
                  (3, 12), "ring", 0, 0, "the board already 'syncs two CPUs'"),
    # 400G RoCE: adapter latency 0.5-0.6 us (EDR/HDR measured) to 1-1.5 us (400G, secondary sources);
    # ConnectX-7 class NIC ~$2.2k + cable; oneCCL ring.
    "rdma-400g": Fabric("400 GbE RoCE NICs + switch (oneCCL ring)", "rdma", 45, (0.8, 1.5), (0.6, 1.5), (3, 10),
                        "ring", 2500, 30),
    # FPGA card per node (Alveo V80: PCIe Gen5 x16, 4x QSFP56; $9,495, 190 W) doing a hardware
    # tree reduction over direct-attach cables.  Host DMA doorbell+completion ~1 us (PCIe);
    # FPGA-to-FPGA ~1 us one-way (Catapult measured 2.88 us RTT through a switch).
    "fpga-ar-v1": Fabric("FPGA hardware all-reduce card (PCIe 5 x16, direct QSFP links)", "fpga", 45,
                         (0.8, 1.5), (0.4, 1.0), (1, 4), "tree", 9495 + 400, 190),
    # CXL Type-2 attached FPGA: measured 0.4-0.7 us per uncached host access on Agilex; removes the
    # DMA doorbell path.  Needs a CXL-capable Agilex 7 I-series card (no public price; est. $10k).
    "fpga-ar-v2": Fabric("CXL-attached FPGA all-reduce (Agilex 7 I-series, future)", "fpga", 50, (0.4, 0.7),
                         (0.3, 0.7), (0.5, 2), "tree", 10000, 150, "est. price; months of RTL work"),
    "tcp-10g": Fabric("10 GbE TCP (llama.cpp RPC style)", "tcp", 1.1, (10, 30), (5, 20), (20, 60), "ring", 150, 5,
                      "0.17 ms per-token RTT measured"),
    "pcie-p2p": Fabric("PCIe 5 x16 peer-to-peer (two GPUs, no NVLink)", "pcie", 50, (1.0, 2.0), (0.5, 1.0),
                       (3, 8), "ring"),
    "nvlink": Fabric("NVLink (reference)", "nvlink", 450, (0.2, 0.5), (0.5, 1.0), (2, 5), "ring", 0, 0),
}


def default_scenarios() -> List[Scenario]:
    N, F = NODES, FABRICS
    return [
        # --- what the design has to beat ---------------------------------------------------- #
        Scenario("GPU: 1x used RTX 3090", N["rtx-3090-used"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 2x used RTX 3090, TP over PCIe", N["rtx-3090-used"], 2, F["pcie-p2p"], tag="baseline"),
        Scenario("GPU: 1x used RTX 4090", N["rtx-4090-used"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 2x used RTX 4090, TP over PCIe", N["rtx-4090-used"], 2, F["pcie-p2p"], tag="baseline"),
        Scenario("GPU: 1x RTX 5090 (2026 street price)", N["rtx-5090"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 1x RTX PRO 6000 96 GB", N["rtx-pro-6000"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 1x Intel Arc B580", N["arc-b580"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 1x H100 SXM", N["h100-sxm"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 1x B200", N["b200"], 1, F["none"], tag="baseline"),
        Scenario("GPU: 1x Intel Gaudi 3", N["gaudi3"], 1, F["none"], tag="baseline"),
        Scenario("Unified: Mac Studio M5 Ultra 96 GB", N["m5-ultra-96"], 1, F["none"], tag="baseline"),
        Scenario("Unified: Mac Studio M3 Ultra 512 GB", N["m3-ultra-512"], 1, F["none"], tag="baseline"),
        Scenario("Unified: Strix Halo 128 GB mini PC", N["strix-halo"], 1, F["none"], tag="baseline"),
        Scenario("Unified: DGX Spark 128 GB", N["dgx-spark"], 1, F["none"], tag="baseline"),
        Scenario("Alt: Tenstorrent p150", N["tt-p150"], 1, F["none"], tag="baseline"),
        # --- Rev 0: 'two CPUs synced' = one dual-socket Xeon 6 board (UPI is the sync) -------- #
        Scenario("Rev 0: 1x Xeon 6980P, 192 GB DDR5-6400", N["xeon-6980p-ddr5"], 1, F["none"], tag="rev0"),
        Scenario("Rev 0: 2x Xeon 6980P dual-socket, DDR5-6400, UPI", N["xeon-6980p-ddr5"], 2, F["upi"], tag="rev0"),
        Scenario("Rev 0: 2x Xeon 6980P dual-socket, MRDIMM-8800, UPI", N["xeon-6980p-mrdimm"], 2, F["upi"],
                 tag="rev0"),
        Scenario("Rev 0b: 2x Xeon 6972P dual-socket, DDR5-6400, UPI", N["xeon-6972p-ddr5"], 2, F["upi"], tag="rev0"),
        Scenario("Rev 0c: 2x Xeon 6732P dual-socket (budget), UPI", N["xeon-6732p-ddr5"], 2, F["upi"], tag="rev0"),
        # --- Rev 1: four hosts, off-the-shelf 400G RDMA, oneCCL/xFasterTransformer TP --------- #
        Scenario("Rev 1: 4x Xeon 6980P hosts, 400G RDMA", N["xeon-6980p-ddr5"], 4, F["rdma-400g"],
                 fixed_cost_usd=9000, fixed_power_w=300, tag="rev1"),
        Scenario("Rev 1: 8x Xeon 6980P hosts, 400G RDMA", N["xeon-6980p-ddr5"], 8, F["rdma-400g"],
                 fixed_cost_usd=12000, fixed_power_w=400, tag="rev1"),
        Scenario("Rev 1b: 8x Xeon 6732P hosts (budget), 400G RDMA", N["xeon-6732p-ddr5"], 8, F["rdma-400g"],
                 fixed_cost_usd=12000, fixed_power_w=400, tag="rev1"),
        # --- Rev 2: the proposal - FPGA hardware all-reduce fabric between the hosts ----------- #
        Scenario("Rev 2: 4x Xeon 6980P, FPGA all-reduce (Alveo V80 per node)", N["xeon-6980p-ddr5"], 4,
                 F["fpga-ar-v1"], tag="rev2"),
        Scenario("Rev 2: 4x Xeon 6980P, FPGA fabric + 4 micro-batch overlap", N["xeon-6980p-ddr5"], 4,
                 F["fpga-ar-v1"], microbatches=4, tag="rev2"),
        Scenario("Rev 2: 8x Xeon 6980P, FPGA all-reduce fabric", N["xeon-6980p-ddr5"], 8, F["fpga-ar-v1"],
                 tag="rev2"),
        Scenario("Rev 2: 8x Xeon 6732P (budget), FPGA all-reduce fabric", N["xeon-6732p-ddr5"], 8,
                 F["fpga-ar-v1"], tag="rev2"),
        Scenario("Rev 2b: 4x Xeon 6980P, CXL-attached FPGA fabric (future)", N["xeon-6980p-ddr5"], 4,
                 F["fpga-ar-v2"], microbatches=4, tag="rev2"),
        # --- Rev 3: the FPGA does the math (Positron / FlightLLM style) ------------------------ #
        Scenario("Rev 3: 1x Agilex 7 M-series HBM FPGA as the engine", N["agilex7m-hbm"], 1, F["none"], tag="rev3"),
        Scenario("Rev 3: 4x Agilex 7 M-series HBM FPGAs, FPGA fabric", N["agilex7m-hbm"], 4, F["fpga-ar-v1"],
                 tag="rev3"),
        Scenario("Rev 3: 1x Alveo V80 HBM FPGA as the engine", N["alveo-v80"], 1, F["none"], tag="rev3"),
        # --- what people actually try at home ------------------------------------------------ #
        Scenario("Anti-pattern: 8x Xeon 6732P over 10 GbE TCP", N["xeon-6732p-ddr5"], 8, F["tcp-10g"], tag="anti"),
    ]


# Measured anchor points used to sanity-check the simulator (--validate).  Sources in the notes.
ANCHORS = [
    # (scenario builder, model key, batch, context, measured tok/s, source)
    (lambda: Scenario("2x Xeon 6980P MRDIMM, UPI", NODES["xeon-6980p-mrdimm"], 2, FABRICS["upi"]),
     "llama70b-int8", 1, 1024, 13.1, "SGLang-CPU, 70B INT8, TPOT 76.5 ms (LMSYS/Intel, Jul 2025)"),
    (lambda: Scenario("2x Xeon 6980P MRDIMM, UPI", NODES["xeon-6980p-mrdimm"], 2, FABRICS["upi"]),
     "deepseek-v3-int8", 1, 1024, 14.7, "SGLang-CPU, DeepSeek-R1 INT8 (LMSYS/Intel, Jul 2025)"),
    (lambda: Scenario("1x RTX 5090", NODES["rtx-5090"], 1, FABRICS["none"]),
     "llama8b-q4", 1, 128, 300 * 3.8 / 4.6, "llama.cpp Llama-2-7B Q4_0 tg128 = 300 tok/s (3.8 GB), rescaled to 8B Q4 (4.6 GB)"),
    (lambda: Scenario("1x RTX 3090", NODES["rtx-3090-used"], 1, FABRICS["none"]),
     "llama8b-q4", 1, 128, 162 * 3.8 / 4.6, "llama.cpp Llama-2-7B Q4_0 tg128 = 162 tok/s, rescaled to 8B Q4"),
    (lambda: Scenario("M3 Ultra 512", NODES["m3-ultra-512"], 1, FABRICS["none"]),
     "deepseek-v3-q4", 1, 512, 18.0, "DeepSeek-R1 4-bit 17-19 tok/s short context (2025-26 reports; ~45 % MBU). "
                                      "Reports of 25-30 tok/s on 70B Q4 are not physically possible at 819 GB/s and were discarded"),
    (lambda: Scenario("Strix Halo", NODES["strix-halo"], 1, FABRICS["none"]),
     "llama70b-q4", 1, 512, 5.1, "llama.cpp 70B Q4 ~5.1 tok/s (2025-26 reports)"),
    (lambda: Scenario("DGX Spark", NODES["dgx-spark"], 1, FABRICS["none"]),
     "llama70b-q4", 1, 512, 4.4, "Ollama 70B Q4 4.4 tok/s (reviews, 2025-26)"),
]


def validate(samples: int, seed: int) -> None:
    rng = random.Random(seed)
    print("Simulator vs measured anchor points (batch 1 decode):\n")
    print(f"{'configuration':44s} {'model':28s} {'measured':>9s} {'sim p10':>8s} {'sim p50':>8s} {'sim p90':>8s}  source")
    for build, mk, b, ctx, meas, src in ANCHORS:
        sc, m = build(), MODELS[mk]
        tps = sorted(b / sample_step(sc, m, b, ctx, rng)["t_step"] for _ in range(samples))
        print(f"{sc.name:44s} {mk:28s} {meas:9.1f} {pct(tps,.1):8.1f} {pct(tps,.5):8.1f} {pct(tps,.9):8.1f}  {src}")
    print("\nA measured value inside or just below the p10-p90 band means the preset ranges are honest "
          "(the simulator is deliberately optimistic: no software stalls, perfect load balance).")


def fmt_row(r: Dict, batches: List[int]) -> List[str]:
    cells = [r["scenario"], f"{r['capex_usd']:,}", str(r["power_w"])]
    for b in batches:
        x = r.get(f"b{b}")
        if x is None:
            cells += ["no fit", "no fit"]
        else:
            cells += [f"{x['tps_p10']:.0f}-{x['tps_p90']:.0f} ({x['tps_p50']:.0f})",
                      f"{x['usd_per_mtok_p10']:.2f}-{x['usd_per_mtok_p90']:.2f}"]
    b1 = r.get(f"b{batches[0]}")
    cells += [f"{100 * b1['comm_share_p50']:.0f}%" if b1 else "-", f"{r['prefill_tps_p50']:,.0f}"]
    return cells


def print_table(rows: List[Dict], m: ModelSpec, batches: List[int], markdown: bool) -> None:
    hdr = ["scenario", "capex $", "W"]
    for b in batches:
        hdr += [f"tok/s b={b} p10-p90 (p50)", f"$/Mtok b={b} best-worst"]
    hdr += [f"comm% b={batches[0]}", "prefill tok/s"]
    table = [fmt_row(r, batches) for r in rows]
    if markdown:
        print(f"**Model: {m.name}** (active weights {m.total_active_bytes()/1e9:.1f} GB, resident "
              f"{m.total_resident_bytes()/1e9:.0f} GB)\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for t in table:
            print("| " + " | ".join(t) + " |")
        return
    w = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(hdr)]
    print(f"\nModel: {m.name}  (active weights {m.total_active_bytes()/1e9:.1f} GB, resident "
          f"{m.total_resident_bytes()/1e9:.0f} GB)\n")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for t in table:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(t)))


def explain(sc: Scenario, m: ModelSpec, batch: int, context: int, samples: int, seed: int) -> None:
    rng = random.Random(seed)
    parts = {"dev": [], "comm": [], "step": []}
    for _ in range(samples):
        s = sample_step(sc, m, batch, context, rng)
        parts["dev"].append(s["t_dev"]); parts["comm"].append(s["t_comm"]); parts["step"].append(s["t_step"])
    n = sc.n_nodes
    print(f"{sc.name}\n  model {m.name}, batch {batch}, context {context}, {n} node(s), fabric: {sc.fabric.name}")
    print(f"  weight bytes per node per step : {m.total_active_bytes()/n/1e9:.2f} GB")
    print(f"  compute (dev) time  p50        : {statistics.median(parts['dev'])*1e3:.2f} ms")
    print(f"  all-reduce time     p50        : {statistics.median(parts['comm'])*1e3:.2f} ms  "
          f"({2*m.layers} collectives per token)")
    print(f"  step time           p50        : {statistics.median(parts['step'])*1e3:.2f} ms  -> "
          f"{batch/statistics.median(parts['step']):.1f} tok/s")
    print(f"  capex ${capex(sc):,.0f}, power {power(sc):.0f} W")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama70b-q4", choices=sorted(MODELS))
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--utilisation", type=float, default=0.5)
    ap.add_argument("--kwh", type=float, default=0.15)
    ap.add_argument("--tags", default="", help="comma list of scenario tags to include (baseline,rev0,rev1,rev2,rev3,anti)")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--explain", metavar="SCENARIO_NAME", help="print a per-step breakdown for one scenario")
    ap.add_argument("--validate", action="store_true", help="compare the simulator with measured anchor points")
    args = ap.parse_args(argv)

    if args.validate:
        validate(args.samples, args.seed)
        return 0

    m = MODELS[args.model]
    batches = [int(x) for x in args.batches.split(",")]
    e = Econ(args.years, args.utilisation, args.kwh)
    scenarios = default_scenarios()
    if args.tags:
        keep = set(args.tags.split(","))
        scenarios = [s for s in scenarios if s.tag in keep]

    if args.explain:
        match = [s for s in scenarios if s.name == args.explain]
        if not match:
            ap.error("unknown scenario; names are:\n  " + "\n  ".join(s.name for s in scenarios))
        explain(match[0], m, batches[0], args.context, args.samples, args.seed)
        return 0

    rows = [run(sc, m, batches, args.context, args.prompt, args.samples, e, args.seed) for sc in scenarios]
    if args.json:
        print(json.dumps({"model": asdict(m), "econ": asdict(e), "rows": rows}, indent=2))
    else:
        print_table(rows, m, batches, args.markdown)
        if not args.markdown:
            print(f"\n{args.samples} Monte-Carlo samples per cell; ranges are P10-P90 over the uncertain parameters. "
                  f"Cost: {e.years:g}-yr amortisation, {e.utilisation:.0%} utilisation, ${e.usd_per_kwh}/kWh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
