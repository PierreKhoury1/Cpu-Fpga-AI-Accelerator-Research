#!/usr/bin/env python3
"""
inference_cost_model.py - paper prototype for "N CPUs synced by an FPGA vs one GPU".

A dependency-free roofline + communication model for LLM inference.  It answers,
for a given model and hardware configuration:

  * decode tokens/s (single stream and batched)
  * prefill tokens/s
  * the per-token communication floor when the model is split across N devices
  * cost per million generated tokens (hardware amortisation + electricity)

The point of the tool is to let you test the idea *before* buying hardware:
change the numbers in PRESETS (or pass your own on the command line) and see
whether any CPU+interconnect combination gets near the GPU on $/Mtoken.

Model of a decode step (batch B, N devices, tensor parallel):
    weight bytes read per device   = W / N
    memory time                    = (W / N) / bw_dev
    compute time                   = 2 * P_active * B / (N * ops_dev)
    device time                    = max(memory time, compute time)   (roofline)
    comm per step (ring all-reduce) = 2 * layers * [ lat + 2*(N-1)/N * B*h*2 bytes / bw_link ]
    step time                      = device time + comm (decode comm is not hidden)
    tokens/s                       = B / step time

Pipeline parallel is also modelled.  Here "batch" means concurrent sequences.
At batch 1 pipeline parallel gives NO speed-up: the token still walks through
all the weights, one stage after another, so the step time equals a single
device's.  With >= N concurrent sequences every stage is busy with a different
batch and aggregate throughput approaches N x a single stage's (each stage
reads only its W/N shard per step), at the cost of N x the latency.

All formulas are deliberately simple and optimistic for the CPU side (perfect
load balance, ideal roofline, no software overhead).  If the CPU design loses
under these assumptions it loses in reality too.

Usage:
    python3 tools/inference_cost_model.py                 # compare presets
    python3 tools/inference_cost_model.py --model llama70b-q4 --json
    python3 tools/inference_cost_model.py --list
    python3 tools/inference_cost_model.py --custom \
        --dev-bw 460 --dev-tops 60 --dev-price 7000 --dev-power 400 \
        --n 4 --link-bw 64 --link-lat 2.0 --model llama70b-q4

Numbers in PRESETS are taken from the research notes in
research_notes/FPGA synced CPUs as GPU alternative/ and are dated there.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Device:
    """One compute device (a CPU socket+memory, a GPU, an FPGA board, ...)."""
    name: str
    mem_bw_gbs: float          # usable memory bandwidth, GB/s (measured, not peak, where known)
    peak_tops_int8: float      # dense INT8 TOPS (use BF16 TFLOPS if you run BF16)
    price_usd: float           # price of the device *including* the memory it needs
    power_w: float             # power under AI load, W
    mem_capacity_gb: float     # memory the model must fit in
    compute_efficiency: float = 0.5   # fraction of peak that real GEMMs reach
    notes: str = ""


@dataclass
class Link:
    """The interconnect between devices (what the FPGA in the proposal would be)."""
    name: str
    bw_gbs: float              # usable bandwidth per direction per device, GB/s
    latency_us: float          # one-way latency of a small message, microseconds
    notes: str = ""


@dataclass
class Model:
    name: str
    params_b: float            # total parameters, billions
    active_params_b: float     # parameters touched per token (== params for dense; experts+shared for MoE)
    bytes_per_param: float     # 0.5 for 4-bit, 1 for INT8, 2 for BF16
    layers: int
    hidden: int
    kv_bytes_per_token: float  # KV-cache bytes per token across all layers (both K and V)
    notes: str = ""

    @property
    def weight_bytes(self) -> float:
        return self.active_params_b * 1e9 * self.bytes_per_param

    @property
    def flops_per_token(self) -> float:
        return 2.0 * self.active_params_b * 1e9


@dataclass
class Scenario:
    name: str
    device: Device
    n_devices: int = 1
    link: Optional[Link] = None
    parallelism: str = "tp"    # "tp" (tensor) or "pp" (pipeline); ignored for n_devices == 1
    extra_fixed_cost_usd: float = 0.0   # chassis, the FPGA board itself, NICs ...
    extra_power_w: float = 0.0


@dataclass
class Economics:
    amortisation_years: float = 3.0
    utilisation: float = 0.5          # fraction of wall-clock the box is busy generating
    electricity_usd_per_kwh: float = 0.15
    hosting_usd_per_month: float = 0.0


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
def decode_step_seconds(sc: Scenario, m: Model, batch: int, context_tokens: int) -> Dict[str, float]:
    """Time for one decode step (one new token for each of `batch` sequences)."""
    d, n = sc.device, max(1, sc.n_devices)
    ops = d.peak_tops_int8 * 1e12 * d.compute_efficiency

    kv_bytes = m.kv_bytes_per_token * context_tokens * batch   # KV read per step
    bytes_per_dev = (m.weight_bytes + kv_bytes) / n
    t_mem = bytes_per_dev / (d.mem_bw_gbs * 1e9)
    t_cmp = m.flops_per_token * batch / (n * ops)
    t_dev = max(t_mem, t_cmp)

    t_comm = 0.0
    if n > 1:
        if sc.link is None:
            raise ValueError(f"scenario {sc.name}: n_devices>1 needs a link")
        act_bytes = batch * m.hidden * 2  # bf16 activations
        if sc.parallelism == "tp":
            # two all-reduces per layer (after attention, after MLP); ring all-reduce
            per_ar = sc.link.latency_us * 1e-6 * 2 * math.log2(n) + \
                     (2 * (n - 1) / n) * act_bytes / (sc.link.bw_gbs * 1e9)
            t_comm = 2 * m.layers * per_ar
        elif sc.parallelism == "pp":
            # one point-to-point hop per stage boundary, sequential at batch 1
            per_hop = sc.link.latency_us * 1e-6 + act_bytes / (sc.link.bw_gbs * 1e9)
            t_comm = (n - 1) * per_hop
            # pipeline parallel does not divide the sequential weight walk at batch 1:
            # every token still passes through every stage one after another.
            t_dev = t_dev * n if batch < n else t_dev * n / min(batch, n)
        else:
            raise ValueError("parallelism must be 'tp' or 'pp'")

    return {"t_mem": t_mem, "t_cmp": t_cmp, "t_dev": t_dev, "t_comm": t_comm,
            "t_step": t_dev + t_comm}


def decode_tokens_per_s(sc: Scenario, m: Model, batch: int = 1, context_tokens: int = 1024) -> float:
    return batch / decode_step_seconds(sc, m, batch, context_tokens)["t_step"]


def prefill_tokens_per_s(sc: Scenario, m: Model, prompt_tokens: int = 2048) -> float:
    """Prefill is compute-bound (large matrix-matrix).  Comm is amortised over the prompt."""
    d, n = sc.device, max(1, sc.n_devices)
    ops = d.peak_tops_int8 * 1e12 * d.compute_efficiency
    t_cmp = m.flops_per_token * prompt_tokens / (n * ops)
    t_mem = m.weight_bytes / n / (d.mem_bw_gbs * 1e9)
    t = max(t_cmp, t_mem)
    if n > 1 and sc.link is not None and sc.parallelism == "tp":
        act_bytes = prompt_tokens * m.hidden * 2
        t += 2 * m.layers * ((2 * (n - 1) / n) * act_bytes / (sc.link.bw_gbs * 1e9)
                             + sc.link.latency_us * 1e-6 * 2 * math.log2(n))
    return prompt_tokens / t


def fits(sc: Scenario, m: Model, context_tokens: int, batch: int) -> bool:
    need = m.weight_bytes + m.kv_bytes_per_token * context_tokens * batch
    return need <= sc.device.mem_capacity_gb * 1e9 * sc.n_devices


def capex(sc: Scenario) -> float:
    return sc.device.price_usd * sc.n_devices + sc.extra_fixed_cost_usd


def power_w(sc: Scenario) -> float:
    return sc.device.power_w * sc.n_devices + sc.extra_power_w


def usd_per_million_tokens(sc: Scenario, m: Model, econ: Economics, batch: int, context_tokens: int) -> float:
    tps = decode_tokens_per_s(sc, m, batch, context_tokens)
    hours = econ.amortisation_years * 365 * 24
    tokens = tps * 3600 * hours * econ.utilisation
    cost = capex(sc) \
        + power_w(sc) / 1000 * hours * econ.electricity_usd_per_kwh \
        + econ.hosting_usd_per_month * 12 * econ.amortisation_years
    return cost / tokens * 1e6


def comm_floor_tokens_per_s(n: int, link: Link, m: Model) -> float:
    """Upper bound on batch-1 tokens/s set by interconnect latency alone (tensor parallel)."""
    if n <= 1:
        return math.inf
    t = 2 * m.layers * link.latency_us * 1e-6 * 2 * math.log2(n)
    return 1.0 / t


# --------------------------------------------------------------------------- #
# Presets  (see research notes for sources and dates; edit freely)
# --------------------------------------------------------------------------- #
MODELS: Dict[str, Model] = {
    "llama8b-q4": Model("Llama-3.1-8B Q4_K_M", 8.0, 8.0, 0.57, 32, 4096,
                        kv_bytes_per_token=32 * 2 * 8 * 128 * 2,   # 32 layers, K+V, 8 KV heads, 128 dim, bf16
                        notes="4.9 GB weights at Q4_K_M; GQA with 8 KV heads"),
    "llama8b-bf16": Model("Llama-3.1-8B BF16", 8.0, 8.0, 2.0, 32, 4096,
                          kv_bytes_per_token=32 * 2 * 8 * 128 * 2),
    "llama70b-q4": Model("Llama-3.x-70B Q4_K_M", 70.6, 70.6, 0.57, 80, 8192,
                         kv_bytes_per_token=80 * 2 * 8 * 128 * 2,
                         notes="~40 GB weights at Q4_K_M; GQA 8 KV heads"),
    "llama70b-int8": Model("Llama-3.x-70B INT8", 70.6, 70.6, 1.0, 80, 8192,
                           kv_bytes_per_token=80 * 2 * 8 * 128 * 2),
    "deepseek-v3-q4": Model("DeepSeek-V3/R1 671B MoE Q4", 671.0, 37.0, 0.57, 61, 7168,
                            kv_bytes_per_token=61 * (512 + 64) * 2,   # MLA: compressed KV + rope part, bf16
                            notes="37B active per token; total weights ~380 GB at Q4 must still fit in memory"),
}

DEVICES: Dict[str, Device] = {
    # Prices are Oct 2026 street incl. the memory and host share each device needs; bandwidth is
    # theoretical peak and compute_efficiency/mem efficiency are folded into the numbers below as
    # "usable" values (see tools/sim_xeon_fpga_cluster.py for the ranged version and sources).
    # ---- Intel Xeon 6 sockets (AMX) ------------------------------------------------------ #
    "xeon-6980p-ddr5": Device("Xeon 6980P 128c + 12x16 GB DDR5-6400 (192 GB)", 614 * 0.62, 524, 14000 + 192 * 35 + 3500,
                              620, 192, 0.10, "usable bw = 62 % of 614 GB/s; AMX reaches ~10 % of 524 TOPS peak in LLM prefill"),
    "xeon-6980p-mrdimm": Device("Xeon 6980P 128c + 12x32 GB MRDIMM-8800 (384 GB)", 845 * 0.62, 524,
                                14000 + 384 * 45 + 3500, 620, 384, 0.10, "MRDIMM $/GB est."),
    "xeon-6972p-ddr5": Device("Xeon 6972P 96c + 12x16 GB DDR5-6400", 614 * 0.62, 472, 12500 + 192 * 35 + 3500, 620, 192, 0.10),
    "xeon-6732p-ddr5": Device("Xeon 6732P 32c + 8x16 GB DDR5-6400 (128 GB)", 410 * 0.55, 249, 4500 + 128 * 35 + 2500,
                              440, 128, 0.10, "budget 6700P socket"),
    # ---- GPUs and accelerators ------------------------------------------------------------ #
    "rtx-3090-used": Device("RTX 3090 24 GB (used $0.7-1.2k)", 936 * 0.66, 284, 950 + 1200, 470, 24, 0.6),
    "rtx-4090-used": Device("RTX 4090 24 GB (used $1.5-2.5k)", 1008 * 0.70, 660, 2000 + 1500, 600, 24, 0.6),
    "rtx-5090": Device("RTX 5090 32 GB (street $7.4-8.5k)", 1792 * 0.68, 838, 7800 + 1500, 725, 32, 0.6),
    "rtx-pro-6000": Device("RTX PRO 6000 Blackwell 96 GB ($16-18k)", 1792 * 0.68, 1000, 17000 + 2000, 750, 96, 0.6),
    "arc-b580": Device("Intel Arc B580 12 GB ($580-655)", 456 * 0.65, 233, 620 + 800, 290, 12, 0.5),
    "h100-sxm": Device("H100 SXM 80 GB ($25k used/new mid)", 3350 * 0.75, 1979, 25000 + 5000, 850, 80, 0.65),
    "b200": Device("B200 180 GB ($40k)", 8000 * 0.75, 4500, 40000 + 6000, 1150, 180, 0.65),
    "gaudi3": Device("Intel Gaudi 3 128 GB ($15.6k list 2024)", 3700 * 0.70, 1835, 15625 + 5000, 1050, 128, 0.6),
    "m5-ultra-96": Device("Mac Studio M5 Ultra 96 GB ($5,499)", 1200 * 0.55, 60, 5499, 200, 96, 0.45),
    "m3-ultra-512": Device("Mac Studio M3 Ultra 512 GB ($9,499)", 819 * 0.50, 56, 9499, 180, 512, 0.45),
    "strix-halo": Device("Ryzen AI Max+ 395 mini PC 128 GB ($1.9k)", 256 * 0.78, 50, 1900, 120, 128, 0.45),
    "dgx-spark": Device("DGX Spark 128 GB ($4,699)", 273 * 0.65, 500, 4699, 240, 128, 0.5),
    "tt-p150": Device("Tenstorrent p150 32 GB ($1,399)", 512 * 0.47, 664, 1399 + 1000, 420, 32, 0.35),
    # ---- FPGA as engine ------------------------------------------------------------------- #
    "alveo-v80": Device("AMD Alveo V80 32 GB HBM2e ($9,495)", 819 * 0.5, 20, 9495 + 1500, 290, 32, 0.45,
                        "35 % MBU demonstrated (FlightLLM-class), 93 % claimed (Positron); 50 % used"),
    "agilex7m-hbm": Device("Altera Agilex 7 M-series HBM2e kit 32 GB (~$10.7k)", 820 * 0.5, 88, 10700 + 1500, 250, 32, 0.45),
}

LINKS: Dict[str, Link] = {
    # latency = effective one-way latency of one all-reduce hop incl. software, see the notes
    "upi-2socket": Link("UPI 2.0 between 2 sockets on one board", 150, 0.4, "remote memory ~200 ns; SGLang measured 3 % overhead"),
    "pcie5-x16": Link("PCIe 5.0 x16 DMA hop (what an FPGA bridge sees on the host side)", 50, 1.5, "63 GB/s theoretical"),
    "cxl3-x16": Link("CXL Type-2 FPGA hop", 50, 0.6, "0.4-0.7 us per uncached access measured on Agilex"),
    "rdma-400g": Link("400 GbE RoCE / NDR InfiniBand", 45, 2.0, "adapter 0.5-1.5 us + software"),
    "10gbe-tcp": Link("10 GbE TCP (llama.cpp RPC)", 1.1, 50.0, "0.17 ms per-token RTT measured"),
    "fpga-ideal": Link("Hypothetical FPGA sync fabric: PCIe-5 DMA in, 1 us FPGA hops", 45, 1.2),
    "nvlink5": Link("NVIDIA NVLink 5 (reference)", 900, 1.0, "900 GB/s per direction per GPU"),
}


def default_scenarios() -> List[Scenario]:
    D, L = DEVICES, LINKS
    return [
        Scenario("1x used RTX 3090", D["rtx-3090-used"], 1),
        Scenario("2x used RTX 3090 TP over PCIe", D["rtx-3090-used"], 2, L["pcie5-x16"]),
        Scenario("1x used RTX 4090", D["rtx-4090-used"], 1),
        Scenario("2x used RTX 4090 TP over PCIe", D["rtx-4090-used"], 2, L["pcie5-x16"]),
        Scenario("1x RTX 5090 (2026 street)", D["rtx-5090"], 1),
        Scenario("1x RTX PRO 6000 96 GB", D["rtx-pro-6000"], 1),
        Scenario("1x Intel Arc B580", D["arc-b580"], 1),
        Scenario("1x H100 SXM", D["h100-sxm"], 1),
        Scenario("1x B200", D["b200"], 1),
        Scenario("1x Intel Gaudi 3", D["gaudi3"], 1),
        Scenario("1x Mac Studio M5 Ultra 96 GB", D["m5-ultra-96"], 1),
        Scenario("1x Mac Studio M3 Ultra 512 GB", D["m3-ultra-512"], 1),
        Scenario("1x Strix Halo 128 GB mini PC", D["strix-halo"], 1),
        Scenario("1x DGX Spark", D["dgx-spark"], 1),
        Scenario("1x Tenstorrent p150", D["tt-p150"], 1),
        Scenario("Rev 0: 1x Xeon 6980P socket, DDR5-6400", D["xeon-6980p-ddr5"], 1),
        Scenario("Rev 0: 2x Xeon 6980P dual socket (UPI)", D["xeon-6980p-ddr5"], 2, L["upi-2socket"]),
        Scenario("Rev 0: 2x Xeon 6980P dual socket, MRDIMM (UPI)", D["xeon-6980p-mrdimm"], 2, L["upi-2socket"]),
        Scenario("Rev 0c: 2x Xeon 6732P dual socket, budget (UPI)", D["xeon-6732p-ddr5"], 2, L["upi-2socket"]),
        Scenario("Rev 1: 4x Xeon 6980P hosts, 400G RDMA", D["xeon-6980p-ddr5"], 4, L["rdma-400g"],
                 extra_fixed_cost_usd=4 * 2500 + 9000, extra_power_w=4 * 30 + 300),
        Scenario("Rev 2: 4x Xeon 6980P hosts, FPGA bridge (Alveo V80 each)", D["xeon-6980p-ddr5"], 4, L["fpga-ideal"],
                 extra_fixed_cost_usd=4 * 9900, extra_power_w=4 * 190),
        Scenario("Rev 2: 8x Xeon 6980P hosts, FPGA bridge", D["xeon-6980p-ddr5"], 8, L["fpga-ideal"],
                 extra_fixed_cost_usd=8 * 9900, extra_power_w=8 * 190),
        Scenario("Rev 2: 8x Xeon 6732P hosts (budget), FPGA bridge", D["xeon-6732p-ddr5"], 8, L["fpga-ideal"],
                 extra_fixed_cost_usd=8 * 9900, extra_power_w=8 * 190),
        Scenario("Rev 3: 1x Agilex 7 M-series HBM FPGA as engine", D["agilex7m-hbm"], 1),
        Scenario("Rev 3: 4x Agilex HBM FPGAs, FPGA fabric", D["agilex7m-hbm"], 4, L["fpga-ideal"]),
        Scenario("Anti-pattern: 8x Xeon 6732P over 10 GbE TCP", D["xeon-6732p-ddr5"], 8, L["10gbe-tcp"],
                 extra_fixed_cost_usd=1500, extra_power_w=100),
        Scenario("Anti-pattern: 8x Xeon 6732P over 10 GbE PP", D["xeon-6732p-ddr5"], 8, L["10gbe-tcp"],
                 parallelism="pp", extra_fixed_cost_usd=1500, extra_power_w=100),
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def evaluate(sc: Scenario, m: Model, econ: Economics, context: int, batches: List[int]) -> Dict:
    row: Dict = {"scenario": sc.name, "n_devices": sc.n_devices, "capex_usd": round(capex(sc)),
                 "power_w": round(power_w(sc)), "fits": fits(sc, m, context, min(batches))}
    if not row["fits"]:
        return row
    for b in batches:
        if not fits(sc, m, context, b):
            row[f"decode_tps_b{b}"] = "no fit"
            row[f"usd_per_mtok_b{b}"] = "no fit"
            continue
        st = decode_step_seconds(sc, m, b, context)
        row[f"decode_tps_b{b}"] = round(b / st["t_step"], 1)
        row[f"usd_per_mtok_b{b}"] = round(usd_per_million_tokens(sc, m, econ, b, context), 3)
    row["comm_share_b1_pct"] = round(100 * decode_step_seconds(sc, m, 1, context)["t_comm"] /
                                     decode_step_seconds(sc, m, 1, context)["t_step"], 1)
    row["prefill_tps"] = round(prefill_tokens_per_s(sc, m))
    if sc.n_devices > 1 and sc.link is not None and sc.parallelism == "tp":
        row["latency_floor_tps"] = round(comm_floor_tokens_per_s(sc.n_devices, sc.link, m), 1)
    return row


def print_table(rows: List[Dict], m: Model, batches: List[int]) -> None:
    cols = ["scenario", "capex_usd", "power_w"] + \
           [c for b in batches for c in (f"decode_tps_b{b}", f"usd_per_mtok_b{b}")] + \
           ["prefill_tps", "comm_share_b1_pct"]
    hdr = {"scenario": "scenario", "capex_usd": "capex $", "power_w": "W",
           "prefill_tps": "prefill tok/s", "comm_share_b1_pct": "comm% @b1"}
    for b in batches:
        hdr[f"decode_tps_b{b}"] = f"tok/s b={b}"
        hdr[f"usd_per_mtok_b{b}"] = f"$/Mtok b={b}"
    widths = {c: max(len(hdr[c]), *(len(str(r.get(c, "n/a"))) for r in rows)) for c in cols}
    print(f"\nModel: {m.name}  (active weights {m.weight_bytes/1e9:.1f} GB, {m.flops_per_token/1e9:.1f} GFLOP/token)\n")
    print("  ".join(hdr[c].ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        if not r["fits"]:
            print(r["scenario"].ljust(widths["scenario"]) + "  (does not fit in memory)")
            continue
        print("  ".join(str(r.get(c, "n/a")).ljust(widths[c]) for c in cols))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama70b-q4", choices=sorted(MODELS))
    ap.add_argument("--context", type=int, default=2048, help="tokens of context during decode")
    ap.add_argument("--batches", default="1,8,32", help="comma-separated batch sizes")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--utilisation", type=float, default=0.5)
    ap.add_argument("--kwh", type=float, default=0.15)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true", help="list devices, links and models")
    # custom scenario
    ap.add_argument("--custom", action="store_true", help="evaluate one custom scenario instead of presets")
    ap.add_argument("--dev-bw", type=float, help="device memory bandwidth GB/s")
    ap.add_argument("--dev-tops", type=float, help="device dense INT8 TOPS")
    ap.add_argument("--dev-price", type=float, help="device price USD incl. memory")
    ap.add_argument("--dev-power", type=float, help="device power W")
    ap.add_argument("--dev-mem", type=float, default=256, help="device memory GB")
    ap.add_argument("--dev-eff", type=float, default=0.5, help="fraction of peak compute achieved")
    ap.add_argument("--n", type=int, default=1, help="number of devices")
    ap.add_argument("--link", choices=sorted(LINKS), help="preset link")
    ap.add_argument("--link-bw", type=float, help="link GB/s per direction")
    ap.add_argument("--link-lat", type=float, help="link latency us")
    ap.add_argument("--pp", action="store_true", help="pipeline instead of tensor parallel")
    ap.add_argument("--fixed-cost", type=float, default=0.0, help="extra fixed cost (FPGA board, chassis)")
    args = ap.parse_args(argv)

    if args.list:
        for k, d in DEVICES.items():
            print(f"device {k:16s} {d.mem_bw_gbs:6.0f} GB/s {d.peak_tops_int8:6.0f} TOPS ${d.price_usd:8.0f} {d.power_w:5.0f} W  {d.notes}")
        for k, l in LINKS.items():
            print(f"link   {k:16s} {l.bw_gbs:6.1f} GB/s {l.latency_us:6.2f} us  {l.notes}")
        for k, mm in MODELS.items():
            print(f"model  {k:16s} {mm.weight_bytes/1e9:6.1f} GB active, {mm.layers} layers, hidden {mm.hidden}  {mm.notes}")
        return 0

    m = MODELS[args.model]
    econ = Economics(args.years, args.utilisation, args.kwh)
    batches = [int(x) for x in args.batches.split(",")]

    if args.custom:
        if None in (args.dev_bw, args.dev_tops, args.dev_price, args.dev_power):
            ap.error("--custom needs --dev-bw --dev-tops --dev-price --dev-power")
        dev = Device("custom device", args.dev_bw, args.dev_tops, args.dev_price, args.dev_power,
                     args.dev_mem, args.dev_eff)
        link = None
        if args.n > 1:
            if args.link:
                link = LINKS[args.link]
            elif args.link_bw is not None and args.link_lat is not None:
                link = Link("custom link", args.link_bw, args.link_lat)
            else:
                ap.error("--n > 1 needs --link or --link-bw/--link-lat")
        scenarios = [Scenario(f"{args.n}x custom", dev, args.n, link, "pp" if args.pp else "tp", args.fixed_cost)]
    else:
        scenarios = default_scenarios()

    rows = [evaluate(sc, m, econ, args.context, batches) for sc in scenarios]
    if args.json:
        print(json.dumps({"model": asdict(m), "economics": asdict(econ), "rows": rows}, indent=2))
    else:
        print_table(rows, m, batches)
        print("\nAssumptions: ideal roofline per device, decode comm not overlapped, "
              f"{econ.amortisation_years:g}-yr amortisation at {econ.utilisation:.0%} utilisation, "
              f"${econ.electricity_usd_per_kwh}/kWh. Edit PRESETS or use --custom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
