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
    # ---- CPUs (one socket + its memory) ---------------------------------- #
    "epyc-9755": Device("AMD EPYC 9755 (128c Zen5) + 12ch DDR5-6000 768GB", 450, 70, 12500, 500, 768,
                        notes="bw: measured STREAM-class ~450 GB/s of 576 theoretical; TOPS: AVX-512 VNNI est.; price ≈ CPU $12.6k list"),
    "epyc-9755-mem": Device("EPYC 9755 socket incl. 768GB DDR5 RDIMM", 450, 70, 12500 + 768 * 5, 500, 768,
                            notes="adds DDR5 RDIMM at ~$5/GB"),
    "xeon-6980p": Device("Intel Xeon 6980P (128c) + 12ch MRDIMM-8800", 700, 250, 17800, 500, 1536,
                         notes="AMX INT8 peak is much higher than AVX; measured bw ~700 GB/s with MRDIMM; price $17.8k list"),
    "xeon-6980p-mem": Device("Xeon 6980P socket incl. 768GB MRDIMM", 700, 250, 17800 + 768 * 8, 500, 768,
                             notes="MRDIMM ~$8/GB"),
    "strix-halo": Device("AMD Ryzen AI Max+ 395 mini PC 128GB", 215, 50, 2000, 120, 128,
                         notes="256-bit LPDDR5X-8000 = 256 GB/s peak, ~215 measured; NPU 50 TOPS; $2k class"),
    "m3-ultra-512": Device("Apple Mac Studio M3 Ultra 512GB", 800, 60, 9500, 250, 512,
                           notes="819 GB/s unified memory; GPU ~56 TFLOPS fp16-ish; $9.5k"),
    # ---- GPUs ------------------------------------------------------------ #
    "rtx-3090-used": Device("NVIDIA RTX 3090 24GB (used)", 936, 284, 750, 350, 24,
                            notes="GDDR6X 936 GB/s; 284 dense INT8 TOPS; used ~$750"),
    "rtx-4090": Device("NVIDIA RTX 4090 24GB", 1008, 660, 1800, 450, 24,
                       notes="1008 GB/s; 660 dense INT8 TOPS (1321 sparse)"),
    "rtx-5090": Device("NVIDIA RTX 5090 32GB", 1792, 1676, 2500, 575, 32,
                       notes="1.79 TB/s GDDR7; ~1.7 PFLOPS FP8/INT8 dense-equivalent class"),
    "h100-sxm": Device("NVIDIA H100 SXM 80GB", 3350, 1979, 25000, 700, 80,
                       notes="3.35 TB/s HBM3; 1979 dense INT8 TOPS; street ~$25k"),
    "b200": Device("NVIDIA B200 192GB", 8000, 4500, 40000, 1000, 192,
                   notes="8 TB/s HBM3e; ~4.5 PFLOPS FP8 dense; ~$40k"),
    # ---- FPGA ----------------------------------------------------------- #
    "alveo-v80": Device("AMD Alveo V80 (Versal HBM, 32GB HBM2e)", 820, 100, 10000, 190, 32,
                        notes="820 GB/s HBM2e; no AI engines, DSP INT8 est. ~100 TOPS; ~$10k"),
}

LINKS: Dict[str, Link] = {
    "pcie5-x16": Link("PCIe 5.0 x16 (what an FPGA bridge between hosts would get)", 50, 2.0,
                      notes="64 GB/s raw, ~50 usable; DMA round-trip ~1-3 us"),
    "pcie6-x16": Link("PCIe 6.0 x16", 100, 2.0),
    "cxl3-x16": Link("CXL 3.x x16 memory-semantic link", 50, 0.6,
                     notes="load/store latency ~170-250 ns device, ~0.5-1 us for a sync message"),
    "100gbe-rdma": Link("100 GbE RoCE / RDMA", 11, 4.0, notes="12.5 GB/s raw; ~2-5 us RDMA latency"),
    "400gbe-rdma": Link("400 GbE RoCE / RDMA", 45, 3.0),
    "10gbe-tcp": Link("10 GbE TCP (what llama.cpp RPC over a home LAN sees)", 1.1, 30.0),
    "upi-2socket": Link("Intel UPI / AMD xGMI between 2 sockets", 150, 0.15,
                        notes="in-package-class: ~100-200 GB/s, ~100-200 ns"),
    "nvlink5": Link("NVIDIA NVLink 5 (for reference)", 900, 1.0, notes="1.8 TB/s bidirectional per GPU"),
    "fpga-ideal": Link("Hypothetical FPGA sync fabric, PCIe-5 bandwidth, 300 ns latency", 50, 0.3,
                       notes="the most optimistic thing an FPGA bridge could be"),
}


def default_scenarios() -> List[Scenario]:
    return [
        Scenario("1x used RTX 3090", DEVICES["rtx-3090-used"], 1),
        Scenario("1x RTX 4090", DEVICES["rtx-4090"], 1),
        Scenario("1x RTX 5090", DEVICES["rtx-5090"], 1),
        Scenario("2x RTX 3090 (used) TP over PCIe", DEVICES["rtx-3090-used"], 2, LINKS["pcie5-x16"]),
        Scenario("1x H100 SXM", DEVICES["h100-sxm"], 1),
        Scenario("1x Mac Studio M3 Ultra 512GB", DEVICES["m3-ultra-512"], 1),
        Scenario("1x Strix Halo 128GB mini PC", DEVICES["strix-halo"], 1),
        Scenario("1x EPYC 9755 socket + 768GB", DEVICES["epyc-9755-mem"], 1),
        Scenario("1x Xeon 6980P socket + 768GB MRDIMM", DEVICES["xeon-6980p-mem"], 1),
        Scenario("2x Xeon 6980P (dual socket, UPI) TP", DEVICES["xeon-6980p-mem"], 2, LINKS["upi-2socket"]),
        Scenario("4x EPYC 9755 hosts, FPGA bridge @ PCIe5 TP", DEVICES["epyc-9755-mem"], 4, LINKS["fpga-ideal"],
                 extra_fixed_cost_usd=4 * 2500 + 10000, extra_power_w=4 * 150 + 200),
        Scenario("4x Xeon 6980P hosts, FPGA bridge @ PCIe5 TP", DEVICES["xeon-6980p-mem"], 4, LINKS["fpga-ideal"],
                 extra_fixed_cost_usd=4 * 2500 + 10000, extra_power_w=4 * 150 + 200),
        Scenario("8x Strix Halo, FPGA bridge @ PCIe5 TP", DEVICES["strix-halo"], 8, LINKS["fpga-ideal"],
                 extra_fixed_cost_usd=10000, extra_power_w=200),
        Scenario("8x Strix Halo, 10GbE TP (llama.cpp RPC style)", DEVICES["strix-halo"], 8, LINKS["10gbe-tcp"],
                 extra_fixed_cost_usd=1500, extra_power_w=100),
        Scenario("8x Strix Halo, 10GbE PP (pipeline)", DEVICES["strix-halo"], 8, LINKS["10gbe-tcp"],
                 parallelism="pp", extra_fixed_cost_usd=1500, extra_power_w=100),
        Scenario("1x Alveo V80 FPGA (HBM) as the engine", DEVICES["alveo-v80"], 1),
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
