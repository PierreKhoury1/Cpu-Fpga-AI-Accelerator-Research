# CPU + FPGA AI accelerator research

Can an FPGA that synchronises several CPUs make them act like a GPU for AI inference, at
GPU-class performance and lower cost? This repository holds the research, a concrete Intel
Xeon based prototype sketch, and two simulators that let you test the idea on paper with your
own prices before buying hardware.

## Verdict in one paragraph

An FPGA sync fabric fixes the one thing it can fix, the latency of the 160 all-reduces per token
that tensor parallelism needs, and that is worth about 1 ms of a 33 ms token step on four Xeon
6980P hosts (about 3 %). The step is set by streaming the model weights out of DRAM, which the
fabric cannot touch, and in October 2026 a Xeon socket's DRAM bandwidth costs $40-60 per GB/s
against about $2 per GB/s for a used consumer GPU. The simulated cluster lands at $6-12 per
million tokens (70B model, batch 16) versus $0.5-1.0 for a GPU or a Mac Studio and $0.32 at a
public API. The only FPGA architecture with a physics case is one where an HBM FPGA does the
math itself, which is what Positron sells and is moving off FPGAs to an ASIC.

## Contents

| Path | What it is |
|---|---|
| `reports/FPGA synced CPUs as GPU alternative.md` | The full research report: landscape, what the FPGA can and cannot solve, the prototype, test plan, alternatives |
| `docs/prototype_intel_fpga_cluster.md` | Build sheet with real parts and Oct 2026 prices, design revisions Rev 0 (dual-socket Xeon 6) to Rev 3 (HBM FPGA engine), simulated estimates, go/no-go gates |
| `docs/sim_results.md` | Generated simulation tables for four models and the validation against seven measured data points |
| `tools/sim_xeon_fpga_cluster.py` | Monte-Carlo, layer-by-layer simulator of N Xeon hosts joined by UPI, 400G RDMA or an FPGA all-reduce card, with GPU baselines; outputs P10-P90 bands |
| `tools/inference_cost_model.py` | Simpler roofline + communication calculator with a `--custom` mode for your own device, link and price |
| `tools/test_inference_cost_model.py` | Unit tests for the calculator's formulas |
| `research_notes/FPGA synced CPUs as GPU alternative/` | Five sourced, dated note files behind the report (CPU capability and cost; multi-CPU scaling and interconnects; GPU and cheap-baseline economics; FPGA accelerators; roofline theory and alternative accelerators) |

## Running the simulators

No dependencies beyond Python 3.

```bash
# all scenarios, Llama-70B 4-bit, batch 1 and 16, with P10-P90 bands
python3 tools/sim_xeon_fpga_cluster.py --model llama70b-q4 --batches 1,16

# check the simulator against measured points
python3 tools/sim_xeon_fpga_cluster.py --validate

# per-step breakdown for one scenario
python3 tools/sim_xeon_fpga_cluster.py --explain "Rev 2: 4x Xeon 6980P, FPGA all-reduce (Alveo V80 per node)"

# the quick calculator with your own device: bandwidth GB/s, INT8 TOPS, price, watts, N devices, link
python3 tools/inference_cost_model.py --custom --dev-bw 614 --dev-tops 524 --dev-price 24000 \
    --dev-power 620 --dev-eff 0.1 --n 4 --link fpga-ideal --model llama70b-q4

# unit tests
cd tools && python3 -m unittest test_inference_cost_model
```

Every preset number is dated and sourced in the research notes; figures the notes could not
source are marked `est.` in the code and documents.
