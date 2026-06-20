# Bitcoin PIR Whitepaper

This repository contains the LaTeX source for:

**Private Information Retrieval for Bitcoin Light Clients**

## Build

The local build uses XeLaTeX and BibTeX:

```bash
./build.sh
```

The generated paper is `main.pdf`.

## Benchmark Tables

The evaluation section includes generated artifact-size and DPF communication
tables. Regenerate them from the local Bitcoin PIR artifact root with:

```bash
./benchmarks/collect_artifact_metrics.py
./benchmarks/benchmark_scan_latency.py --trials 30 --warmups 1
./benchmarks/benchmark_dpf_round_latency.py --trials 30 --warmups 1
./benchmarks/benchmark_harmony_hint_latency.py --trials 3 --warmups 1
./benchmarks/benchmark_harmony_online_latency.py --trials 30 --warmups 1
./benchmarks/benchmark_onionpir_synthetic.py
./build.sh
```

The benchmark matrix and promotion criteria are tracked in
`benchmarks/BENCHMARK_PROTOCOL.md`.

The DPF round benchmark builds a small Rust runner against
`/Users/cusgadmin/BitcoinPIR/pir-runtime-core` and records local warm-cache
key generation, two-server-share evaluation, and client verification timings.
The HarmonyPIR benchmark builds a small Rust runner against
`/Users/cusgadmin/BitcoinPIR/harmonypir-wasm` and records local warm-cache
offline hint-generation and online-query timings for HMR12 and FastPRP.
The OnionPIR benchmark calls the runtime implementation in
`/Users/cusgadmin/BitcoinPIR` and records a synthetic-capacity phase timing.
It is not a real-artifact end-to-end latency benchmark.
