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
./benchmarks/benchmark_onionpir_synthetic.py
./build.sh
```

The OnionPIR benchmark calls the runtime implementation in
`/Users/cusgadmin/BitcoinPIR` and records a synthetic-capacity phase timing.
It is not a real-artifact end-to-end latency benchmark.
