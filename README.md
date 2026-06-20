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
./build.sh
```
