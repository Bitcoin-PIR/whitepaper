#!/usr/bin/env python3
"""Collect reproducible artifact metrics for the Bitcoin PIR whitepaper.

This script intentionally measures only stable, file-derived quantities:
database artifact sizes, cuckoo-table headers, and exact DPF communication
costs implied by those headers. Latency/throughput benchmarks belong in a
separate controlled run because they depend on machine load, cache state, and
deployment topology.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INDEX_SLOT_SIZE = 13
CHUNK_SLOT_SIZE = 44
INDEX_RECORD_SIZE = 25
CHUNK_RECORD_SIZE = 40
DPF_KEYS_PER_GROUP = 2
DPF_SERVERS = 2


@dataclass(frozen=True)
class DatabaseSpec:
    name: str
    kind: str
    base_height: int
    tip_height: int
    artifact_dir: Path
    index_source: Path | None
    chunk_source: Path | None
    raw_source: Path | None


@dataclass(frozen=True)
class CuckooHeader:
    magic: int
    k: int
    slots_per_bin: int
    bins_per_table: int
    num_hashes: int
    master_seed: int
    tag_seed: int | None


def byte_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size


def gib(n: int | None) -> str:
    if n is None:
        return ""
    return f"{n / 1024**3:.3f}"


def mib(n: int | None) -> str:
    if n is None:
        return ""
    return f"{n / 1024**2:.2f}"


def human_bytes(n: int | None) -> str:
    if n is None:
        return "--"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def parse_cuckoo_header(path: Path) -> CuckooHeader:
    with path.open("rb") as f:
        data = f.read(40)
    if len(data) < 32:
        raise ValueError(f"{path} is too small to contain a cuckoo header")
    magic, k, slots, bins, num_hashes, master_seed = struct.unpack_from("<QIIIIQ", data, 0)
    tag_seed = None
    if len(data) >= 40 and slots == 4:
        tag_seed = struct.unpack_from("<Q", data, 32)[0]
    return CuckooHeader(magic, k, slots, bins, num_hashes, master_seed, tag_seed)


def dpf_n(bins_per_table: int) -> int:
    if bins_per_table <= 1:
        return 1
    return math.ceil(math.log2(bins_per_table))


def dpf_key_wire_size(domain_bits: int) -> int:
    # Mirrors build/src/test_chunk_pir_batched.rs:
    # (dpf_n + 2) AES blocks plus one correction-bit vector.
    return (domain_bits + 2) * 16 + math.ceil(domain_bits / 8)


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def collect_environment() -> dict[str, object]:
    cpu = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    physical = run_text(["sysctl", "-n", "hw.physicalcpu"])
    logical = run_text(["sysctl", "-n", "hw.logicalcpu"])
    mem = run_text(["sysctl", "-n", "hw.memsize"])
    sw_vers = run_text(["sw_vers"])
    git_rev = run_text(["git", "-C", "/Users/cusgadmin/BitcoinPIR", "rev-parse", "HEAD"])
    return {
        "system": platform.platform(),
        "cpu": cpu,
        "physical_cpus": int(physical) if physical.isdigit() else physical,
        "logical_cpus": int(logical) if logical.isdigit() else logical,
        "memory_bytes": int(mem) if mem.isdigit() else mem,
        "macos": sw_vers,
        "bitcoinpir_git_rev": git_rev,
    }


def metric_specs(data_root: Path) -> list[DatabaseSpec]:
    return [
        DatabaseSpec(
            name="full_948454",
            kind="full",
            base_height=0,
            tip_height=948454,
            artifact_dir=data_root / "checkpoints" / "948454",
            index_source=data_root / "intermediate" / "full_948454" / "utxo_chunks_index_nodust.bin",
            chunk_source=data_root / "intermediate" / "full_948454" / "utxo_chunks_nodust.bin",
            raw_source=data_root / "intermediate" / "full_948454" / "utxo_set.bin",
        ),
        DatabaseSpec(
            name="delta_940611_948454",
            kind="delta",
            base_height=940611,
            tip_height=948454,
            artifact_dir=data_root / "deltas" / "940611_948454",
            index_source=data_root / "intermediate" / "delta_index_940611_948454.bin",
            chunk_source=data_root / "intermediate" / "delta_chunks_940611_948454.bin",
            raw_source=data_root / "intermediate" / "delta_grouped_940611_948454.bin",
        ),
        DatabaseSpec(
            name="delta_940611_944000",
            kind="delta",
            base_height=940611,
            tip_height=944000,
            artifact_dir=data_root / "deltas" / "940611_944000",
            index_source=data_root / "intermediate" / "delta_index_940611_944000.bin",
            chunk_source=data_root / "intermediate" / "delta_chunks_940611_944000.bin",
            raw_source=data_root / "intermediate" / "delta_grouped_940611_944000.bin",
        ),
    ]


def merkle_bucket_bytes(path: Path) -> int | None:
    files = list(path.glob("merkle_bucket_*"))
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    return sum(p.stat().st_size for p in files)


def artifact_rows(specs: Iterable[DatabaseSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        logical_index = byte_size(spec.index_source)
        logical_chunk = byte_size(spec.chunk_source)
        raw = byte_size(spec.raw_source)
        artifacts = {
            "raw_input_or_delta_grouped": raw,
            "logical_index_records": logical_index,
            "logical_chunk_payload": logical_chunk,
            "dpf_index_cuckoo": byte_size(spec.artifact_dir / "batch_pir_cuckoo.bin"),
            "dpf_chunk_cuckoo": byte_size(spec.artifact_dir / "chunk_pir_cuckoo.bin"),
            "bucket_merkle_total": merkle_bucket_bytes(spec.artifact_dir),
            "onion_ntt_store": byte_size(spec.artifact_dir / "onion_shared_ntt.bin"),
            "onion_index_all": byte_size(spec.artifact_dir / "onion_index_all.bin"),
        }
        index_records = logical_index // INDEX_RECORD_SIZE if logical_index else None
        chunk_records = logical_chunk // CHUNK_RECORD_SIZE if logical_chunk else None
        for artifact, size in artifacts.items():
            rows.append(
                {
                    "database": spec.name,
                    "kind": spec.kind,
                    "base_height": spec.base_height,
                    "tip_height": spec.tip_height,
                    "artifact": artifact,
                    "bytes": "" if size is None else size,
                    "mib": mib(size),
                    "gib": gib(size),
                    "index_records": "" if index_records is None else index_records,
                    "chunk_records": "" if chunk_records is None else chunk_records,
                }
            )
    return rows


def communication_rows(specs: Iterable[DatabaseSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        for level, filename, slot_size in [
            ("index", "batch_pir_cuckoo.bin", INDEX_SLOT_SIZE),
            ("chunk", "chunk_pir_cuckoo.bin", CHUNK_SLOT_SIZE),
        ]:
            path = spec.artifact_dir / filename
            if not path.exists():
                continue
            h = parse_cuckoo_header(path)
            bits = dpf_n(h.bins_per_table)
            key_size = dpf_key_wire_size(bits)
            result_size = h.slots_per_bin * slot_size
            keys_per_round_per_server = h.k * DPF_KEYS_PER_GROUP
            upload_per_server = keys_per_round_per_server * key_size
            download_per_server = keys_per_round_per_server * result_size
            total_both = DPF_SERVERS * (upload_per_server + download_per_server)
            rows.append(
                {
                    "database": spec.name,
                    "kind": spec.kind,
                    "level": level,
                    "k": h.k,
                    "slots_per_bin": h.slots_per_bin,
                    "bins_per_table": h.bins_per_table,
                    "dpf_n": bits,
                    "dpf_key_bytes": key_size,
                    "result_bytes": result_size,
                    "upload_per_server_bytes": upload_per_server,
                    "download_per_server_bytes": download_per_server,
                    "total_both_servers_bytes": total_both,
                    "total_both_servers_mib": f"{total_both / 1024**2:.3f}",
                    "artifact_bytes": path.stat().st_size,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex_tables(path: Path, artifact: list[dict[str, object]], comm: list[dict[str, object]]) -> None:
    by_db: dict[str, dict[str, int]] = {}
    for row in artifact:
        if row["bytes"] == "":
            continue
        by_db.setdefault(str(row["database"]), {})[str(row["artifact"])] = int(row["bytes"])

    comm_subset = [
        r
        for r in comm
        if r["database"] in {"full_948454", "delta_940611_948454"} and r["level"] in {"index", "chunk"}
    ]

    lines: list[str] = []
    lines.append("% Auto-generated by benchmarks/collect_artifact_metrics.py.")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\hline")
    lines.append("Artifact & Full snapshot & 7,843-block delta \\\\")
    lines.append("\\hline")
    labels = [
        ("Raw UTXO/delta input", "raw_input_or_delta_grouped"),
        ("Logical index", "logical_index_records"),
        ("Logical chunk payload", "logical_chunk_payload"),
        ("DPF/Harmony index cuckoo", "dpf_index_cuckoo"),
        ("DPF/Harmony chunk cuckoo", "dpf_chunk_cuckoo"),
        ("Bucket Merkle artifacts", "bucket_merkle_total"),
        ("OnionPIR NTT store", "onion_ntt_store"),
        ("OnionPIR index store", "onion_index_all"),
    ]
    for label, key in labels:
        full = human_bytes(by_db.get("full_948454", {}).get(key))
        delta = human_bytes(by_db.get("delta_940611_948454", {}).get(key))
        lines.append(f"{label} & {full} & {delta} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Measured artifact sizes for the current Bitcoin PIR full snapshot at height 948,454 and the delta from 940,611 to 948,454.}")
    lines.append("\\label{tab:artifact-sizes}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\hline")
    lines.append("Database & Level & Bins/table & $n$ & Key & Round traffic \\\\")
    lines.append("\\hline")
    for r in comm_subset:
        db_label = "Full" if r["database"] == "full_948454" else "Delta"
        level = "INDEX" if r["level"] == "index" else "CHUNK"
        bins = f"{int(r['bins_per_table']):,}"
        bits = int(r["dpf_n"])
        key = human_bytes(int(r["dpf_key_bytes"]))
        total = human_bytes(int(r["total_both_servers_bytes"]))
        lines.append(f"{db_label} & {level} & {bins} & {bits} & {key} & {total} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Exact two-server DPF communication per PBC round, derived from current cuckoo-table headers. Each round sends two DPF keys per group to each server and receives one bucket response for each key.}")
    lines.append("\\label{tab:dpf-round-costs}")
    lines.append("\\end{table}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Volumes/Bitcoin/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    specs = [s for s in metric_specs(args.data_root) if s.artifact_dir.exists()]
    artifacts = artifact_rows(specs)
    comm = communication_rows(specs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "artifact_metrics.csv", artifacts)
    write_csv(args.out_dir / "dpf_round_costs.csv", comm)
    write_tex_tables(args.out_dir / "artifact_tables.tex", artifacts, comm)
    (args.out_dir / "environment.json").write_text(json.dumps(collect_environment(), indent=2) + "\n")

    print(f"Wrote {args.out_dir / 'artifact_metrics.csv'}")
    print(f"Wrote {args.out_dir / 'dpf_round_costs.csv'}")
    print(f"Wrote {args.out_dir / 'artifact_tables.tex'}")
    print(f"Wrote {args.out_dir / 'environment.json'}")


if __name__ == "__main__":
    main()
