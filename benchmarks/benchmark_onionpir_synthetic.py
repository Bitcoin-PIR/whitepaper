#!/usr/bin/env python3
"""Run the runtime OnionPIR synthetic-capacity benchmark and emit paper tables.

The upstream benchmark is intentionally treated as a synthetic-capacity phase
measurement. It uses the runtime's random/test data path, not the current
Bitcoin artifact files, so this script records phase costs and the reported
verification status without promoting the run to an end-to-end correctness or
deployment-latency benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUNTIME_REPO = Path("/Users/cusgadmin/BitcoinPIR")
DEFAULT_OUT_DIR = Path("benchmarks/results")
COMMAND = ["cargo", "run", "--release", "-p", "runtime", "--bin", "onionpir_bench"]


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True).strip()
    except Exception:
        return ""


def parse_duration_seconds(value: str) -> float:
    normalized = value.strip().replace("\u00b5", "u")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ns|us|ms|s)", normalized)
    if not match:
        raise ValueError(f"could not parse duration: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return amount
    if unit == "ms":
        return amount / 1_000
    if unit == "us":
        return amount / 1_000_000
    return amount / 1_000_000_000


def parse_size_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB)", value.strip())
    if not match:
        raise ValueError(f"could not parse byte size: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2)
    factor = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
    return int(round(amount * factor))


def human_bytes(n: int) -> str:
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def search_one(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"could not parse {label}")
    return match.group(1)


def parse_sections(stdout: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"={60}\n"
        r"=== (?P<label>Index level|Chunk level) \(num_entries=(?P<num_entries>[0-9]+)\) ===\n"
        r"={60}\n"
        r"(?P<body>.*?)(?=\n={60}\n=== |\n=== Benchmark complete ===|\Z)",
        flags=re.DOTALL,
    )
    rows: list[dict[str, object]] = []
    for match in pattern.finditer(stdout):
        body = match.group("body")
        galois_size, galois_time = re.search(
            r"galois_keys:\s+([0-9.]+ [KMG]?B) \(gen time: ([^)]+)\)",
            body,
        ).groups()
        gsw_size, gsw_time = re.search(
            r"gsw_keys:\s+([0-9.]+ [KMG]?B) \(gen time: ([^)]+)\)",
            body,
        ).groups()
        query_size, query_time = re.search(
            r"query size:\s+([0-9.]+ [KMG]?B) \(gen time: ([^)]+)\)",
            body,
        ).groups()
        decrypt_size, decrypt_time = re.search(
            r"decrypted size:\s+([0-9.]+ [KMG]?B) \(time: ([^)]+)\)",
            body,
        ).groups()
        response_size = search_one(r"response size:\s+([0-9.]+ [KMG]?B)", body, "response size")
        populate_time = search_one(r"populate time .*?: ([^\n]+)", body, "populate time")
        save_realign_ms = search_one(r"DB gen\+NTT\+realign: ([0-9]+) ms", body, "DB gen+NTT+realign")
        answer_time = search_one(r"answer_query avg: ([^ ]+) \([0-9]+ rounds\)", body, "answer time")
        rounds = search_one(r"answer_query avg: [^ ]+ \(([0-9]+) rounds\)", body, "rounds")
        logical_db_mb = float(search_one(r"db_size_mb:\s+([0-9.]+) MB", body, "logical DB MB"))
        physical_db_mb = float(search_one(r"physical_size_mb:\s+([0-9.]+) MB", body, "physical DB MB"))
        ntt_expansion = float(search_one(r"NTT expansion:\s+([0-9.]+)x", body, "NTT expansion"))

        row = {
            "level": "index" if match.group("label") == "Index level" else "chunk",
            "num_entries": int(match.group("num_entries")),
            "padded_entries": int(search_one(r"num_entries \(padded\):\s+([0-9]+)", body, "padded entries")),
            "entry_size_bytes": int(search_one(r"entry_size:\s+([0-9]+) bytes", body, "entry size")),
            "logical_db_mb": f"{logical_db_mb:.2f}",
            "physical_db_mb": f"{physical_db_mb:.2f}",
            "ntt_expansion": f"{ntt_expansion:.2f}",
            "populate_ntt_s": f"{parse_duration_seconds(populate_time):.9f}",
            "save_ntt_realign_s": f"{float(save_realign_ms) / 1_000:.9f}",
            "galois_keys_bytes": parse_size_bytes(galois_size),
            "galois_keygen_s": f"{parse_duration_seconds(galois_time):.9f}",
            "gsw_keys_bytes": parse_size_bytes(gsw_size),
            "gsw_keygen_s": f"{parse_duration_seconds(gsw_time):.9f}",
            "total_keys_bytes": parse_size_bytes(search_one(r"total keys:\s+([0-9.]+ [KMG]?B)", body, "total keys")),
            "query_bytes": parse_size_bytes(query_size),
            "query_gen_s": f"{parse_duration_seconds(query_time):.9f}",
            "response_bytes": parse_size_bytes(response_size),
            "answer_avg_s": f"{parse_duration_seconds(answer_time):.9f}",
            "decrypt_bytes": parse_size_bytes(decrypt_size),
            "decrypt_s": f"{parse_duration_seconds(decrypt_time):.9f}",
            "timing_rounds": int(rounds),
            "verification": search_one(r"verification:\s+([^\n]+)", body, "verification").strip(),
        }
        rows.append(row)
    if len(rows) != 2:
        raise ValueError(f"expected two OnionPIR sections, found {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by benchmarks/benchmark_onionpir_synthetic.py.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Level & Entries & NTT store & Keys & Query & Response & Answer avg \\\\",
        "\\hline",
    ]
    for row in rows:
        level = "INDEX" if row["level"] == "index" else "CHUNK"
        entries = f"{int(row['num_entries']):,} ({int(row['padded_entries']):,})"
        lines.append(
            f"{level} & {entries} & {float(row['physical_db_mb']):.0f} MB & "
            f"{human_bytes(int(row['total_keys_bytes']))} & {human_bytes(int(row['query_bytes']))} & "
            f"{human_bytes(int(row['response_bytes']))} & {float(row['answer_avg_s']):.3f} s \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\caption{OnionPIR synthetic-capacity phase benchmark from \\texttt{runtime/src/bin/onionpir\\_bench}. Entries show requested entries with padded entries in parentheses. The benchmark uses the runtime random-data path and is not an end-to-end real-artifact latency measurement.}",
        "\\label{tab:onionpir-synthetic}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def collect_metadata(runtime_repo: Path, returncode: int, stderr: str) -> dict[str, object]:
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "runtime_repo": str(runtime_repo),
        "runtime_git_rev": run_text(["git", "rev-parse", "HEAD"], cwd=runtime_repo),
        "whitepaper_git_rev": run_text(["git", "rev-parse", "HEAD"]),
        "system": platform.platform(),
        "cpu": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpus": run_text(["sysctl", "-n", "hw.logicalcpu"]),
        "memory_bytes": run_text(["sysctl", "-n", "hw.memsize"]),
        "returncode": returncode,
        "stderr": stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-repo", type=Path, default=DEFAULT_RUNTIME_REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Regenerate CSV/TeX from onionpir_synthetic_stdout.txt without rerunning cargo.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.out_dir / "onionpir_synthetic_stdout.txt"
    stderr = ""
    returncode = 0

    if args.parse_only:
        stdout = stdout_path.read_text()
    else:
        completed = subprocess.run(
            COMMAND,
            cwd=args.runtime_repo,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        stdout_path.write_text(stdout)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    rows = parse_sections(stdout)
    write_csv(args.out_dir / "onionpir_synthetic_summary.csv", rows)
    write_tex(args.out_dir / "onionpir_synthetic_table.tex", rows)
    metadata = collect_metadata(args.runtime_repo, returncode, stderr)
    (args.out_dir / "onionpir_synthetic_environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"Wrote {stdout_path}")
    print(f"Wrote {args.out_dir / 'onionpir_synthetic_summary.csv'}")
    print(f"Wrote {args.out_dir / 'onionpir_synthetic_table.tex'}")


if __name__ == "__main__":
    main()
