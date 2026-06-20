# Bitcoin PIR Benchmark Protocol

This note defines how benchmark-backed claims in the whitepaper should be
generated. The goal is to make each table reproducible, comparable across
database scales, and honest about which environment was measured.

## Current Scale Set

The committed local runs use artifacts under `/Volumes/Bitcoin/data`:

| Scale | Artifact root | Role |
| --- | --- | --- |
| `delta_940611_944000` | `deltas/940611_944000` | Small real delta for smoke and scaling trend checks. |
| `delta_940611_948454` | `deltas/940611_948454` | Current measured delta used in the deployment path. |
| `full_948454` | `checkpoints/948454` | Current full checkpoint scale. |

The final paper should add at least one future full checkpoint, and may add
synthetic miniature databases only for correctness isolation. Synthetic rows
must not be mixed into real-artifact latency claims.

## Environment Classes

Report environment in the table caption or surrounding text:

| Class | Purpose | Required metadata |
| --- | --- | --- |
| Local warm-cache | CPU/protocol-path baseline after a fixed warmup. | Machine, CPU count, memory, git revisions, artifact root, trials, warmups. |
| Local cold-cache | Sensitivity to disk/page-cache state. | Cache eviction or reboot method, storage device, same metadata as warm-cache. |
| Two-host LAN/WAN | Deployed network latency and server placement. | Host regions, RTT, transport, server concurrency, TLS/WebSocket framing. |
| Browser/WASM | Wallet-facing client cost. | Browser, device, wasm build, JS timing source, network placement. |
| TEE deployment | Confidential-compute path. | Instance type, SEV-SNP/TEE mode, attestation path, memory pressure. |

Do not collapse these classes into one number. A local warm-cache number can
support a protocol implementation claim, but not a deployed wallet latency
claim.

## Trial Discipline

Online latency tables should use at least 30 measured trials per
configuration after a fixed warmup. Each runner should emit one CSV row per
measured trial and include enough fields to recompute the table:

- database name and artifact path
- protocol, level, backend, and cache/environment label
- trial index and warmup count
- upload/download byte counts
- phase timings, not just a total
- verified query count and failure count
- git revisions for the whitepaper and implementation repo

Preprocessing measurements may use fewer independent trials when a run is
expensive, but the table must say so and separate preprocessing from online
latency.

## Protocol Coverage

| Protocol path | Current committed evidence | Remaining publication work |
| --- | --- | --- |
| Artifact sizes and DPF wire cost | Deterministic CSV/TeX from `collect_artifact_metrics.py`. | Add future checkpoint and verify deterministic root bundle provenance. |
| Local scan baseline | 30 warm-cache trials over current real artifacts. | Add cold-cache runs and explain storage/cache conditions. |
| DPF-PIR local online round | 30 warm-cache trials over current real artifacts; verifies selected bins. | Add deployed WebSocket, wallet query assignment, and Merkle proof retrieval/verification. |
| HarmonyPIR offline hints | 3 warm-cache trials over current real artifacts for HMR12 and FastPRP. | Increase selected online configurations to 30 trials; add hint download and browser/WASM client timing. |
| HarmonyPIR online query | 3 warm-cache trials over current real artifacts for HMR12 and FastPRP; verifies recovered bins. | Add 30-trial selected configurations and deployed query-server placement. |
| OnionPIR | Synthetic-capacity phase timing only. | Replace with a correctness-preserving real-artifact run before using it as end-to-end evidence. |
| Merkle proof path | Artifact sizes are recorded indirectly. | Add proof retrieval and verification timing for each protocol path. |

## Promotion Criteria

A benchmark row is ready for an academic table only when:

1. The raw trial CSV is committed or archived with an immutable reference.
2. The summary table is regenerated from that CSV, not typed by hand.
3. The run verifies correctness for randomized selected queries where the
   protocol returns data.
4. The environment class is explicit and not generalized beyond what was
   measured.
5. The result includes median and tail latency, at minimum p95 for 30-trial
   online runs.
6. Any known correctness caveat, synthetic-data caveat, or implementation
   optimization target is stated near the table.

## Reproduction Commands

Current local warm-cache commands:

```bash
./benchmarks/collect_artifact_metrics.py
./benchmarks/benchmark_scan_latency.py --trials 30 --warmups 1
./benchmarks/benchmark_dpf_round_latency.py --trials 30 --warmups 1
./benchmarks/benchmark_harmony_hint_latency.py --trials 3 --warmups 1
./benchmarks/benchmark_harmony_online_latency.py --trials 3 --warmups 1
./benchmarks/benchmark_onionpir_synthetic.py
./build.sh
```

For a selected 30-trial HarmonyPIR online follow-up, start with:

```bash
./benchmarks/benchmark_harmony_online_latency.py \
  --database full_948454 \
  --level chunk \
  --backend hmr12 \
  --trials 30 \
  --warmups 1
```

Commit the resulting CSV, summary, table, environment JSON, and rebuilt PDF
only after confirming the failure count is zero.
