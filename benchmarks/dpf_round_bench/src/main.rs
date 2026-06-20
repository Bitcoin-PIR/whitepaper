use libdpf::{Dpf, DpfKey};
use memmap2::Mmap;
use pir_core::cuckoo::read_cuckoo_header_with_anchor;
use pir_core::params::{compute_dpf_n, TableParams, CHUNK_PARAMS, INDEX_PARAMS};
use pir_runtime_core::eval::{process_chunk_group, process_index_group};
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Level {
    Index,
    Chunk,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Index => "index",
            Level::Chunk => "chunk",
        }
    }

    fn params(self) -> &'static TableParams {
        match self {
            Level::Index => &INDEX_PARAMS,
            Level::Chunk => &CHUNK_PARAMS,
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Level::Index => "batch_pir_cuckoo.bin",
            Level::Chunk => "chunk_pir_cuckoo.bin",
        }
    }
}

struct Args {
    data_root: PathBuf,
    trials: usize,
    warmups: usize,
    database: Option<String>,
    level: Option<Level>,
}

struct DatabaseSpec {
    name: &'static str,
    kind: &'static str,
    artifact_dir: PathBuf,
}

struct Artifact {
    database: String,
    kind: String,
    level: Level,
    path: PathBuf,
}

struct TrialKeys {
    targets: Vec<(u64, u64)>,
    server0: Vec<(DpfKey, DpfKey)>,
    server1: Vec<(DpfKey, DpfKey)>,
}

struct GroupOutput {
    q0: Vec<u8>,
    q1: Vec<u8>,
}

struct ServerRun {
    outputs: Vec<GroupOutput>,
    wall: Duration,
    dpf_sum: Duration,
    fetch_sum: Duration,
    checksum: u64,
}

struct VerifyRun {
    wall: Duration,
    failures: usize,
    checksum: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: dpf-round-bench [--data-root PATH] [--trials N] [--warmups N] [--database NAME] [--level index|chunk]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        data_root: PathBuf::from("/Volumes/Bitcoin/data"),
        trials: 3,
        warmups: 1,
        database: None,
        level: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--data-root" => args.data_root = PathBuf::from(iter.next().unwrap_or_else(|| usage_value())),
            "--trials" => {
                args.trials = iter
                    .next()
                    .unwrap_or_else(|| usage_value())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--warmups" => {
                args.warmups = iter
                    .next()
                    .unwrap_or_else(|| usage_value())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--database" => args.database = Some(iter.next().unwrap_or_else(|| usage_value())),
            "--level" => {
                let value = iter.next().unwrap_or_else(|| usage_value());
                args.level = match value.as_str() {
                    "index" => Some(Level::Index),
                    "chunk" => Some(Level::Chunk),
                    "all" => None,
                    _ => usage(),
                }
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    args
}

fn usage_value() -> String {
    usage()
}

fn database_specs(data_root: &Path) -> Vec<DatabaseSpec> {
    vec![
        DatabaseSpec {
            name: "full_948454",
            kind: "full",
            artifact_dir: data_root.join("checkpoints").join("948454"),
        },
        DatabaseSpec {
            name: "delta_940611_948454",
            kind: "delta",
            artifact_dir: data_root.join("deltas").join("940611_948454"),
        },
        DatabaseSpec {
            name: "delta_940611_944000",
            kind: "delta",
            artifact_dir: data_root.join("deltas").join("940611_944000"),
        },
    ]
}

fn selected_artifacts(args: &Args) -> Vec<Artifact> {
    let levels = match args.level {
        Some(level) => vec![level],
        None => vec![Level::Index, Level::Chunk],
    };
    let mut out = Vec::new();
    for db in database_specs(&args.data_root) {
        if let Some(name) = &args.database {
            if name != "all" && name != db.name {
                continue;
            }
        }
        for level in &levels {
            let path = db.artifact_dir.join(level.file_name());
            if path.exists() {
                out.push(Artifact {
                    database: db.name.to_string(),
                    kind: db.kind.to_string(),
                    level: *level,
                    path,
                });
            }
        }
    }
    out
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn generate_keys(k: usize, bins_per_table: usize, dpf_n: u8, seed: u64) -> TrialKeys {
    let dpf = Dpf::with_default_key();
    let mut state = seed;
    let mut targets = Vec::with_capacity(k);
    let mut server0 = Vec::with_capacity(k);
    let mut server1 = Vec::with_capacity(k);

    for _ in 0..k {
        let alpha0 = splitmix64(&mut state) % bins_per_table as u64;
        let alpha1 = splitmix64(&mut state) % bins_per_table as u64;
        let (s0_q0, s1_q0) = dpf.gen(alpha0, dpf_n);
        let (s0_q1, s1_q1) = dpf.gen(alpha1, dpf_n);
        targets.push((alpha0, alpha1));
        server0.push((s0_q0, s0_q1));
        server1.push((s1_q0, s1_q1));
    }

    TrialKeys {
        targets,
        server0,
        server1,
    }
}

fn checksum_bytes(data: &[u8]) -> u64 {
    let mut acc = 0xcbf29ce484222325u64;
    for &b in data {
        acc ^= b as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

fn xor_into(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

fn touch_pages(data: &[u8]) -> u64 {
    let mut sink = 0u8;
    for off in (0..data.len()).step_by(4096) {
        sink ^= data[off];
    }
    std::hint::black_box(sink);
    sink as u64
}

fn process_server(
    level: Level,
    keys: &[(DpfKey, DpfKey)],
    data: &[u8],
    header_size: usize,
    bins_per_table: usize,
    table_byte_size: usize,
) -> ServerRun {
    let started = Instant::now();
    let groups: Vec<(GroupOutput, Duration, Duration, u64)> = (0..keys.len())
        .into_par_iter()
        .map(|group| {
            let start = header_size + group * table_byte_size;
            let table = &data[start..start + table_byte_size];
            match level {
                Level::Index => {
                    let (q0, q1, timing) =
                        process_index_group(&keys[group].0, &keys[group].1, table, bins_per_table);
                    let checksum = checksum_bytes(&q0) ^ checksum_bytes(&q1);
                    (
                        GroupOutput { q0, q1 },
                        timing.dpf_eval,
                        timing.fetch_xor,
                        checksum,
                    )
                }
                Level::Chunk => {
                    let key_refs = [&keys[group].0, &keys[group].1];
                    let (mut results, timing) =
                        process_chunk_group(&key_refs, table, bins_per_table);
                    let q0 = results.remove(0);
                    let q1 = results.remove(0);
                    let checksum = checksum_bytes(&q0) ^ checksum_bytes(&q1);
                    (
                        GroupOutput { q0, q1 },
                        timing.dpf_eval,
                        timing.fetch_xor,
                        checksum,
                    )
                }
            }
        })
        .collect();
    let wall = started.elapsed();

    let mut outputs = Vec::with_capacity(groups.len());
    let mut dpf_sum = Duration::ZERO;
    let mut fetch_sum = Duration::ZERO;
    let mut checksum = 0u64;
    for (out, dpf, fetch, c) in groups {
        outputs.push(out);
        dpf_sum += dpf;
        fetch_sum += fetch;
        checksum ^= c;
    }

    ServerRun {
        outputs,
        wall,
        dpf_sum,
        fetch_sum,
        checksum,
    }
}

fn verify_outputs(
    targets: &[(u64, u64)],
    s0: &ServerRun,
    s1: &ServerRun,
    data: &[u8],
    header_size: usize,
    result_size: usize,
    table_byte_size: usize,
) -> VerifyRun {
    let started = Instant::now();
    let mut failures = 0usize;
    let mut checksum = 0u64;

    for (group, &(alpha0, alpha1)) in targets.iter().enumerate() {
        let table_start = header_size + group * table_byte_size;
        for (query_idx, alpha) in [alpha0, alpha1].into_iter().enumerate() {
            let mut combined = if query_idx == 0 {
                s0.outputs[group].q0.clone()
            } else {
                s0.outputs[group].q1.clone()
            };
            let other = if query_idx == 0 {
                &s1.outputs[group].q0
            } else {
                &s1.outputs[group].q1
            };
            xor_into(&mut combined, other);
            checksum ^= checksum_bytes(&combined);

            let bin_start = table_start + alpha as usize * result_size;
            let expected = &data[bin_start..bin_start + result_size];
            if combined != expected {
                failures += 1;
            }
        }
    }

    VerifyRun {
        wall: started.elapsed(),
        failures,
        checksum,
    }
}

fn dpf_key_wire_size(dpf_n: u8) -> usize {
    (dpf_n as usize + 2) * 16 + (dpf_n as usize).div_ceil(8)
}

fn run_trial(
    artifact: &Artifact,
    mmap: &[u8],
    header_size: usize,
    bins_per_table: usize,
    dpf_n: u8,
    trial_index: usize,
    measured: bool,
) -> Option<String> {
    let params = artifact.level.params();
    let k = params.k;
    let result_size = params.bin_size();
    let table_byte_size = bins_per_table * result_size;
    let key_size = dpf_key_wire_size(dpf_n);
    let upload_per_server = k * params.cuckoo_num_hashes * key_size;
    let download_per_server = k * params.cuckoo_num_hashes * result_size;
    let verified_queries = k * params.cuckoo_num_hashes;

    let seed = 0x4250_4952_4450_4600u64
        ^ (trial_index as u64).wrapping_mul(0x1000_0000_01b3)
        ^ (bins_per_table as u64)
        ^ ((result_size as u64) << 32);

    let keygen_start = Instant::now();
    let keys = generate_keys(k, bins_per_table, dpf_n, seed);
    let keygen = keygen_start.elapsed();

    let s0 = process_server(
        artifact.level,
        &keys.server0,
        mmap,
        header_size,
        bins_per_table,
        table_byte_size,
    );
    let s1 = process_server(
        artifact.level,
        &keys.server1,
        mmap,
        header_size,
        bins_per_table,
        table_byte_size,
    );
    let verify = verify_outputs(
        &keys.targets,
        &s0,
        &s1,
        mmap,
        header_size,
        result_size,
        table_byte_size,
    );

    if !measured {
        return None;
    }

    let server_parallel = s0.wall.max(s1.wall);
    let server_sequential = s0.wall + s1.wall;
    let total_parallel = keygen + server_parallel + verify.wall;
    let total_local_sequential = keygen + server_sequential + verify.wall;
    let checksum = s0.checksum ^ s1.checksum ^ verify.checksum;

    Some(format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{:016x},{}",
        artifact.database,
        artifact.kind,
        artifact.level.as_str(),
        trial_index,
        artifact.path.display(),
        k,
        bins_per_table,
        dpf_n,
        mmap.len(),
        table_byte_size * k,
        result_size,
        key_size,
        upload_per_server,
        download_per_server,
        keygen.as_secs_f64(),
        s0.wall.as_secs_f64(),
        s1.wall.as_secs_f64(),
        server_parallel.as_secs_f64(),
        server_sequential.as_secs_f64(),
        s0.dpf_sum.as_secs_f64(),
        s1.dpf_sum.as_secs_f64(),
        s0.fetch_sum.as_secs_f64(),
        s1.fetch_sum.as_secs_f64(),
        verify.wall.as_secs_f64(),
        total_parallel.as_secs_f64(),
        total_local_sequential.as_secs_f64(),
        verified_queries,
        verify.failures,
        checksum,
        "warm-cache"
    ))
}

fn main() {
    let args = parse_args();
    let artifacts = selected_artifacts(&args);
    if artifacts.is_empty() {
        eprintln!("no matching artifacts found under {}", args.data_root.display());
        std::process::exit(1);
    }

    println!("database,kind,level,trial,path,k,bins_per_table,dpf_n,file_bytes,body_bytes,result_size_bytes,key_size_bytes,upload_per_server_bytes,download_per_server_bytes,keygen_s,server0_s,server1_s,server_parallel_s,server_local_sequential_s,server0_dpf_eval_sum_s,server1_dpf_eval_sum_s,server0_fetch_xor_sum_s,server1_fetch_xor_sum_s,client_verify_s,total_parallel_s,total_local_sequential_s,verified_queries,verification_failures,checksum,cache_state");

    for artifact in artifacts {
        eprintln!(
            "{} {}: {}",
            artifact.database,
            artifact.level.as_str(),
            artifact.path.display()
        );
        let file = File::open(&artifact.path).unwrap_or_else(|e| {
            eprintln!("failed to open {}: {}", artifact.path.display(), e);
            std::process::exit(1);
        });
        let mmap = unsafe { Mmap::map(&file) }.unwrap_or_else(|e| {
            eprintln!("failed to mmap {}: {}", artifact.path.display(), e);
            std::process::exit(1);
        });
        let params = artifact.level.params();
        let header = read_cuckoo_header_with_anchor(&mmap, params).unwrap_or_else(|e| {
            eprintln!("failed to parse {}: {}", artifact.path.display(), e);
            std::process::exit(1);
        });
        let bins_per_table = header.bins_per_table;
        let dpf_n = compute_dpf_n(bins_per_table);
        let body_bytes = params.k * bins_per_table * params.bin_size();
        if mmap.len() < header.header_size + body_bytes {
            eprintln!(
                "{} is too short: got {}, expected at least {}",
                artifact.path.display(),
                mmap.len(),
                header.header_size + body_bytes
            );
            std::process::exit(1);
        }

        let warm_checksum = touch_pages(&mmap);
        eprintln!("  warmup touch checksum {:02x}", warm_checksum);

        for i in 0..args.warmups {
            eprintln!("  warmup {}/{}", i + 1, args.warmups);
            let _ = run_trial(&artifact, &mmap, header.header_size, bins_per_table, dpf_n, i, false);
        }
        for trial in 1..=args.trials {
            eprintln!("  trial {}/{}", trial, args.trials);
            if let Some(row) = run_trial(
                &artifact,
                &mmap,
                header.header_size,
                bins_per_table,
                dpf_n,
                trial,
                true,
            ) {
                println!("{}", row);
            }
        }
    }
}
