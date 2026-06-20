use libdpf::{Dpf, DpfKey};
use memmap2::Mmap;
use pir_core::cuckoo::read_cuckoo_header_with_anchor;
use pir_core::merkle::{compute_bin_leaf_hash, compute_parent_n, Hash256, ZERO_HASH};
use pir_core::params::{compute_dpf_n, TableParams, CHUNK_PARAMS, INDEX_PARAMS};
use pir_runtime_core::eval::{process_merkle_sibling_group, xor_into};
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MERKLE_ARITY: usize = 8;
const MERKLE_ROW_SIZE: usize = MERKLE_ARITY * 32;

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

    fn data_file_name(self) -> &'static str {
        match self {
            Level::Index => "batch_pir_cuckoo.bin",
            Level::Chunk => "chunk_pir_cuckoo.bin",
        }
    }

    fn sibling_file_name(self, level: usize) -> String {
        match self {
            Level::Index => format!("merkle_bucket_index_sib_L{level}.bin"),
            Level::Chunk => format!("merkle_bucket_chunk_sib_L{level}.bin"),
        }
    }

    fn sibling_magic(self, level: usize) -> u64 {
        match self {
            Level::Index => 0xBA7C_B000_0000_0000u64 | ((level as u64) << 16),
            Level::Chunk => 0xBA7C_B000_0000_0000u64 | (1u64 << 40) | ((level as u64) << 16),
        }
    }

    fn tree_offset(self) -> usize {
        match self {
            Level::Index => 0,
            Level::Chunk => INDEX_PARAMS.k,
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
    artifact_dir: PathBuf,
    data_path: PathBuf,
}

struct DataTable {
    mmap: Mmap,
    header_size: usize,
    bins_per_table: usize,
    result_size: usize,
    table_byte_size: usize,
    file_bytes: usize,
}

struct SiblingTable {
    mmap: Mmap,
    header_size: usize,
    bins_per_table: usize,
    table_byte_size: usize,
    file_bytes: usize,
}

#[derive(Clone)]
struct TreeTop {
    cache_from: usize,
    arity: usize,
    levels: Vec<Vec<Hash256>>,
}

struct TrialKeys {
    targets: Vec<Vec<usize>>,
    server0: Vec<Vec<DpfKey>>,
    server1: Vec<Vec<DpfKey>>,
    keygen: Duration,
    upload_per_server: usize,
}

struct ServerRun {
    outputs: Vec<Vec<Vec<u8>>>,
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

struct TrialRun {
    keygen: Duration,
    server0: Duration,
    server1: Duration,
    server_parallel: Duration,
    server0_dpf_sum: Duration,
    server1_dpf_sum: Duration,
    server0_fetch_sum: Duration,
    server1_fetch_sum: Duration,
    client_verify: Duration,
    verified_groups: usize,
    failures: usize,
    upload_per_server: usize,
    download_per_server: usize,
    checksum: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: merkle-round-bench [--data-root PATH] [--trials N] [--warmups N] [--database NAME] [--level index|chunk|all]"
    );
    std::process::exit(2);
}

fn next_value(iter: &mut impl Iterator<Item = String>) -> String {
    iter.next().unwrap_or_else(|| usage())
}

fn parse_args() -> Args {
    let mut args = Args {
        data_root: PathBuf::from("/Volumes/Bitcoin/data"),
        trials: 30,
        warmups: 1,
        database: None,
        level: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--data-root" => args.data_root = PathBuf::from(next_value(&mut iter)),
            "--trials" => args.trials = next_value(&mut iter).parse().unwrap_or_else(|_| usage()),
            "--warmups" => args.warmups = next_value(&mut iter).parse().unwrap_or_else(|_| usage()),
            "--database" => args.database = Some(next_value(&mut iter)),
            "--level" => {
                args.level = match next_value(&mut iter).as_str() {
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
        if !db.artifact_dir.join("merkle_bucket_tree_tops.bin").exists() {
            continue;
        }
        for level in &levels {
            let data_path = db.artifact_dir.join(level.data_file_name());
            let first_sib = db.artifact_dir.join(level.sibling_file_name(0));
            if data_path.exists() && first_sib.exists() {
                out.push(Artifact {
                    database: db.name.to_string(),
                    kind: db.kind.to_string(),
                    level: *level,
                    artifact_dir: db.artifact_dir.clone(),
                    data_path,
                });
            }
        }
    }
    out
}

fn sibling_params(level: Level, sibling_level: usize) -> TableParams {
    let base = level.params();
    TableParams {
        k: base.k,
        num_hashes: 0,
        master_seed: 0,
        slots_per_bin: 1,
        cuckoo_num_hashes: 0,
        slot_size: MERKLE_ROW_SIZE,
        dpf_n: 0,
        magic: level.sibling_magic(sibling_level),
        header_size: 32,
        has_tag_seed: false,
    }
}

fn mmap_file(path: &Path) -> Mmap {
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("failed to open {}: {}", path.display(), e);
        std::process::exit(1);
    });
    unsafe { Mmap::map(&file) }.unwrap_or_else(|e| {
        eprintln!("failed to mmap {}: {}", path.display(), e);
        std::process::exit(1);
    })
}

fn load_data_table(path: &Path, params: &TableParams) -> DataTable {
    let mmap = mmap_file(path);
    let header = read_cuckoo_header_with_anchor(&mmap, params).unwrap_or_else(|e| {
        eprintln!("failed to parse {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let result_size = params.bin_size();
    let table_byte_size = header.bins_per_table * result_size;
    let file_bytes = mmap.len();
    if file_bytes < header.header_size + params.k * table_byte_size {
        eprintln!(
            "{} is too short: got {}, expected at least {}",
            path.display(),
            file_bytes,
            header.header_size + params.k * table_byte_size
        );
        std::process::exit(1);
    }
    DataTable {
        mmap,
        header_size: header.header_size,
        bins_per_table: header.bins_per_table,
        result_size,
        table_byte_size,
        file_bytes,
    }
}

fn load_sibling_tables(artifact: &Artifact) -> Vec<SiblingTable> {
    let mut out = Vec::new();
    for level_idx in 0.. {
        let path = artifact
            .artifact_dir
            .join(artifact.level.sibling_file_name(level_idx));
        if !path.exists() {
            break;
        }
        let params = sibling_params(artifact.level, level_idx);
        let mmap = mmap_file(&path);
        let header = read_cuckoo_header_with_anchor(&mmap, &params).unwrap_or_else(|e| {
            eprintln!("failed to parse {}: {}", path.display(), e);
            std::process::exit(1);
        });
        let table_byte_size = header.bins_per_table * params.bin_size();
        let file_bytes = mmap.len();
        if file_bytes < header.header_size + params.k * table_byte_size {
            eprintln!(
                "{} is too short: got {}, expected at least {}",
                path.display(),
                file_bytes,
                header.header_size + params.k * table_byte_size
            );
            std::process::exit(1);
        }
        out.push(SiblingTable {
            mmap,
            header_size: header.header_size,
            bins_per_table: header.bins_per_table,
            table_byte_size,
            file_bytes,
        });
    }
    out
}

fn parse_tree_tops(path: &Path) -> Vec<TreeTop> {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });
    if data.len() < 4 {
        eprintln!("{} too short for tree-top header", path.display());
        std::process::exit(1);
    }
    let mut off = 0usize;
    let num_trees = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let mut trees = Vec::with_capacity(num_trees);
    for _ in 0..num_trees {
        if off + 8 > data.len() {
            eprintln!("truncated tree-top record in {}", path.display());
            std::process::exit(1);
        }
        let cache_from = data[off] as usize;
        off += 1;
        let _total_nodes = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let arity = u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        let num_levels = data[off] as usize;
        off += 1;
        let mut levels = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            if off + 4 > data.len() {
                eprintln!("truncated tree-top level in {}", path.display());
                std::process::exit(1);
            }
            let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + n * 32 > data.len() {
                eprintln!("truncated tree-top hashes in {}", path.display());
                std::process::exit(1);
            }
            let mut level = Vec::with_capacity(n);
            for _ in 0..n {
                let mut h = [0u8; 32];
                h.copy_from_slice(&data[off..off + 32]);
                off += 32;
                level.push(h);
            }
            levels.push(level);
        }
        trees.push(TreeTop {
            cache_from,
            arity,
            levels,
        });
    }
    if off != data.len() {
        eprintln!(
            "{} has {} trailing bytes after tree-top parse",
            path.display(),
            data.len() - off
        );
        std::process::exit(1);
    }
    trees
}

fn read_roots(path: &Path) -> Vec<Hash256> {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });
    if data.len() % 32 != 0 {
        eprintln!("{} length is not a hash multiple", path.display());
        std::process::exit(1);
    }
    data.chunks_exact(32)
        .map(|chunk| {
            let mut h = [0u8; 32];
            h.copy_from_slice(chunk);
            h
        })
        .collect()
}

fn checksum_bytes(data: &[u8]) -> u64 {
    let mut acc = 0xcbf29ce484222325u64;
    for &b in data {
        acc ^= b as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

fn touch_pages(data: &[u8]) -> u64 {
    let mut sink = 0u8;
    for off in (0..data.len()).step_by(4096) {
        sink ^= data[off];
    }
    std::hint::black_box(sink);
    sink as u64
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn target_bin(trial_index: usize, group: usize, bins_per_table: usize, level: Level) -> usize {
    let mut state = 0x4250_4952_4d45_5200u64
        ^ (trial_index as u64).wrapping_mul(0x1000_0000_01b3)
        ^ (group as u64).wrapping_mul(0x9e37)
        ^ (bins_per_table as u64)
        ^ ((level as u8 as u64) << 56);
    (splitmix64(&mut state) % bins_per_table as u64) as usize
}

fn dpf_key_wire_size(dpf_n: u8) -> usize {
    (dpf_n as usize + 2) * 16 + (dpf_n as usize).div_ceil(8)
}

fn generate_keys(
    level: Level,
    trial_index: usize,
    data: &DataTable,
    siblings: &[SiblingTable],
) -> TrialKeys {
    let dpf = Dpf::with_default_key();
    let k = level.params().k;
    let mut targets = Vec::with_capacity(siblings.len());
    let mut server0 = Vec::with_capacity(siblings.len());
    let mut server1 = Vec::with_capacity(siblings.len());
    let mut upload_per_server = 0usize;
    let started = Instant::now();

    for (sib_level, table) in siblings.iter().enumerate() {
        let dpf_n = compute_dpf_n(table.bins_per_table);
        upload_per_server += k * dpf_key_wire_size(dpf_n);
        let mut level_targets = Vec::with_capacity(k);
        let mut level_s0 = Vec::with_capacity(k);
        let mut level_s1 = Vec::with_capacity(k);
        for group in 0..k {
            let mut node_idx = target_bin(trial_index, group, data.bins_per_table, level);
            for _ in 0..sib_level {
                node_idx /= MERKLE_ARITY;
            }
            let alpha = node_idx / MERKLE_ARITY;
            let (s0, s1) = dpf.gen(alpha as u64, dpf_n);
            level_targets.push(alpha);
            level_s0.push(s0);
            level_s1.push(s1);
        }
        targets.push(level_targets);
        server0.push(level_s0);
        server1.push(level_s1);
    }

    TrialKeys {
        targets,
        server0,
        server1,
        keygen: started.elapsed(),
        upload_per_server,
    }
}

fn process_server(keys: &[Vec<DpfKey>], siblings: &[SiblingTable]) -> ServerRun {
    let started = Instant::now();
    let mut outputs = Vec::with_capacity(siblings.len());
    let mut dpf_sum = Duration::ZERO;
    let mut fetch_sum = Duration::ZERO;
    let mut checksum = 0u64;

    for (level_idx, table) in siblings.iter().enumerate() {
        let level_started = Instant::now();
        let groups: Vec<(Vec<u8>, Duration, Duration, u64)> = (0..keys[level_idx].len())
            .into_par_iter()
            .map(|group| {
                let start = table.header_size + group * table.table_byte_size;
                let table_bytes = &table.mmap[start..start + table.table_byte_size];
                let key_refs = [&keys[level_idx][group]];
                let (mut result, timing) = process_merkle_sibling_group(
                    &key_refs,
                    table_bytes,
                    table.bins_per_table,
                    MERKLE_ROW_SIZE,
                );
                let row = result.remove(0);
                let checksum = checksum_bytes(&row);
                (row, timing.dpf_eval, timing.fetch_xor, checksum)
            })
            .collect();
        let _level_wall = level_started.elapsed();
        let mut level_outputs = Vec::with_capacity(groups.len());
        for (row, dpf, fetch, c) in groups {
            dpf_sum += dpf;
            fetch_sum += fetch;
            checksum ^= c;
            level_outputs.push(row);
        }
        outputs.push(level_outputs);
    }

    ServerRun {
        outputs,
        wall: started.elapsed(),
        dpf_sum,
        fetch_sum,
        checksum,
    }
}

fn verify_trial(
    artifact: &Artifact,
    data: &DataTable,
    siblings: &[SiblingTable],
    tree_tops: &[TreeTop],
    roots: &[Hash256],
    keys: &TrialKeys,
    s0: &ServerRun,
    s1: &ServerRun,
    trial_index: usize,
) -> VerifyRun {
    let started = Instant::now();
    let k = artifact.level.params().k;
    let tree_offset = artifact.level.tree_offset();
    let mut current_hashes = Vec::with_capacity(k);
    let mut node_idxs = Vec::with_capacity(k);
    let mut failures = 0usize;
    let mut checksum = 0u64;

    for group in 0..k {
        let bin = target_bin(trial_index, group, data.bins_per_table, artifact.level);
        let table_start = data.header_size + group * data.table_byte_size;
        let bin_start = table_start + bin * data.result_size;
        let leaf = compute_bin_leaf_hash(
            bin as u32,
            &data.mmap[bin_start..bin_start + data.result_size],
        );
        current_hashes.push(leaf);
        node_idxs.push(bin);
        checksum ^= checksum_bytes(&leaf);
    }

    for level_idx in 0..siblings.len() {
        for group in 0..k {
            let mut row = s0.outputs[level_idx][group].clone();
            xor_into(&mut row, &s1.outputs[level_idx][group]);
            checksum ^= checksum_bytes(&row);

            let child_pos = node_idxs[group] % MERKLE_ARITY;
            let expected_target = node_idxs[group] / MERKLE_ARITY;
            if keys.targets[level_idx][group] != expected_target {
                failures += 1;
            }
            let mut children = Vec::with_capacity(MERKLE_ARITY);
            for c in 0..MERKLE_ARITY {
                if c == child_pos {
                    children.push(current_hashes[group]);
                } else {
                    let off = c * 32;
                    let mut h = [0u8; 32];
                    h.copy_from_slice(&row[off..off + 32]);
                    children.push(h);
                }
            }
            current_hashes[group] = compute_parent_n(&children);
            node_idxs[group] = expected_target;
        }
    }

    for group in 0..k {
        let tree_idx = tree_offset + group;
        let top = &tree_tops[tree_idx];
        if top.arity != MERKLE_ARITY || top.cache_from != siblings.len() {
            failures += 1;
            continue;
        }
        for level_nodes in top.levels.iter().take(top.levels.len().saturating_sub(1)) {
            let child_pos = node_idxs[group] % MERKLE_ARITY;
            let parent_start = (node_idxs[group] / MERKLE_ARITY) * MERKLE_ARITY;
            let mut children = Vec::with_capacity(MERKLE_ARITY);
            for c in 0..MERKLE_ARITY {
                let idx = parent_start + c;
                if c == child_pos {
                    children.push(current_hashes[group]);
                } else if idx < level_nodes.len() {
                    children.push(level_nodes[idx]);
                } else {
                    children.push(ZERO_HASH);
                }
            }
            current_hashes[group] = compute_parent_n(&children);
            node_idxs[group] /= MERKLE_ARITY;
        }
        let expected_root = top
            .levels
            .last()
            .and_then(|level| level.first().copied())
            .unwrap_or(ZERO_HASH);
        if roots.get(tree_idx).copied() != Some(expected_root) {
            failures += 1;
        }
        if current_hashes[group] != expected_root {
            failures += 1;
        }
        checksum ^= checksum_bytes(&current_hashes[group]);
    }

    VerifyRun {
        wall: started.elapsed(),
        failures,
        checksum,
    }
}

fn run_trial(
    artifact: &Artifact,
    data: &DataTable,
    siblings: &[SiblingTable],
    tree_tops: &[TreeTop],
    roots: &[Hash256],
    trial_index: usize,
    measured: bool,
) -> Option<TrialRun> {
    let keys = generate_keys(artifact.level, trial_index, data, siblings);
    let s0 = process_server(&keys.server0, siblings);
    let s1 = process_server(&keys.server1, siblings);
    let verify = verify_trial(
        artifact,
        data,
        siblings,
        tree_tops,
        roots,
        &keys,
        &s0,
        &s1,
        trial_index,
    );
    if !measured {
        return None;
    }

    let server_parallel = s0.wall.max(s1.wall);
    let download_per_server = siblings.len() * artifact.level.params().k * MERKLE_ROW_SIZE;
    Some(TrialRun {
        keygen: keys.keygen,
        server0: s0.wall,
        server1: s1.wall,
        server_parallel,
        server0_dpf_sum: s0.dpf_sum,
        server1_dpf_sum: s1.dpf_sum,
        server0_fetch_sum: s0.fetch_sum,
        server1_fetch_sum: s1.fetch_sum,
        client_verify: verify.wall,
        verified_groups: artifact.level.params().k,
        failures: verify.failures,
        upload_per_server: keys.upload_per_server,
        download_per_server,
        checksum: s0.checksum ^ s1.checksum ^ verify.checksum,
    })
}

fn main() {
    let args = parse_args();
    let artifacts = selected_artifacts(&args);
    if artifacts.is_empty() {
        eprintln!(
            "no matching Merkle artifacts found under {}",
            args.data_root.display()
        );
        std::process::exit(1);
    }

    println!("database,kind,level,trial,path,k,bins_per_table,arity,sibling_levels,tree_top_bytes,sibling_bytes_total,file_bytes,upload_per_server_bytes,download_per_server_bytes,keygen_s,server0_s,server1_s,server_parallel_s,server0_dpf_eval_sum_s,server1_dpf_eval_sum_s,server0_fetch_xor_sum_s,server1_fetch_xor_sum_s,client_verify_s,total_parallel_s,verified_groups,verification_failures,checksum,cache_state");

    for artifact in artifacts {
        eprintln!(
            "{} {}: {}",
            artifact.database,
            artifact.level.as_str(),
            artifact.artifact_dir.display()
        );
        let data = load_data_table(&artifact.data_path, artifact.level.params());
        let siblings = load_sibling_tables(&artifact);
        let tree_top_path = artifact.artifact_dir.join("merkle_bucket_tree_tops.bin");
        let roots_path = artifact.artifact_dir.join("merkle_bucket_roots.bin");
        let tree_tops = parse_tree_tops(&tree_top_path);
        let roots = read_roots(&roots_path);
        let expected_trees = INDEX_PARAMS.k + CHUNK_PARAMS.k;
        if tree_tops.len() != expected_trees || roots.len() != expected_trees {
            eprintln!(
                "{} has tree_tops={} roots={}, expected {}",
                artifact.artifact_dir.display(),
                tree_tops.len(),
                roots.len(),
                expected_trees
            );
            std::process::exit(1);
        }
        for group in 0..artifact.level.params().k {
            let tree_idx = artifact.level.tree_offset() + group;
            if tree_tops[tree_idx].cache_from != siblings.len() {
                eprintln!(
                    "{} {} group {} has cache_from {}, but {} sibling files are present",
                    artifact.database,
                    artifact.level.as_str(),
                    group,
                    tree_tops[tree_idx].cache_from,
                    siblings.len()
                );
                std::process::exit(1);
            }
        }
        let tree_top_bytes = std::fs::metadata(&tree_top_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let sibling_bytes_total: usize = siblings.iter().map(|s| s.file_bytes).sum();
        let file_bytes = data.file_bytes + sibling_bytes_total + tree_top_bytes;
        let mut warm_checksum = touch_pages(&data.mmap);
        for s in &siblings {
            warm_checksum ^= touch_pages(&s.mmap);
        }
        eprintln!("  warmup touch checksum {:02x}", warm_checksum);
        for i in 0..args.warmups {
            eprintln!("  warmup {}/{}", i + 1, args.warmups);
            let _ = run_trial(&artifact, &data, &siblings, &tree_tops, &roots, i, false);
        }
        for trial in 1..=args.trials {
            eprintln!("  trial {}/{}", trial, args.trials);
            let run = run_trial(&artifact, &data, &siblings, &tree_tops, &roots, trial, true)
                .expect("measured run");
            let total_parallel = run.keygen + run.server_parallel + run.client_verify;
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{:016x},{}",
                artifact.database,
                artifact.kind,
                artifact.level.as_str(),
                trial,
                artifact.artifact_dir.display(),
                artifact.level.params().k,
                data.bins_per_table,
                MERKLE_ARITY,
                siblings.len(),
                tree_top_bytes,
                sibling_bytes_total,
                file_bytes,
                run.upload_per_server,
                run.download_per_server,
                run.keygen.as_secs_f64(),
                run.server0.as_secs_f64(),
                run.server1.as_secs_f64(),
                run.server_parallel.as_secs_f64(),
                run.server0_dpf_sum.as_secs_f64(),
                run.server1_dpf_sum.as_secs_f64(),
                run.server0_fetch_sum.as_secs_f64(),
                run.server1_fetch_sum.as_secs_f64(),
                run.client_verify.as_secs_f64(),
                total_parallel.as_secs_f64(),
                run.verified_groups,
                run.failures,
                run.checksum,
                "warm-cache"
            );
        }
    }
}
