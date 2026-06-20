use harmonypir::params::Params;
use harmonypir::prp::fast::FastPrpWrapper;
use harmonypir::prp::hoang::HoangPrp;
use harmonypir::prp::BatchPrp;
use harmonypir_wasm::{
    compute_rounds, derive_group_key, find_best_t, pad_n_for_t, PRP_FASTPRP, PRP_HMR12,
};
use memmap2::Mmap;
use pir_core::cuckoo::read_cuckoo_header_with_anchor;
use pir_core::params::{TableParams, CHUNK_PARAMS, INDEX_PARAMS};
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

    fn k_offset(self) -> u32 {
        match self {
            Level::Index => 0,
            Level::Chunk => INDEX_PARAMS.k as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Hmr12,
    FastPrp,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Backend::Hmr12 => "hmr12",
            Backend::FastPrp => "fastprp",
        }
    }

    fn id(self) -> u8 {
        match self {
            Backend::Hmr12 => PRP_HMR12,
            Backend::FastPrp => PRP_FASTPRP,
        }
    }
}

struct Args {
    data_root: PathBuf,
    trials: usize,
    warmups: usize,
    database: Option<String>,
    level: Option<Level>,
    backend: Option<Backend>,
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

struct TrialRun {
    wall: Duration,
    hints_bytes: usize,
    checksum: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: harmony-hint-bench [--data-root PATH] [--trials N] [--warmups N] [--database NAME] [--level index|chunk|all] [--backend hmr12|fastprp|all]"
    );
    std::process::exit(2);
}

fn next_value(iter: &mut impl Iterator<Item = String>) -> String {
    iter.next().unwrap_or_else(|| usage())
}

fn parse_args() -> Args {
    let mut args = Args {
        data_root: PathBuf::from("/Volumes/Bitcoin/data"),
        trials: 3,
        warmups: 1,
        database: None,
        level: None,
        backend: None,
    };

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--data-root" => args.data_root = PathBuf::from(next_value(&mut iter)),
            "--trials" => {
                args.trials = next_value(&mut iter).parse().unwrap_or_else(|_| usage());
            }
            "--warmups" => {
                args.warmups = next_value(&mut iter).parse().unwrap_or_else(|_| usage());
            }
            "--database" => args.database = Some(next_value(&mut iter)),
            "--level" => {
                args.level = match next_value(&mut iter).as_str() {
                    "index" => Some(Level::Index),
                    "chunk" => Some(Level::Chunk),
                    "all" => None,
                    _ => usage(),
                }
            }
            "--backend" => {
                args.backend = match next_value(&mut iter).as_str() {
                    "hmr12" => Some(Backend::Hmr12),
                    "fastprp" => Some(Backend::FastPrp),
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

fn selected_backends(args: &Args) -> Vec<Backend> {
    match args.backend {
        Some(backend) => vec![backend],
        None => vec![Backend::Hmr12, Backend::FastPrp],
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

fn touch_pages(data: &[u8]) -> u64 {
    let mut sink = 0u8;
    for off in (0..data.len()).step_by(4096) {
        sink ^= data[off];
    }
    std::hint::black_box(sink);
    sink as u64
}

fn xor_into(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

struct HintParams {
    k: usize,
    bins_per_table: usize,
    entry_size: usize,
    padded_n: usize,
    t: usize,
    m: usize,
    domain: usize,
    rounds: usize,
    header_size: usize,
}

fn generate_hints_for_group(
    backend: Backend,
    prp_key: &[u8; 16],
    k_offset: u32,
    group_id: usize,
    table: &[u8],
    hp: &HintParams,
) -> (usize, u64) {
    let derived_key = derive_group_key(prp_key, k_offset + group_id as u32);
    let cell_of = match backend {
        Backend::FastPrp => {
            let prp = FastPrpWrapper::new(&derived_key, hp.domain);
            prp.batch_forward()
        }
        Backend::Hmr12 => {
            let prp = HoangPrp::new(hp.domain, hp.rounds, &derived_key);
            prp.batch_forward()
        }
    };

    let mut hints = vec![0u8; hp.m * hp.entry_size];
    for k in 0..hp.padded_n {
        let segment = cell_of[k] / hp.t;
        if k < hp.bins_per_table {
            let entry_start = k * hp.entry_size;
            let entry = &table[entry_start..entry_start + hp.entry_size];
            let hint_start = segment * hp.entry_size;
            xor_into(&mut hints[hint_start..hint_start + hp.entry_size], entry);
        }
    }

    let checksum = checksum_bytes(&hints);
    (hints.len(), checksum)
}

fn run_trial(
    artifact: &Artifact,
    backend: Backend,
    mmap: &[u8],
    hp: &HintParams,
    prp_key: &[u8; 16],
) -> TrialRun {
    let table_bytes = hp.bins_per_table * hp.entry_size;
    let started = Instant::now();
    let groups: Vec<(usize, u64)> = (0..hp.k)
        .into_par_iter()
        .map(|group_id| {
            let start = hp.header_size + group_id * table_bytes;
            let table = &mmap[start..start + table_bytes];
            generate_hints_for_group(
                backend,
                prp_key,
                artifact.level.k_offset(),
                group_id,
                table,
                hp,
            )
        })
        .collect();
    let wall = started.elapsed();
    let hints_bytes = groups.iter().map(|(n, _)| *n).sum();
    let checksum = groups.iter().fold(0u64, |acc, (_, c)| acc ^ c);
    TrialRun {
        wall,
        hints_bytes,
        checksum,
    }
}

fn main() {
    let args = parse_args();
    let artifacts = selected_artifacts(&args);
    let backends = selected_backends(&args);
    if artifacts.is_empty() {
        eprintln!("no matching artifacts found under {}", args.data_root.display());
        std::process::exit(1);
    }

    println!("database,kind,level,backend,trial,path,k,bins_per_table,padded_n,t,m,max_queries,entry_size_bytes,hints_bytes_total,hints_bytes_per_group,request_bytes_per_query,response_bytes_per_query,artifact_bytes,hint_generation_s,checksum,cache_state");

    let prp_key = [0x42u8; 16];
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
        let entry_size = params.bin_size();
        let t_raw = find_best_t(bins_per_table as u32);
        let (padded_n, t_val) = pad_n_for_t(bins_per_table as u32, t_raw);
        let hp = {
            let padded = padded_n as usize;
            let t = t_val as usize;
            let harmony_params = Params::new(padded, entry_size, t).expect("valid HarmonyPIR params");
            HintParams {
                k: params.k,
                bins_per_table,
                entry_size,
                padded_n: padded,
                t,
                m: harmony_params.m,
                domain: 2 * padded,
                rounds: compute_rounds(padded_n),
                header_size: header.header_size,
            }
        };
        let table_bytes_total = hp.k * hp.bins_per_table * hp.entry_size;
        if mmap.len() < hp.header_size + table_bytes_total {
            eprintln!(
                "{} is too short: got {}, expected at least {}",
                artifact.path.display(),
                mmap.len(),
                hp.header_size + table_bytes_total
            );
            std::process::exit(1);
        }

        let warm_checksum = touch_pages(&mmap);
        eprintln!("  warmup touch checksum {:02x}", warm_checksum);

        for backend in &backends {
            eprintln!("  backend {}", backend.as_str());
            for i in 0..args.warmups {
                eprintln!("    warmup {}/{}", i + 1, args.warmups);
                let _ = run_trial(&artifact, *backend, &mmap, &hp, &prp_key);
            }
            for trial in 1..=args.trials {
                eprintln!("    trial {}/{}", trial, args.trials);
                let run = run_trial(&artifact, *backend, &mmap, &hp, &prp_key);
                let hints_per_group = run.hints_bytes / hp.k;
                let request_bytes = (hp.t - 1) * 4;
                let response_bytes = (hp.t - 1) * hp.entry_size;
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:016x},{}",
                    artifact.database,
                    artifact.kind,
                    artifact.level.as_str(),
                    backend.as_str(),
                    trial,
                    artifact.path.display(),
                    hp.k,
                    hp.bins_per_table,
                    hp.padded_n,
                    hp.t,
                    hp.m,
                    hp.padded_n / hp.t,
                    hp.entry_size,
                    run.hints_bytes,
                    hints_per_group,
                    request_bytes,
                    response_bytes,
                    mmap.len(),
                    run.wall.as_secs_f64(),
                    run.checksum ^ (backend.id() as u64),
                    "warm-cache"
                );
            }
        }
    }
}
