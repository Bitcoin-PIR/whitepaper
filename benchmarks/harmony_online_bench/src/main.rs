use harmonypir::params::Params;
use harmonypir::prp::fast::FastPrpWrapper;
use harmonypir::prp::hoang::HoangPrp;
use harmonypir::prp::BatchPrp;
use harmonypir_wasm::{
    compute_rounds, derive_group_key, find_best_t, pad_n_for_t, HarmonyGroup, PRP_FASTPRP,
    PRP_HMR12,
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

struct HintParams {
    k: usize,
    bins_per_table: usize,
    entry_size: usize,
    padded_n: usize,
    t: usize,
    m: usize,
    domain: usize,
    rounds: usize,
    max_queries: usize,
    header_size: usize,
}

struct HintBundle {
    hints: Vec<Vec<u8>>,
    generation: Duration,
    checksum: u64,
}

struct RequestRecord {
    group: usize,
    query: usize,
    bytes: Vec<u8>,
}

struct TrialRun {
    state_setup: Duration,
    request_build: Duration,
    server_response: Duration,
    response_process: Duration,
    request_bytes_total: usize,
    response_bytes_total: usize,
    verified_queries: usize,
    failures: usize,
    checksum: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: harmony-online-bench [--data-root PATH] [--trials N] [--warmups N] [--database NAME] [--level index|chunk|all] [--backend hmr12|fastprp|all]"
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

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn query_for_trial(trial_index: usize, group: usize, hp: &HintParams, backend: Backend) -> usize {
    let mut state = 0x4250_4952_484f_4e00u64
        ^ (trial_index as u64).wrapping_mul(0x1000_0000_01b3)
        ^ (group as u64).wrapping_mul(0x9e37)
        ^ (hp.bins_per_table as u64)
        ^ ((hp.entry_size as u64) << 32)
        ^ ((backend.id() as u64) << 56);
    (splitmix64(&mut state) % hp.bins_per_table as u64) as usize
}

fn generate_hints_for_group(
    backend: Backend,
    prp_key: &[u8; 16],
    protocol_group_id: u32,
    table: &[u8],
    hp: &HintParams,
) -> (Vec<u8>, u64) {
    let derived_key = derive_group_key(prp_key, protocol_group_id);
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
    (hints, checksum)
}

fn generate_hint_bundle(
    artifact: &Artifact,
    backend: Backend,
    mmap: &[u8],
    hp: &HintParams,
    prp_key: &[u8; 16],
) -> HintBundle {
    let table_bytes = hp.bins_per_table * hp.entry_size;
    let started = Instant::now();
    let groups: Vec<(Vec<u8>, u64)> = (0..hp.k)
        .into_par_iter()
        .map(|group| {
            let start = hp.header_size + group * table_bytes;
            let table = &mmap[start..start + table_bytes];
            generate_hints_for_group(
                backend,
                prp_key,
                artifact.level.k_offset() + group as u32,
                table,
                hp,
            )
        })
        .collect();
    let generation = started.elapsed();
    let mut hints = Vec::with_capacity(groups.len());
    let mut checksum = 0u64;
    for (group_hints, group_checksum) in groups {
        checksum ^= group_checksum;
        hints.push(group_hints);
    }
    HintBundle {
        hints,
        generation,
        checksum,
    }
}

fn server_response(request: &[u8], table: &[u8], hp: &HintParams) -> Vec<u8> {
    let count = request.len() / 4;
    let mut response = Vec::with_capacity(count * hp.entry_size);
    for chunk in request.chunks_exact(4) {
        let idx = u32::from_le_bytes(chunk.try_into().expect("u32 request index")) as usize;
        if idx < hp.bins_per_table {
            let start = idx * hp.entry_size;
            response.extend_from_slice(&table[start..start + hp.entry_size]);
        } else {
            response.resize(response.len() + hp.entry_size, 0u8);
        }
    }
    response
}

fn build_groups(
    artifact: &Artifact,
    backend: Backend,
    hp: &HintParams,
    prp_key: &[u8; 16],
    hints: &[Vec<u8>],
) -> Vec<HarmonyGroup> {
    let mut groups = Vec::with_capacity(hp.k);
    for group in 0..hp.k {
        let protocol_group_id = artifact.level.k_offset() + group as u32;
        let mut harmony_group = HarmonyGroup::new_with_backend(
            hp.bins_per_table as u32,
            hp.entry_size as u32,
            hp.t as u32,
            prp_key,
            protocol_group_id,
            backend.id(),
        )
        .unwrap_or_else(|e| panic!("HarmonyGroup init failed: {e:?}"));
        harmony_group
            .load_hints(&hints[group])
            .unwrap_or_else(|e| panic!("load_hints failed: {e:?}"));
        groups.push(harmony_group);
    }
    groups
}

fn run_trial(
    artifact: &Artifact,
    backend: Backend,
    mmap: &[u8],
    hp: &HintParams,
    prp_key: &[u8; 16],
    hints: &[Vec<u8>],
    trial_index: usize,
) -> TrialRun {
    let table_bytes = hp.bins_per_table * hp.entry_size;

    let started = Instant::now();
    let mut groups = build_groups(artifact, backend, hp, prp_key, hints);
    let state_setup = started.elapsed();

    let started = Instant::now();
    let mut requests = Vec::with_capacity(hp.k);
    for (group, harmony_group) in groups.iter_mut().enumerate() {
        let query = query_for_trial(trial_index, group, hp, backend);
        let request = harmony_group
            .build_request(query as u32)
            .unwrap_or_else(|e| panic!("build_request failed: {e:?}"));
        requests.push(RequestRecord {
            group,
            query,
            bytes: request.request(),
        });
    }
    let request_build = started.elapsed();

    let started = Instant::now();
    let responses: Vec<Vec<u8>> = requests
        .par_iter()
        .map(|request| {
            let start = hp.header_size + request.group * table_bytes;
            let table = &mmap[start..start + table_bytes];
            server_response(&request.bytes, table, hp)
        })
        .collect();
    let server_response_time = started.elapsed();

    let started = Instant::now();
    let mut failures = 0usize;
    let mut checksum = 0u64;
    for (i, request) in requests.iter().enumerate() {
        let answer = groups[request.group]
            .process_response(&responses[i])
            .unwrap_or_else(|e| panic!("process_response failed: {e:?}"));
        let table_start = hp.header_size + request.group * table_bytes;
        let entry_start = table_start + request.query * hp.entry_size;
        let expected = &mmap[entry_start..entry_start + hp.entry_size];
        checksum ^= checksum_bytes(&answer);
        if answer != expected {
            failures += 1;
        }
    }
    let response_process = started.elapsed();

    TrialRun {
        state_setup,
        request_build,
        server_response: server_response_time,
        response_process,
        request_bytes_total: requests.iter().map(|r| r.bytes.len()).sum(),
        response_bytes_total: responses.iter().map(Vec::len).sum(),
        verified_queries: hp.k,
        failures,
        checksum,
    }
}

fn main() {
    let args = parse_args();
    let artifacts = selected_artifacts(&args);
    let backends = selected_backends(&args);
    if artifacts.is_empty() {
        eprintln!(
            "no matching artifacts found under {}",
            args.data_root.display()
        );
        std::process::exit(1);
    }

    println!("database,kind,level,backend,trial,path,k,bins_per_table,padded_n,t,m,max_queries,entry_size_bytes,hints_bytes_total,hints_bytes_per_group,request_bytes_total,request_bytes_per_group,response_bytes_total,response_bytes_per_group,state_setup_s,request_build_s,server_response_s,response_process_s,total_online_s,verified_queries,failures,artifact_bytes,hint_generation_s,checksum,cache_state");

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
            let harmony_params =
                Params::new(padded, entry_size, t).expect("valid HarmonyPIR params");
            HintParams {
                k: params.k,
                bins_per_table,
                entry_size,
                padded_n: padded,
                t,
                m: harmony_params.m,
                domain: 2 * padded,
                rounds: compute_rounds(padded_n),
                max_queries: harmony_params.max_queries,
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
            let hints = generate_hint_bundle(&artifact, *backend, &mmap, &hp, &prp_key);
            let hints_bytes_total: usize = hints.hints.iter().map(Vec::len).sum();
            let hints_bytes_per_group = hints_bytes_total / hp.k;
            for i in 0..args.warmups {
                eprintln!("    warmup {}/{}", i + 1, args.warmups);
                let _ = run_trial(&artifact, *backend, &mmap, &hp, &prp_key, &hints.hints, i);
            }
            for trial in 1..=args.trials {
                eprintln!("    trial {}/{}", trial, args.trials);
                let run = run_trial(
                    &artifact,
                    *backend,
                    &mmap,
                    &hp,
                    &prp_key,
                    &hints.hints,
                    trial,
                );
                let total_online = run.request_build + run.server_response + run.response_process;
                let checksum = hints.checksum ^ run.checksum ^ (backend.id() as u64);
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{:.9},{:016x},{}",
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
                    hp.max_queries,
                    hp.entry_size,
                    hints_bytes_total,
                    hints_bytes_per_group,
                    run.request_bytes_total,
                    run.request_bytes_total / hp.k,
                    run.response_bytes_total,
                    run.response_bytes_total / hp.k,
                    run.state_setup.as_secs_f64(),
                    run.request_build.as_secs_f64(),
                    run.server_response.as_secs_f64(),
                    run.response_process.as_secs_f64(),
                    total_online.as_secs_f64(),
                    run.verified_queries,
                    run.failures,
                    mmap.len(),
                    hints.generation.as_secs_f64(),
                    checksum,
                    "warm-cache"
                );
            }
        }
    }
}
