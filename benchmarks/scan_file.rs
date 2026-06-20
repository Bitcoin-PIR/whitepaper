use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

fn parse_u64(s: &str, name: &str) -> u64 {
    s.parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid {}: {}", name, s))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: scan_file <path> <offset> <bytes>");
        std::process::exit(2);
    }

    let path = &args[1];
    let offset = parse_u64(&args[2], "offset");
    let target = parse_u64(&args[3], "bytes");

    let mut file = File::open(path).expect("open input");
    file.seek(SeekFrom::Start(offset)).expect("seek input");

    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut total = 0u64;
    let mut checksum = 0u64;
    let start = Instant::now();

    while total < target {
        let want = ((target - total) as usize).min(buf.len());
        let n = file.read(&mut buf[..want]).expect("read input");
        if n == 0 {
            break;
        }

        // Touch one byte per cache line so the compiler cannot remove
        // the read loop. This intentionally measures streaming scan cost,
        // not a full protocol implementation.
        for i in (0..n).step_by(64) {
            checksum = checksum.rotate_left(5)
                ^ (buf[i] as u64)
                ^ total.wrapping_add(i as u64);
        }

        total += n as u64;
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("{},{:.9},{}", total, elapsed, checksum);
}
