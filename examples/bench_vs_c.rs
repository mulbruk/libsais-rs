use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use libsais_rs::{libsais, SaSint};

unsafe extern "C" {
    fn probe_public_libsais(t: *const u8, sa: *mut SaSint, n: SaSint, fs: SaSint) -> SaSint;
}

struct Workload {
    name: String,
    bytes: Vec<u8>,
}

/// A description of a workload that has not yet been materialized into memory.
///
/// Materialization is deferred so the benchmark holds at most one workload's
/// bytes at a time — important when the corpora directory contains tens or
/// hundreds of megabytes of input across many files.
enum WorkloadSpec {
    File { display_name: String, path: PathBuf },
    Generated { name: String, len: usize },
}

impl WorkloadSpec {
    fn from_path(path: PathBuf) -> Self {
        let display_name = path.display().to_string();
        Self::File { display_name, path }
    }

    fn generated(name: &str, len: usize) -> Self {
        Self::Generated {
            name: name.to_string(),
            len,
        }
    }

    fn materialize(&self) -> Workload {
        match self {
            Self::File { display_name, path } => {
                let bytes = fs::read(path)
                    .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
                Workload {
                    name: display_name.clone(),
                    bytes,
                }
            }
            Self::Generated { name, len } => generated_workload(name, *len),
        }
    }
}

fn generated_workload(name: &str, len: usize) -> Workload {
    let mut state: u32 = 0x243f_6a88;
    let mut bytes = Vec::with_capacity(len);

    for i in 0..len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut value = ((state >> 16) & 0xff) as u8;

        if i % 31 < 12 {
            value = ((i / 31) % 23) as u8;
        }
        if i % 97 >= 64 {
            value = bytes[i - 64];
        }

        bytes.push(value);
    }

    Workload {
        name: name.to_string(),
        bytes,
    }
}

/// Recursively collect benchmark inputs from `root`, sorted alphabetically
/// within each directory and walked depth-first as entries are encountered.
///
/// Hidden entries (names starting with `.`) and empty files are skipped.
fn collect_corpora_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk_dir(root, &mut out)?;
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }

        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            walk_dir(&path, out)?;
        } else if file_type.is_file() && entry.metadata()?.len() > 0 {
            out.push(path);
        }
    }

    Ok(())
}

fn iterations_for_len(len: usize) -> usize {
    if len <= 32 * 1024 {
        200
    } else if len <= 512 * 1024 {
        40
    } else if len <= 2 * 1024 * 1024 {
        10
    } else {
        5
    }
}

fn bench_one<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

fn verify_outputs(bytes: &[u8]) {
    let n = SaSint::try_from(bytes.len()).expect("input length must fit SaSint");
    let mut sa_rust = vec![0; bytes.len()];
    let mut sa_c = vec![0; bytes.len()];

    let rust_result = libsais(bytes, &mut sa_rust, 0, None);
    let c_result = unsafe { probe_public_libsais(bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };

    assert_eq!(rust_result, c_result, "result mismatch for input length {}", bytes.len());
    assert_eq!(sa_rust, sa_c, "suffix array mismatch for input length {}", bytes.len());
}

fn benchmark_workload(workload: &Workload) {
    let n = SaSint::try_from(workload.bytes.len()).expect("input length must fit SaSint");
    let iterations = iterations_for_len(workload.bytes.len());

    verify_outputs(&workload.bytes);

    let mut sa_rust = vec![0; workload.bytes.len()];
    let rust_total = bench_one(iterations, || {
        let result = libsais(&workload.bytes, &mut sa_rust, 0, None);
        black_box(result);
        black_box(&sa_rust);
    });

    let mut sa_c = vec![0; workload.bytes.len()];
    let c_total = bench_one(iterations, || {
        let result = unsafe { probe_public_libsais(workload.bytes.as_ptr(), sa_c.as_mut_ptr(), n, 0) };
        black_box(result);
        black_box(&sa_c);
    });

    let rust_avg = rust_total.as_secs_f64() * 1000.0 / iterations as f64;
    let c_avg = c_total.as_secs_f64() * 1000.0 / iterations as f64;
    let ratio = rust_avg / c_avg;

    println!(
        "{:<48} len={:>9} iter={:>3}  rust={:>9.3} ms  c={:>9.3} ms  ratio={:>5.2}x",
        workload.name,
        workload.bytes.len(),
        iterations,
        rust_avg,
        c_avg,
        ratio
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let specs: Vec<WorkloadSpec> = if !args.is_empty() {
        args.into_iter()
            .map(|arg| {
                let path = PathBuf::from(&arg);
                if !path.exists() {
                    panic!("path does not exist: {arg}");
                }
                WorkloadSpec::from_path(path)
            })
            .collect()
    } else {
        let mut specs = vec![
            WorkloadSpec::from_path(PathBuf::from("README.md")),
            WorkloadSpec::from_path(PathBuf::from("libsais/src/libsais.c")),
            WorkloadSpec::generated("generated/mixed-1MiB", 1 << 20),
        ];

        let corpora_dir = Path::new("corpora");
        if corpora_dir.is_dir() {
            match collect_corpora_files(corpora_dir) {
                Ok(paths) => {
                    for path in paths {
                        specs.push(WorkloadSpec::from_path(path));
                    }
                }
                Err(err) => {
                    eprintln!("warning: failed to walk {}: {err}", corpora_dir.display());
                }
            }
        }

        specs
    };

    println!("Benchmarking libsais Rust vs upstream C");
    println!("release build, single-threaded, fs=0, suffix array construction");
    println!("workloads: {}", specs.len());
    println!();

    for spec in &specs {
        let workload = spec.materialize();
        benchmark_workload(&workload);
    }
}
