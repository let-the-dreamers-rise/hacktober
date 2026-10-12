//! Golden tests: every `tests/golden/*.chi` program runs with the offline
//! AI backend and its output must match the neighbouring `.out` file.
//!
//! Run `BLESS=1 cargo test --test golden` to (re)write the expected output
//! after checking that the new output is right.

use std::path::{Path, PathBuf};

use chimera_lang::Engine;

fn run(path: &Path) -> String {
    let src = std::fs::read_to_string(path).expect("readable test program");
    let mut engine = Engine::offline();
    engine.capture_output();
    engine.set_base_dir(path.parent().expect("test programs live in a directory"));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("test.chi");
    let result = engine.run(name, &src);
    let mut out = engine.take_output();
    if let Err(e) = result {
        out.push_str(&format!("error: {}\n", e.message));
    }
    out
}

#[test]
fn golden_programs() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("tests/golden exists")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "chi"))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no golden programs found");
    let bless = std::env::var_os("BLESS").is_some();
    let mut failures = Vec::new();
    for program in &programs {
        // Each program runs on its own thread with a big stack, like the CLI.
        let p = program.clone();
        let actual = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || run(&p))
            .expect("spawn test thread")
            .join()
            .unwrap_or_else(|_| "<panicked>\n".into());
        let expected_path = program.with_extension("out");
        if bless {
            std::fs::write(&expected_path, &actual).expect("write .out file");
            continue;
        }
        let expected = std::fs::read_to_string(&expected_path).unwrap_or_default();
        if expected != actual {
            failures.push(format!(
                "--- {} ---\n{}",
                program.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                diff(&expected, &actual)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} golden program(s) differ:\n\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A minimal line diff: the first differing lines with context.
fn diff(expected: &str, actual: &str) -> String {
    let e: Vec<&str> = expected.lines().collect();
    let a: Vec<&str> = actual.lines().collect();
    let first = e
        .iter()
        .zip(&a)
        .position(|(x, y)| x != y)
        .unwrap_or(e.len().min(a.len()));
    let mut out = String::new();
    for i in first.saturating_sub(2)..(first + 4).min(e.len().max(a.len())) {
        match (e.get(i), a.get(i)) {
            (Some(x), Some(y)) if x == y => out.push_str(&format!("  {x}\n")),
            (x, y) => {
                if let Some(x) = x {
                    out.push_str(&format!("- {x}\n"));
                }
                if let Some(y) = y {
                    out.push_str(&format!("+ {y}\n"));
                }
            }
        }
    }
    out
}
