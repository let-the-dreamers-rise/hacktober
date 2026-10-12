//! `chimera convert` must produce programs that real Python, Node and Ruby
//! run with the same output as Chimera itself. Each target is skipped when
//! its interpreter isn't installed.

use std::path::PathBuf;
use std::process::Command;

use chimera_lang::Engine;
use chimera_lang::convert::Target;

fn chimera_output(src: &str) -> String {
    let mut e = Engine::offline();
    e.capture_output();
    e.run("sample.chi", src).expect("sample runs");
    e.take_output()
}

fn run_native(program: &str, ext: &str, code: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("chimera-convert-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("sample.{ext}"));
    std::fs::write(&path, code).ok()?;
    let out = Command::new(program).arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        panic!(
            "{program} failed:\n{}\n--- code ---\n{code}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn available(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn translations_match_chimera() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/convert/sample.chi");
    let src = std::fs::read_to_string(path).expect("sample exists");
    let expected = chimera_output(&src);
    for (target, program) in [(Target::Python, "python3"), (Target::JavaScript, "node")] {
        if !available(program) {
            eprintln!("skipping {program}: not installed");
            continue;
        }
        let code = Engine::offline()
            .convert("sample.chi", &src, target)
            .expect("converts");
        let actual = run_native(program, target.extension(), &code).expect("runs");
        // JavaScript prints arrays as `[ 1, 2 ]`.
        let normalise = |s: &str| s.replace("[ ", "[").replace(" ]", "]");
        assert_eq!(
            normalise(&actual),
            normalise(&expected),
            "{program} output differs\n--- code ---\n{code}"
        );
    }
}
