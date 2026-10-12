//! The `chimera` command-line tool.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The tree-walking interpreter recurses deeply; give it a big stack.
    let child = std::thread::Builder::new()
        .name("chimera".into())
        .stack_size(1 << 30)
        .spawn(move || chimera::cli::main(args));
    match child.map(|h| h.join()) {
        Ok(Ok(code)) => ExitCode::from(code),
        _ => ExitCode::from(101),
    }
}
