//! Command-line parsing and the subcommands.

use std::io::{IsTerminal, Read};

use chimera_lang::Engine;

/// Runs the CLI with `args` (without the program name); returns the exit
/// code.
pub fn main(args: Vec<String>) -> u8 {
    let mut offline = false;
    let mut rest = Vec::new();
    for a in args {
        match a.as_str() {
            "--offline" => offline = true,
            _ => rest.push(a),
        }
    }
    let new_engine = || {
        if offline {
            Engine::offline()
        } else {
            Engine::new()
        }
    };
    match rest.first().map(String::as_str) {
        None => {
            let mut src = String::new();
            if std::io::stdin().is_terminal() {
                eprintln!("usage: chimera <file.chi> [args...]");
                return 2;
            }
            if std::io::stdin().read_to_string(&mut src).is_err() {
                eprintln!("error: could not read stdin");
                return 1;
            }
            run_source(new_engine(), "<stdin>", &src, Vec::new())
        }
        Some("-e" | "eval") => {
            let code = rest.get(1).cloned().unwrap_or_default();
            run_source(
                new_engine(),
                "<eval>",
                &code,
                rest[2.min(rest.len())..].to_vec(),
            )
        }
        Some("ast") if rest.len() > 1 => {
            let src = std::fs::read_to_string(&rest[1]).unwrap_or_else(|_| rest[1].clone());
            let mut engine = new_engine();
            match engine.parse("<ast>", &src) {
                Ok(stmts) => {
                    println!("{stmts:#?}");
                    0
                }
                Err(e) => {
                    eprint!("{}", e.report);
                    1
                }
            }
        }
        Some("run") if rest.len() > 1 => run_file(new_engine(), &rest[1], rest[2..].to_vec()),
        Some(path) => run_file(new_engine(), path, rest[1..].to_vec()),
    }
}

fn run_file(engine: Engine, path: &str, args: Vec<String>) -> u8 {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return 1;
        }
    };
    let mut engine = engine;
    if let Some(dir) = std::path::Path::new(path).parent() {
        engine.set_base_dir(dir);
    }
    let mut argv = vec![path.to_owned()];
    argv.extend(args);
    run_source(engine, path, &src, argv)
}

fn run_source(mut engine: Engine, name: &str, src: &str, argv: Vec<String>) -> u8 {
    engine.set_args(argv);
    engine.set_color(std::io::stderr().is_terminal());
    match engine.run(name, src) {
        Ok(_) => 0,
        Err(e) => {
            eprint!("{}", e.report);
            1
        }
    }
}
