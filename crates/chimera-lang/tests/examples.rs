//! Every program in `examples/` must run cleanly (with the offline AI
//! backend), so the examples never rot.

use std::path::PathBuf;

use chimera_lang::Engine;

#[test]
fn examples_run() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("examples directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "chi"))
        .collect();
    files.sort();
    assert!(!files.is_empty());
    for path in files {
        let src = std::fs::read_to_string(&path).expect("readable example");
        let p = path.clone();
        let result = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let mut e = Engine::offline();
                e.capture_output();
                e.set_base_dir(p.parent().expect("has a directory"));
                let r = e.run(&p.display().to_string(), &src).map(|_| ());
                (r, e.take_output())
            })
            .expect("spawn")
            .join()
            .expect("example thread");
        match result {
            (Ok(()), out) => assert!(!out.trim().is_empty(), "{} printed nothing", path.display()),
            (Err(e), out) => panic!("{} failed:\n{out}\n{}", path.display(), e.report),
        }
    }
}
