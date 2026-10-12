//! `import`, `from … import`, `use`, `require`: built-in modules, `.chi`
//! files, and politely ignored foreign packages.

use std::path::PathBuf;
use std::rc::Rc;

use chimera_core::Span;

use super::{Env, Interp};
use crate::ast::Import;
use crate::symbol::Sym;
use crate::value::*;

/// Package roots from other ecosystems; importing them is a no-op so that
/// pasted Java/C#/Rust/Swift code keeps running.
const FOREIGN_ROOTS: &[&str] = &[
    "java",
    "javax",
    "kotlin",
    "kotlinx",
    "scala",
    "android",
    "org",
    "com",
    "net",
    "System",
    "Microsoft",
    "Foundation",
    "UIKit",
    "SwiftUI",
    "Cocoa",
    "std",
    "core",
    "alloc",
    "crate",
    "super",
    "self",
    "__future__",
    "typing",
    "dataclasses",
    "abc",
    "enum",
    "functools",
    "itertools",
    "collections",
    "io",
    "react",
    "lodash",
    "strings",
    "strconv",
    "bufio",
    "errors",
    "unicode",
    "Data",
    "Control",
    "Prelude",
    "Text",
    "__foreign",
    "stdio",
    "iostream",
    "stdlib",
    "vector",
    "string",
    "cmath",
    "bits",
    "algorithm",
    "numbers",
];

impl Interp {
    /// Executes an import statement, binding names in `env`.
    pub fn exec_import(&mut self, imp: &Import, env: &Rc<Env>, span: Span) -> R<()> {
        let module = match &imp.file {
            Some(path) => Some(self.import_file(path, span)?),
            None => self.resolve_module(&imp.path, span)?,
        };
        let Some(module) = module else {
            return Ok(()); // a foreign package
        };
        if imp.wildcard {
            if let Value::Module(m) = &module {
                for (k, v) in m.members.borrow().iter() {
                    if !k.as_str().starts_with('_') {
                        env.define(*k, v.clone());
                    }
                }
            }
            return Ok(());
        }
        if let Some(names) = &imp.names {
            for (name, alias) in names {
                let v = self.get_attr(&module, *name, span)?;
                env.define(alias.unwrap_or(*name), v);
            }
            return Ok(());
        }
        let bind = imp
            .alias
            .or_else(|| imp.path.last().copied())
            .or_else(|| match &module {
                Value::Module(m) => Some(Sym::new(&m.name)),
                _ => None,
            });
        if let Some(name) = bind {
            env.define(name, module);
        }
        Ok(())
    }

    /// Resolves a dotted module path: built-in modules, then `.chi` files
    /// next to the program, then known foreign packages (ignored).
    fn resolve_module(&mut self, path: &[Sym], span: Span) -> R<Option<Value>> {
        let dotted: Vec<String> = path.iter().map(|s| s.as_str().to_string()).collect();
        let joined = dotted.join(".");
        if let Some(m) = self.modules.get(&joined) {
            return Ok(Some(m.clone()));
        }
        // Try the full path, then its last segment (`os.path` → `path`,
        // `numpy` → `numpy`).
        for candidate in [
            joined.as_str(),
            dotted.last().map(String::as_str).unwrap_or(""),
        ] {
            if let Some(m) = crate::builtins::modules::builtin_module(self, candidate) {
                self.modules.insert(joined.clone(), m.clone());
                return Ok(Some(m));
            }
        }
        let file = format!("{}.chi", dotted.join("/"));
        if self.base_dir.join(&file).exists() {
            return self.import_file(&file, span).map(Some);
        }
        let root = dotted.first().map(String::as_str).unwrap_or("");
        if FOREIGN_ROOTS.contains(&root) || root.chars().next().is_some_and(char::is_uppercase) {
            return Ok(None);
        }
        Err(self.error(
            "ImportError",
            format!("no module named `{joined}` (looked for a built-in module and `{file}`)"),
            span,
        ))
    }

    /// Runs a `.chi` file in its own scope and returns it as a module.
    pub fn import_file(&mut self, path: &str, span: Span) -> R<Value> {
        let mut full = PathBuf::from(path);
        if full.extension().is_none() {
            full.set_extension("chi");
        }
        if full.is_relative() {
            full = self.base_dir.join(full);
        }
        let key = full.to_string_lossy().to_string();
        if let Some(m) = self.modules.get(&key) {
            return Ok(m.clone());
        }
        let text = match std::fs::read_to_string(&full) {
            Ok(t) => t,
            Err(e) => {
                return Err(self.error(
                    "ImportError",
                    format!("cannot read `{}`: {e}", full.display()),
                    span,
                ));
            }
        };
        let name = full
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "module".into());
        let file = self.add_file(&full.to_string_lossy(), &text);
        let (stmts, errors) = crate::parser::parse(&text, file);
        if let Some(first) = errors.first() {
            let rendered = first.render(&self.files[file as usize], false);
            return Err(self.error(
                "ImportError",
                format!("`{}` has errors:\n{rendered}", full.display()),
                span,
            ));
        }
        let module = Rc::new(Module::new(&name));
        let value = Value::Module(module.clone());
        // Cache first so circular imports see the (partial) module.
        self.modules.insert(key, value.clone());
        let scope = Env::global();
        let saved_dir = self.base_dir.clone();
        if let Some(parent) = full.parent() {
            self.base_dir = parent.to_path_buf();
        }
        let result = self.run_in(&stmts, file, &scope);
        self.base_dir = saved_dir;
        result?;
        for (k, v) in scope.local_bindings() {
            module.members.borrow_mut().insert(k, v);
        }
        Ok(value)
    }
}
