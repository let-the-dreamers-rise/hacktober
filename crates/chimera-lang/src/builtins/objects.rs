//! Built-in objects that make other languages' I/O idioms work:
//! `console.log`, `System.out.println`, `std::cout << x << std::endl`,
//! `Console.WriteLine`, `fmt.Println`, `IO.puts`, `Math.max`, `JSON.parse`,
//! file handles from `open()`, and the Rust `Formatter` used by `impl Display`.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use chimera_core::Span;

use crate::interp::{Args, Interp, sym};
use crate::value::*;

/// Builds a module value from named members.
pub fn module(name: &str, members: Vec<(&str, Value)>) -> Value {
    let m = Module::new(name);
    for (k, v) in members {
        m.members.borrow_mut().insert(sym(k), v);
    }
    Value::Module(Rc::new(m))
}

/// Looks up a global builtin by name (for aliasing into modules).
pub fn builtin(interp: &Interp, name: &str) -> Value {
    interp
        .builtins
        .get(&sym(name))
        .cloned()
        .unwrap_or(Value::Nil)
}

/// An output stream: `std::cout`, `std::cerr`, `System.out`.
#[derive(Debug)]
pub struct Stream {
    stderr: bool,
}

impl Object for Stream {
    fn type_name(&self) -> &str {
        "ostream"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let write = |interp: &mut Interp, s: &str, stderr: bool| {
            if stderr && matches!(interp.output, crate::interp::Output::Stdout) {
                eprint!("{s}");
            } else {
                interp.write_out(s);
            }
        };
        Some((|| -> R {
            match name {
                "<<" | "write" | "print" | "Write" | "write_str" | "puts_raw" => {
                    for a in &args.pos {
                        let s = interp.to_str(a)?;
                        write(interp, &s, self.stderr);
                    }
                    Ok(this.clone())
                }
                "println" | "WriteLine" | "puts" | "writeln" => {
                    let mut parts = Vec::new();
                    for a in &args.pos {
                        parts.push(interp.to_str(a)?);
                    }
                    write(interp, &(parts.join(" ") + "\n"), self.stderr);
                    Ok(Value::Nil)
                }
                "printf" | "format" => {
                    let fmt = match args.get(0) {
                        Some(Value::Str(s)) => s.clone(),
                        _ => return interp.type_err("printf needs a format string", args.span),
                    };
                    let s = interp.printf(&fmt, &args.pos[1..], args.span)?;
                    write(interp, &s, self.stderr);
                    Ok(this.clone())
                }
                "flush" => Ok(this.clone()),
                _ => interp.type_err(format!("ostream has no method `{name}`"), args.span),
            }
        })())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A Rust `fmt::Formatter` stand-in collecting `write!` output.
#[derive(Debug, Default)]
pub struct Formatter {
    /// Text written so far.
    pub buf: RefCell<String>,
}

impl Object for Formatter {
    fn type_name(&self) -> &str {
        "Formatter"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, _this: &Value) -> Option<R> {
        match name {
            "write_str" | "write_fmt" | "push_str" => Some((|| {
                for a in &args.pos {
                    let s = interp.to_str(a)?;
                    self.buf.borrow_mut().push_str(&s);
                }
                Ok(Value::Nil)
            })()),
            _ => None,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A file opened with `open()`.
#[derive(Debug)]
pub struct FileObj {
    path: std::path::PathBuf,
    mode: String,
    content: RefCell<String>,
    pos: RefCell<usize>,
    closed: RefCell<bool>,
}

impl FileObj {
    fn flush(&self) -> std::io::Result<()> {
        if self.mode.contains('w') || self.mode.contains('a') {
            std::fs::write(&self.path, self.content.borrow().as_bytes())?;
        }
        Ok(())
    }
}

impl Object for FileObj {
    fn type_name(&self) -> &str {
        "file"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let span = args.span;
        Some((|| -> R {
            match name {
                "read" | "read_to_string" | "readAll" | "readText" => {
                    let c = self.content.borrow();
                    let pos = *self.pos.borrow();
                    let s = c[pos.min(c.len())..].to_owned();
                    *self.pos.borrow_mut() = c.len();
                    Ok(Value::str(&s))
                }
                "readline" | "readLine" | "gets" => {
                    let c = self.content.borrow();
                    let pos = *self.pos.borrow();
                    if pos >= c.len() {
                        return Ok(Value::str(""));
                    }
                    let end = c[pos..].find('\n').map_or(c.len(), |i| pos + i + 1);
                    *self.pos.borrow_mut() = end;
                    Ok(Value::str(&c[pos..end]))
                }
                "readlines" | "lines" | "readLines" | "__iter__" | "each_line" => {
                    let c = self.content.borrow();
                    let lines: Vec<Value> = c.lines().map(Value::str).collect();
                    Ok(Value::list(lines))
                }
                "write" | "puts" | "print" | "write_all" | "writeLine" | "WriteLine" => {
                    let mut text = String::new();
                    for a in &args.pos {
                        text.push_str(&interp.to_str(a)?);
                    }
                    if matches!(name, "puts" | "writeLine" | "WriteLine") {
                        text.push('\n');
                    }
                    self.content.borrow_mut().push_str(&text);
                    Ok(Value::Int(text.len() as i64))
                }
                "close" | "__exit__" | "dispose" | "Dispose" | "flush" => {
                    if !*self.closed.borrow() {
                        if let Err(e) = self.flush() {
                            return Err(interp.error(
                                "IOError",
                                format!("cannot write `{}`: {e}", self.path.display()),
                                span,
                            ));
                        }
                        if name != "flush" {
                            *self.closed.borrow_mut() = true;
                        }
                    }
                    Ok(Value::Nil)
                }
                "__enter__" => Ok(this.clone()),
                _ => interp.type_err(format!("file has no method `{name}`"), span),
            }
        })())
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match name {
            "name" | "path" => Some(Ok(Value::str(&self.path.to_string_lossy()))),
            "closed" => Some(Ok(Value::Bool(*self.closed.borrow()))),
            _ => None,
        }
    }

    fn display(&self) -> String {
        format!("<file {} mode={}>", self.path.display(), self.mode)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `open(path, mode)`.
pub fn open_file(interp: &mut Interp, path: &str, mode: &str, span: Span) -> R {
    let full = interp.base_dir.join(path);
    let content = if mode.contains('r') || mode.contains('a') {
        match std::fs::read_to_string(&full) {
            Ok(s) => s,
            Err(_) if mode.contains('a') => String::new(),
            Err(e) => {
                return Err(interp.error("IOError", format!("cannot open `{path}`: {e}"), span));
            }
        }
    } else {
        String::new()
    };
    let pos = if mode.contains('a') { content.len() } else { 0 };
    Ok(Value::Object(Rc::new(FileObj {
        path: full,
        mode: mode.to_owned(),
        content: RefCell::new(content),
        pos: RefCell::new(pos),
        closed: RefCell::new(false),
    })))
}

fn alias(interp: &Interp, names: &[(&'static str, &str)]) -> Vec<(&'static str, Value)> {
    names
        .iter()
        .map(|(k, target)| (*k, builtin(interp, target)))
        .collect()
}

/// Installs the stream objects and language-flavoured namespaces.
pub fn install(interp: &mut Interp) {
    let out = Value::Object(Rc::new(Stream { stderr: false }));
    let err = Value::Object(Rc::new(Stream { stderr: true }));

    let console = module(
        "console",
        alias(
            interp,
            &[
                ("log", "println"),
                ("info", "println"),
                ("debug", "println"),
                ("error", "eprintln"),
                ("warn", "eprintln"),
                ("print", "print"),
            ],
        ),
    );
    if let Value::Module(m) = &console {
        m.members.borrow_mut().insert(
            sym("table"),
            Value::native("console.table", |interp, args| {
                let v = super::arg(interp, &args, 0, "console.table")?;
                let rows = interp.collect(&v, args.span)?;
                let mut headers: Vec<String> = Vec::new();
                let mut cells: Vec<HashMap<String, String>> = Vec::new();
                for r in &rows {
                    let mut row = HashMap::new();
                    if let Value::Map(m) = r {
                        for (k, x) in m.borrow().iter() {
                            let key = interp.to_str(k)?;
                            if !headers.contains(&key) {
                                headers.push(key.clone());
                            }
                            row.insert(key, interp.to_str(x)?);
                        }
                    } else {
                        if !headers.contains(&"value".to_string()) {
                            headers.push("value".into());
                        }
                        row.insert("value".into(), interp.to_str(r)?);
                    }
                    cells.push(row);
                }
                let mut t = chimera_core::Table::new(headers.clone());
                for row in cells {
                    t.add_row(
                        headers
                            .iter()
                            .map(|h| row.get(h).cloned().unwrap_or_default()),
                    );
                }
                let rendered = t.render();
                interp.write_out(&rendered);
                Ok(Value::Nil)
            }),
        );
    }
    let print_stream = module(
        "PrintStream",
        alias(
            interp,
            &[
                ("println", "println"),
                ("print", "print_string"),
                ("printf", "printf"),
                ("format", "sprintf"),
            ],
        ),
    );
    let system = module(
        "System",
        vec![
            ("out", print_stream.clone()),
            ("err", err.clone()),
            (
                "currentTimeMillis",
                Value::native("currentTimeMillis", |_i, _a| {
                    let ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0);
                    Ok(Value::Int(ms))
                }),
            ),
            (
                "nanoTime",
                Value::native("nanoTime", |_i, _a| {
                    let ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as i64)
                        .unwrap_or(0);
                    Ok(Value::Int(ns))
                }),
            ),
            ("exit", builtin(interp, "exit")),
        ],
    );
    let cs_console = module(
        "Console",
        alias(interp, &[("ReadLine", "input"), ("Read", "input")]),
    );
    if let Value::Module(m) = &cs_console {
        // C# composite formatting: `Console.WriteLine("{0} and {1}", a, b)`.
        for (name, newline) in [("WriteLine", true), ("Write", false)] {
            m.members.borrow_mut().insert(
                sym(name),
                Value::native(name, move |interp, args| {
                    let mut text = match args.pos.first() {
                        Some(Value::Str(t)) if args.pos.len() > 1 && t.contains('{') => {
                            interp.brace_format(t, &args.pos[1..], &[], None, args.span)?
                        }
                        _ => {
                            let mut parts = Vec::with_capacity(args.pos.len());
                            for a in &args.pos {
                                parts.push(interp.to_str(a)?);
                            }
                            parts.join(" ")
                        }
                    };
                    if newline {
                        text.push('\n');
                    }
                    interp.write_out(&text);
                    Ok(Value::Nil)
                }),
            );
        }
    }
    let fmt = module(
        "fmt",
        alias(
            interp,
            &[
                ("Println", "println"),
                ("Print", "print_string"),
                ("Printf", "printf"),
                ("Sprintf", "sprintf"),
                ("Sprint", "tostring"),
                ("Sprintln", "tostring"),
                ("Errorf", "sprintf"),
            ],
        ),
    );
    let io = module(
        "IO",
        alias(
            interp,
            &[
                ("puts", "println"),
                ("inspect", "p"),
                ("write", "print_string"),
                ("gets", "input"),
            ],
        ),
    );
    let js_math = module(
        "Math",
        alias(
            interp,
            &[
                ("sqrt", "sqrt"),
                ("abs", "abs"),
                ("floor", "floor"),
                ("ceil", "ceil"),
                ("round", "round"),
                ("trunc", "trunc"),
                ("max", "max"),
                ("min", "min"),
                ("pow", "pow"),
                ("random", "random"),
                ("sin", "sin"),
                ("cos", "cos"),
                ("tan", "tan"),
                ("atan", "atan"),
                ("atan2", "atan2"),
                ("exp", "exp"),
                ("log", "log"),
                ("log2", "log2"),
                ("log10", "log10"),
                ("sign", "sign"),
                ("hypot", "hypot"),
                ("cbrt", "cbrt"),
                ("PI", "pi"),
                ("E", "e"),
                ("Sqrt", "sqrt"),
                ("Abs", "abs"),
                ("Max", "max"),
                ("Min", "min"),
                ("Pow", "pow"),
                ("Floor", "floor"),
                ("Round", "round"),
            ],
        ),
    );
    let json = super::json::module(interp);
    let cout = out.clone();
    let std_ns = module(
        "std",
        vec![
            ("cout", cout.clone()),
            ("cerr", err.clone()),
            ("endl", Value::str("\n")),
            ("to_string", builtin(interp, "tostring")),
            ("stoi", builtin(interp, "tonumber")),
            ("sqrt", builtin(interp, "sqrt")),
            ("abs", builtin(interp, "abs")),
            ("max", builtin(interp, "max")),
            ("min", builtin(interp, "min")),
            ("swap", Value::Nil),
            ("sort", cpp_algorithm("sort")),
            ("stable_sort", cpp_algorithm("sort")),
            ("reverse", cpp_algorithm("reverse")),
            ("accumulate", cpp_algorithm("accumulate")),
            ("count", cpp_algorithm("count")),
            ("find", cpp_algorithm("find")),
            ("fill", cpp_algorithm("fill")),
            ("max_element", cpp_algorithm("max_element")),
            ("min_element", cpp_algorithm("min_element")),
            ("string", Value::Type(TypeTag::Str)),
            ("vector", Value::Type(TypeTag::List)),
            ("map", Value::Type(TypeTag::Map)),
            (
                "process",
                module("process", vec![("exit", builtin(interp, "exit"))]),
            ),
        ],
    );
    let arrays = module(
        "Arrays",
        vec![
            ("toString", builtin(interp, "repr")),
            ("asList", Value::Type(TypeTag::List)),
            (
                "sort",
                Value::native("Arrays.sort", |interp, args| {
                    let xs = super::arg(interp, &args, 0, "Arrays.sort")?;
                    let span = args.span;
                    interp.call_method(&xs, sym("sort"), Args::new(Vec::new(), span), span)?;
                    Ok(Value::Nil)
                }),
            ),
            ("stream", builtin(interp, "__identity")),
            ("fill", Value::Nil),
        ],
    );
    let collections = module(
        "Collections",
        vec![
            ("max", builtin(interp, "max")),
            ("min", builtin(interp, "min")),
            ("reverse", builtin(interp, "reversed")),
            ("shuffle", builtin(interp, "shuffle")),
            (
                "sort",
                Value::native("Collections.sort", |interp, args| {
                    let xs = super::arg(interp, &args, 0, "Collections.sort")?;
                    let span = args.span;
                    interp.call_method(
                        &xs,
                        sym("sort"),
                        Args::new(args.pos[1..].to_vec(), span),
                        span,
                    )?;
                    Ok(Value::Nil)
                }),
            ),
        ],
    );
    let lua_table = module(
        "table",
        vec![
            (
                "insert",
                Value::native("table.insert", |interp, args| {
                    let t = super::arg(interp, &args, 0, "table.insert")?;
                    let Value::List(l) = &t else {
                        return interp.type_err("table.insert needs a list", args.span);
                    };
                    match (args.get(1), args.get(2)) {
                        (Some(pos), Some(v)) => {
                            let p = pos.as_int().unwrap_or(1).max(1) as usize - 1;
                            let len = l.borrow().len();
                            l.borrow_mut().insert(p.min(len), v.clone());
                        }
                        (Some(v), None) => l.borrow_mut().push(v.clone()),
                        _ => {}
                    }
                    Ok(Value::Nil)
                }),
            ),
            ("concat", builtin(interp, "implode")),
            ("sort", builtin(interp, "sorted")),
            (
                "remove",
                Value::native("table.remove", |interp, args| {
                    let t = super::arg(interp, &args, 0, "table.remove")?;
                    let Value::List(l) = &t else {
                        return interp.type_err("table.remove needs a list", args.span);
                    };
                    let len = l.borrow().len();
                    if len == 0 {
                        return Ok(Value::Nil);
                    }
                    let p = args
                        .get(1)
                        .and_then(Value::as_int)
                        .map_or(len, |p| p.max(1) as usize)
                        .min(len);
                    Ok(l.borrow_mut().remove(p - 1))
                }),
            ),
            ("unpack", builtin(interp, "__identity")),
        ],
    );
    let elixir_enum = module(
        "Enum",
        alias(
            interp,
            &[
                ("map", "map"),
                ("filter", "filter"),
                ("reduce", "reduce"),
                ("sum", "sum"),
                ("count", "len"),
                ("sort", "sorted"),
                ("reverse", "reversed"),
                ("max", "max"),
                ("min", "min"),
                ("each", "map"),
                ("zip", "zip"),
                ("take", "take"),
                ("drop", "drop"),
                ("uniq", "unique"),
                ("join", "implode"),
                ("group_by", "group_by"),
                ("with_index", "enumerate"),
                ("any?", "any"),
                ("all?", "all"),
                ("frequencies", "frequencies"),
            ],
        ),
    );
    let members: Vec<(&str, Value)> = vec![
        ("console", console),
        ("System", system),
        ("Console", cs_console),
        ("fmt", fmt),
        ("IO", io),
        ("Math", js_math),
        ("JSON", json.clone()),
        ("json", json),
        ("std", std_ns),
        ("cout", cout),
        ("cerr", err),
        ("endl", Value::str("\n")),
        ("Arrays", arrays),
        ("Collections", collections),
        ("table", lua_table),
        ("Enum_", elixir_enum),
        (
            "io",
            module(
                "io",
                alias(interp, &[("write", "print_string"), ("read", "input")]),
            ),
        ),
    ];
    for (k, v) in members {
        interp.builtins.insert(sym(k), v);
    }
}

/// A C++ iterator pair `(v.begin(), v.end())`: iterators are `(list, index)`
/// tuples.
fn cpp_range(args: &[Value]) -> Option<(Rc<RefCell<Vec<Value>>>, usize, usize)> {
    let (Some(Value::Tuple(a)), Some(Value::Tuple(b))) = (args.first(), args.get(1)) else {
        return None;
    };
    if a.len() != 2 || b.len() != 2 {
        return None;
    }
    let (Value::List(l), Some(i), Some(j)) = (&a[0], a[1].as_int(), b[1].as_int()) else {
        return None;
    };
    let n = l.borrow().len();
    Some((
        l.clone(),
        (i.max(0) as usize).min(n),
        (j.max(0) as usize).min(n),
    ))
}

/// `std::sort(v.begin(), v.end())` and friends; with a container instead
/// of iterators they work on the whole container.
fn cpp_algorithm(name: &'static str) -> Value {
    Value::native(name, move |interp, args| {
        let span = args.span;
        let (list, lo, hi, rest) = match cpp_range(&args.pos) {
            Some((l, i, j)) => (l, i, j, args.pos[2..].to_vec()),
            None => match args.pos.first() {
                Some(Value::List(l)) => {
                    let n = l.borrow().len();
                    (l.clone(), 0, n, args.pos[1..].to_vec())
                }
                Some(other) => {
                    let items = interp.collect(other, span)?;
                    let n = items.len();
                    (Rc::new(RefCell::new(items)), 0, n, args.pos[1..].to_vec())
                }
                None => return interp.type_err(format!("std::{name} needs a range"), span),
            },
        };
        let slice: Vec<Value> = list.borrow()[lo..hi.max(lo)].to_vec();
        let iter_at =
            |i: usize| Value::tuple(vec![Value::List(list.clone()), Value::Int(i as i64)]);
        match name {
            "sort" => {
                let sorted = super::core::sort_values(interp, slice, rest.first(), false, span)?;
                list.borrow_mut().splice(lo..hi.max(lo), sorted);
                Ok(Value::Nil)
            }
            "reverse" => {
                let mut r = slice;
                r.reverse();
                list.borrow_mut().splice(lo..hi.max(lo), r);
                Ok(Value::Nil)
            }
            "fill" => {
                let v = rest.first().cloned().unwrap_or(Value::Nil);
                for x in &mut list.borrow_mut()[lo..hi.max(lo)] {
                    *x = v.clone();
                }
                Ok(Value::Nil)
            }
            "accumulate" => {
                let mut acc = rest.first().cloned().unwrap_or(Value::Int(0));
                for x in slice {
                    acc = match rest.get(1) {
                        Some(f) => super::call(interp, f, vec![acc, x], span)?,
                        None => interp.binary_op(crate::ast::BinOp::Add, acc, x, span)?,
                    };
                }
                Ok(acc)
            }
            "count" => {
                let target = rest.first().cloned().unwrap_or(Value::Nil);
                let mut n = 0;
                for x in &slice {
                    if interp.values_equal(x, &target)? {
                        n += 1;
                    }
                }
                Ok(Value::Int(n))
            }
            "find" => {
                let target = rest.first().cloned().unwrap_or(Value::Nil);
                for (i, x) in slice.iter().enumerate() {
                    if interp.values_equal(x, &target)? {
                        return Ok(iter_at(lo + i));
                    }
                }
                Ok(iter_at(hi))
            }
            _ => {
                // max_element / min_element
                let mut best: Option<usize> = None;
                for (i, x) in slice.iter().enumerate() {
                    let better = match best {
                        None => true,
                        Some(b) => {
                            let ord = interp.compare_values(x, &slice[b], span)?;
                            if name == "max_element" {
                                ord.is_gt()
                            } else {
                                ord.is_lt()
                            }
                        }
                    };
                    if better {
                        best = Some(i);
                    }
                }
                Ok(iter_at(lo + best.unwrap_or(hi - lo)))
            }
        }
    })
}

/// Renders an instance through its Rust-style `fmt(&self, f)` method.
pub fn rust_display(interp: &mut Interp, v: &Value, method: Value) -> R<String> {
    let formatter = Rc::new(Formatter::default());
    let f = Value::Object(formatter.clone());
    let bound = Value::Bound(Rc::new(Bound {
        receiver: v.clone(),
        func: method,
    }));
    interp.call_value(bound, Args::new(vec![f], Span::default()))?;
    let s = formatter.buf.borrow().clone();
    Ok(s)
}
