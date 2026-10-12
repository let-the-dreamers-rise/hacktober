//! Classes, instances, enums and the attribute protocol (`obj.name`,
//! `obj.method(args)`, `obj[index]`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use chimera_core::Span;

use super::{Args, Env, Interp, sym};
use crate::ast::*;
use crate::ml::tensor::SliceSpec;
use crate::symbol::Sym;
use crate::value::*;

/// Method names that act as constructors.
fn is_ctor_name(n: Sym) -> bool {
    let s = n.as_str();
    matches!(&*s, "init" | "__init__" | "constructor" | "initialize")
}

impl Interp {
    // ----- definitions ---------------------------------------------------

    /// Creates a class value from its declaration.
    pub fn define_class(&mut self, def: &Rc<ClassDef>, env: &Rc<Env>) -> R {
        let superclass = match &def.superclass {
            Some(e) => match self.eval(e, env)? {
                Value::Class(c) => Some(c),
                Value::Type(_) | Value::Nil => None,
                other => {
                    let t = other.type_name();
                    return self.type_err(
                        format!("a class can only extend another class, not {t}"),
                        e.span,
                    );
                }
            },
            None => None,
        };
        let class = Rc::new(Class {
            name: def.name,
            superclass,
            methods: RefCell::new(HashMap::new()),
            getters: RefCell::new(HashMap::new()),
            statics: RefCell::new(HashMap::new()),
            attrs: RefCell::new(HashMap::new()),
            def: Some(def.clone()),
            env: Some(env.clone()),
            field_names: def.fields.iter().map(|f| f.name).collect(),
        });
        for m in &def.methods {
            self.add_method(&class, m, env, false)?;
        }
        for m in &def.statics {
            self.add_method(&class, m, env, true)?;
        }
        // Class bodies see their own name (for self-referencing attributes).
        let class_env = Env::child(env, super::EnvKind::Block);
        class_env.define(def.name, Value::Class(class.clone()));
        for (name, e) in &def.class_attrs {
            let v = self.eval(e, &class_env)?;
            class.attrs.borrow_mut().insert(*name, v);
        }
        Ok(Value::Class(class))
    }

    fn method_closure(&mut self, class: &Rc<Class>, def: &Rc<FnDef>, env: &Rc<Env>) -> Value {
        let f = self.make_closure(def, env);
        if let Value::Func(c) = &f {
            *c.class.borrow_mut() = Some(Rc::downgrade(class));
        }
        f
    }

    fn add_method(
        &mut self,
        class: &Rc<Class>,
        def: &Rc<FnDef>,
        env: &Rc<Env>,
        is_static: bool,
    ) -> R<()> {
        let Some(name) = def.name else { return Ok(()) };
        let mut f = self.method_closure(class, def, env);
        let mut kind = if is_static { "static" } else { "method" };
        for d in def.decorators.iter().rev() {
            match &d.kind {
                ExprKind::Name(n) if n.is("property") => kind = "getter",
                ExprKind::Name(n) if n.is("staticmethod") || n.is("classmethod") => kind = "static",
                ExprKind::Member { name: attr, .. }
                    if attr.is("setter") || attr.is("getter") || attr.is("deleter") => {}
                ExprKind::Name(n)
                    if n.is("override") || n.is("abstractmethod") || n.is("dataclass") => {}
                _ => {
                    let dec = self.eval(d, env)?;
                    f = self.call_value(dec, Args::new(vec![f], d.span))?;
                }
            }
        }
        let name = if is_ctor_name(name) {
            sym("init")
        } else {
            name
        };
        match kind {
            "getter" => class.getters.borrow_mut().insert(name, f),
            "static" => class.statics.borrow_mut().insert(name, f),
            _ => class.methods.borrow_mut().insert(name, f),
        };
        Ok(())
    }

    /// Creates an enum value and puts its constructors in scope.
    pub fn define_enum(&mut self, def: &Rc<EnumDef>, env: &Rc<Env>) -> R {
        let mut variants = Vec::with_capacity(def.variants.len());
        let mut next_value = 0i64;
        for v in &def.variants {
            let value = match &v.value {
                Some(e) => {
                    let val = self.eval(e, env)?;
                    if let Some(i) = val.as_int() {
                        next_value = i + 1;
                    }
                    Some(val)
                }
                None => {
                    let val = Value::Int(next_value);
                    next_value += 1;
                    Some(val)
                }
            };
            variants.push(VariantInfo {
                name: v.name,
                arity: v.arity,
                field_names: v.field_names.clone(),
                value,
            });
        }
        let ty = Rc::new(EnumType {
            name: def.name,
            variants,
            methods: RefCell::new(HashMap::new()),
            statics: RefCell::new(HashMap::new()),
        });
        for m in &def.methods {
            if let Some(n) = m.name {
                let f = self.make_closure(m, env);
                ty.methods.borrow_mut().insert(n, f);
            }
        }
        // Constructors are visible unqualified too (`Circle(2.0)`, `Red`),
        // unless that would shadow something already defined.
        for (i, v) in ty.variants.iter().enumerate() {
            if env.get(v.name).is_none() && !self.builtins.contains_key(&v.name)
                || def.name.is("Result")
            {
                env.define(v.name, variant_ctor(&ty, i));
            }
        }
        Ok(Value::Enum(ty))
    }

    /// `impl`/`extension`/Go receivers/Lua `function M.f`: adds methods.
    pub fn exec_impl(
        &mut self,
        target: &Expr,
        methods: &[Rc<FnDef>],
        statics: &[Rc<FnDef>],
        env: &Rc<Env>,
        span: Span,
    ) -> R<()> {
        let t = self.eval(target, env)?;
        match &t {
            Value::Class(c) => {
                for m in methods {
                    self.add_method(c, m, env, false)?;
                }
                for m in statics {
                    self.add_method(c, m, env, true)?;
                }
            }
            Value::Enum(e) => {
                for m in methods {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        e.methods.borrow_mut().insert(n, f);
                    }
                }
                for m in statics {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        e.statics.borrow_mut().insert(n, f);
                    }
                }
            }
            Value::Type(tag) => {
                for m in methods.iter().chain(statics) {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        self.extensions.entry(*tag).or_default().insert(n, f);
                    }
                }
            }
            Value::Map(map) => {
                for m in methods.iter().chain(statics) {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        map.borrow_mut().insert(Value::Str(n.as_str()), f);
                    }
                }
            }
            Value::Module(module) => {
                for m in methods.iter().chain(statics) {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        module.members.borrow_mut().insert(n, f);
                    }
                }
            }
            Value::Instance(inst) => {
                for m in methods.iter().chain(statics) {
                    if let Some(n) = m.name {
                        let f = self.make_closure(m, env);
                        let bound = Value::Bound(Rc::new(Bound {
                            receiver: t.clone(),
                            func: f,
                        }));
                        inst.set(n, bound);
                    }
                }
            }
            other => {
                let ty = other.type_name();
                return self.type_err(format!("cannot add methods to a {ty}"), span);
            }
        }
        Ok(())
    }

    // ----- instances -------------------------------------------------------

    /// `Class(args)`: allocates an instance, fills field defaults and runs
    /// the constructor (or the generated one for structs/data classes).
    pub fn instantiate(&mut self, class: &Rc<Class>, args: Args) -> R {
        let inst = Rc::new(Instance {
            class: class.clone(),
            fields: RefCell::new(Vec::new()),
        });
        self.init_fields(class, &inst)?;
        let value = Value::Instance(inst.clone());
        if let Some(init) = class.find_method(sym("init")) {
            let bound = Value::Bound(Rc::new(Bound {
                receiver: value.clone(),
                func: init,
            }));
            self.call_value(bound, args)?;
            return Ok(value);
        }
        let names = class.all_field_names();
        if args.pos.len() > names.len() {
            let n = class.name;
            if names.is_empty() {
                return self.type_err(
                    format!("`{n}` takes no arguments (it has no constructor)"),
                    args.span,
                );
            }
            return self.type_err(
                format!(
                    "`{n}` takes {} argument(s) but {} were given",
                    names.len(),
                    args.pos.len()
                ),
                args.span,
            );
        }
        for (name, v) in names.iter().zip(args.pos) {
            inst.set(*name, v);
        }
        for (name, v) in args.named {
            if !names.contains(&name) && !names.is_empty() {
                let n = class.name;
                return self.type_err(format!("`{n}` has no field named `{name}`"), args.span);
            }
            inst.set(name, v);
        }
        Ok(value)
    }

    fn init_fields(&mut self, class: &Rc<Class>, inst: &Rc<Instance>) -> R<()> {
        if let Some(sup) = &class.superclass {
            self.init_fields(sup, inst)?;
        }
        let (Some(def), Some(env)) = (&class.def, &class.env) else {
            return Ok(());
        };
        for f in &def.fields {
            let v = match &f.default {
                Some(e) => self.eval(e, env)?,
                None => Value::Nil,
            };
            inst.set(f.name, v);
        }
        Ok(())
    }

    /// `Point { x: 1, y: 2 }`, `new Point { X = 1 }`, `Shape::Rect { w, h }`.
    pub fn eval_struct_lit(
        &mut self,
        ty: &Expr,
        fields: &[(Sym, Expr)],
        env: &Rc<Env>,
        span: Span,
    ) -> R {
        let target = self.eval(ty, env)?;
        let mut values = Vec::with_capacity(fields.len());
        for (n, e) in fields {
            values.push((*n, self.eval(e, env)?));
        }
        match &target {
            Value::Class(c) => {
                let inst = Rc::new(Instance {
                    class: c.clone(),
                    fields: RefCell::new(Vec::new()),
                });
                self.init_fields(c, &inst)?;
                for (n, v) in values {
                    if n.is("..base") {
                        if let Value::Instance(base) = &v {
                            for (k, fv) in base.fields.borrow().iter() {
                                if inst.get(*k).is_none_or(|cur| matches!(cur, Value::Nil)) {
                                    inst.set(*k, fv.clone());
                                }
                            }
                        }
                        continue;
                    }
                    inst.set(n, v);
                }
                Ok(Value::Instance(inst))
            }
            Value::Instance(inst) => {
                for (n, v) in values {
                    inst.set(n, v);
                }
                Ok(target.clone())
            }
            Value::Native(_) => {
                // A struct-like enum variant constructor: pass by field name.
                let args = Args {
                    pos: Vec::new(),
                    named: values,
                    span,
                };
                self.call_value(target.clone(), args)
            }
            other => {
                let t = other.type_name();
                self.type_err(format!("`{t}` is not a struct type"), span)
            }
        }
    }

    // ----- attribute access -----------------------------------------------------

    /// `obj.name` (without a call).
    pub fn get_attr(&mut self, obj: &Value, name: Sym, span: Span) -> R {
        match obj {
            Value::Instance(inst) => {
                if let Some(v) = inst.get(name) {
                    return Ok(v);
                }
                if let Some(g) = inst.class.find_getter(name) {
                    let bound = Value::Bound(Rc::new(Bound {
                        receiver: obj.clone(),
                        func: g,
                    }));
                    return self.call_value(bound, Args::new(Vec::new(), span));
                }
                if let Some(m) = inst.class.find_method(name) {
                    if let Value::Func(c) = &m
                        && c.def.bare
                        && c.required_params() == 0
                    {
                        let bound = Value::Bound(Rc::new(Bound {
                            receiver: obj.clone(),
                            func: m.clone(),
                        }));
                        return self.call_value(bound, Args::new(Vec::new(), span));
                    }
                    return Ok(Value::Bound(Rc::new(Bound {
                        receiver: obj.clone(),
                        func: m,
                    })));
                }
                if let Some(v) = inst.class.find_static(name) {
                    return Ok(v);
                }
                match &*name.as_str() {
                    "__class__" | "class" | "constructor" => {
                        return Ok(Value::Class(inst.class.clone()));
                    }
                    "__dict__" | "fields" => {
                        let mut d = Dict::new();
                        for (k, v) in inst.fields.borrow().iter() {
                            d.insert(Value::Str(k.as_str()), v.clone());
                        }
                        return Ok(Value::Map(Rc::new(RefCell::new(d))));
                    }
                    "message" | "msg" if inst.class.has_ancestor_named("Error") => {
                        return Ok(Value::str(""));
                    }
                    _ => {}
                }
                let candidates: Vec<Rc<str>> = inst
                    .fields
                    .borrow()
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .chain(inst.class.methods.borrow().keys().map(|k| k.as_str()))
                    .collect();
                self.attr_error(&inst.class.name.as_str(), name, &candidates, span)
            }
            Value::Class(c) => {
                if let Some(v) = c.find_static(name) {
                    return Ok(v);
                }
                if let Some(m) = c.find_method(name) {
                    return Ok(m);
                }
                match &*name.as_str() {
                    "name" | "__name__" => Ok(Value::Str(c.name.as_str())),
                    "new" => {
                        let class = c.clone();
                        Ok(Value::native("new", move |interp, args| {
                            interp.instantiate(&class, args)
                        }))
                    }
                    "superclass" | "__base__" => {
                        Ok(c.superclass.clone().map_or(Value::Nil, Value::Class))
                    }
                    _ => {
                        let candidates: Vec<Rc<str>> =
                            c.statics.borrow().keys().map(|k| k.as_str()).collect();
                        self.attr_error(&c.name.as_str(), name, &candidates, span)
                    }
                }
            }
            Value::Map(m) => {
                if let Some(v) = m.borrow().get(&Value::Str(name.as_str())) {
                    return Ok(v);
                }
                if crate::builtins::methods::has_method(obj, &name.as_str()) {
                    return self.builtin_attr(obj, name, span);
                }
                // JavaScript semantics: a missing key reads as undefined.
                Ok(Value::Nil)
            }
            Value::Module(m) => match m.get(name) {
                Some(v) => Ok(v),
                None => {
                    let candidates: Vec<Rc<str>> =
                        m.members.borrow().keys().map(|k| k.as_str()).collect();
                    let module_name = format!("module {}", m.name);
                    self.attr_error(&module_name, name, &candidates, span)
                }
            },
            Value::Enum(e) => {
                if let Some(i) = e.variant_index(name) {
                    return Ok(variant_ctor(e, i));
                }
                if let Some(f) = e.statics.borrow().get(&name) {
                    return Ok(f.clone());
                }
                match &*name.as_str() {
                    "name" | "__name__" => Ok(Value::Str(e.name.as_str())),
                    "values" | "variants" | "entries" | "allCases" | "cases" => Ok(Value::list(
                        (0..e.variants.len())
                            .filter(|&i| e.variants[i].arity == 0)
                            .map(|i| variant_ctor(e, i))
                            .collect(),
                    )),
                    _ => {
                        let candidates: Vec<Rc<str>> =
                            e.variants.iter().map(|v| v.name.as_str()).collect();
                        self.attr_error(&e.name.as_str(), name, &candidates, span)
                    }
                }
            }
            Value::Variant(v) => {
                let info = v.info();
                if let Some(i) = info.field_names.iter().position(|f| *f == name) {
                    return Ok(v.fields[i].clone());
                }
                if let Ok(i) = name.as_str().parse::<usize>()
                    && let Some(f) = v.fields.get(i)
                {
                    return Ok(f.clone());
                }
                if let Some(m) = v.ty.methods.borrow().get(&name) {
                    return Ok(Value::Bound(Rc::new(Bound {
                        receiver: obj.clone(),
                        func: m.clone(),
                    })));
                }
                match &*name.as_str() {
                    "name" | "rawValue" => Ok(Value::Str(info.name.as_str())),
                    "value" | "ordinal" => {
                        Ok(info.value.clone().unwrap_or(Value::Int(v.index as i64)))
                    }
                    _ => {
                        if let Some(p) =
                            crate::builtins::methods::property(self, obj, &name.as_str(), span)?
                        {
                            return Ok(p);
                        }
                        self.attr_error(&info.name.as_str(), name, &[], span)
                    }
                }
            }
            Value::Tuple(t) => {
                if let Ok(i) = name.as_str().parse::<usize>() {
                    return match t.get(i) {
                        Some(v) => Ok(v.clone()),
                        None => {
                            Err(self.error("IndexError", format!("tuple has no field {i}"), span))
                        }
                    };
                }
                self.builtin_attr(obj, name, span)
            }
            Value::Object(o) => {
                let n = name.as_str();
                match o.get_attr(self, &n) {
                    Some(r) => r,
                    None => {
                        let o2 = o.clone();
                        let this = obj.clone();
                        let method = n.to_string();
                        Ok(Value::native(
                            &method.clone(),
                            move |interp, args| match o2.call_method(
                                interp,
                                &method,
                                args.clone(),
                                &this,
                            ) {
                                Some(r) => r,
                                None => {
                                    let t = o2.type_name().to_owned();
                                    interp.type_err(
                                        format!("`{t}` has no method `{method}`"),
                                        args.span,
                                    )
                                }
                            },
                        ))
                    }
                }
            }
            Value::Type(t) => {
                let tag = *t;
                if matches!(&*name.as_str(), "__name__" | "name" | "simpleName" | "Name") {
                    return Ok(Value::str(tag.name()));
                }
                // `str.upper` → a function taking the receiver first.
                let method = name.as_str().to_string();
                if let Some(v) = crate::builtins::type_attr(self, tag, &method, span)? {
                    return Ok(v);
                }
                Ok(Value::native(&method.clone(), move |interp, mut args| {
                    if args.pos.is_empty() {
                        return interp.type_err(
                            format!("`{}.{method}` needs a receiver argument", tag.name()),
                            args.span,
                        );
                    }
                    let recv = args.pos.remove(0);
                    let span = args.span;
                    interp.call_method(&recv, sym(&method), args, span)
                }))
            }
            Value::Nil => {
                let n = name.as_str();
                Err(self.error(
                    "AttributeError",
                    format!("nil has no attribute `{n}` (use `?.` to make access optional)"),
                    span,
                ))
            }
            _ => self.builtin_attr(obj, name, span),
        }
    }

    fn builtin_attr(&mut self, obj: &Value, name: Sym, span: Span) -> R {
        let n = name.as_str();
        if let Some(v) = crate::builtins::methods::property(self, obj, &n, span)? {
            return Ok(v);
        }
        if let Some(tag) = crate::builtins::methods::tag_of(obj)
            && let Some(f) = self
                .extensions
                .get(&tag)
                .and_then(|m| m.get(&name))
                .cloned()
        {
            return Ok(Value::Bound(Rc::new(Bound {
                receiver: obj.clone(),
                func: f,
            })));
        }
        // A method referenced without calling it: bind it.
        let recv = obj.clone();
        let method = n.to_string();
        if crate::builtins::methods::has_method(obj, &method) {
            return Ok(Value::native(&method.clone(), move |interp, args| {
                let span = args.span;
                interp.call_method(&recv, sym(&method), args, span)
            }));
        }
        let t = obj.type_name();
        let names = crate::builtins::methods::method_names(obj);
        self.attr_error(
            &t,
            name,
            &names.iter().map(|s| Rc::from(*s)).collect::<Vec<_>>(),
            span,
        )
    }

    fn attr_error(&mut self, owner: &str, name: Sym, candidates: &[Rc<str>], span: Span) -> R {
        let n = name.as_str();
        let hint = chimera_core::text::suggest(&n, candidates.iter().map(|s| &**s));
        let msg = match hint {
            Some(h) => format!("`{owner}` has no attribute `{n}` (did you mean `{h}`?)"),
            None => format!("`{owner}` has no attribute `{n}`"),
        };
        Err(self.error("AttributeError", msg, span))
    }

    /// `obj.name = value`.
    pub fn set_attr(&mut self, obj: &Value, name: Sym, value: Value, span: Span) -> R<()> {
        match obj {
            Value::Instance(inst) => {
                inst.set(name, value);
                Ok(())
            }
            Value::Class(c) => {
                c.attrs.borrow_mut().insert(name, value);
                Ok(())
            }
            Value::Map(m) => {
                m.borrow_mut().insert(Value::Str(name.as_str()), value);
                Ok(())
            }
            Value::Module(m) => {
                m.members.borrow_mut().insert(name, value);
                Ok(())
            }
            Value::Tensor(t) => {
                match &*name.as_str() {
                    "requires_grad" => t.set_requires_grad(value.truthy()),
                    "grad" if matches!(value, Value::Nil) => t.zero_grad(),
                    "data" => {
                        let new = self.to_tensor(&value, span)?;
                        if new.numel() != t.numel() {
                            return self
                                .value_err("new data must have the same number of elements", span);
                        }
                        let d = new.data();
                        t.update(|x| x.copy_from_slice(&d));
                    }
                    other => {
                        return self.type_err(format!("cannot set `{other}` on a tensor"), span);
                    }
                }
                Ok(())
            }
            Value::Object(o) => {
                if o.set_attr(self, &name.as_str(), value) {
                    Ok(())
                } else {
                    let (t, n) = (o.type_name().to_owned(), name.as_str());
                    self.type_err(format!("cannot set `{n}` on `{t}`"), span)
                }
            }
            other => {
                let (t, n) = (other.type_name(), name.as_str());
                self.type_err(format!("cannot set attribute `{n}` on a {t}"), span)
            }
        }
    }

    /// Calls `obj.name(args)`, falling back to a free function `name(obj, …)`
    /// (uniform function call syntax) when the type has no such method.
    pub fn call_method_or_ufcs(
        &mut self,
        obj: &Value,
        name: Sym,
        args: Args,
        env: &Rc<Env>,
        span: Span,
    ) -> R {
        if let Some(r) = self.try_call_method(obj, name, args.clone(), span)? {
            return Ok(r);
        }
        let f = env.get(name).or_else(|| self.builtins.get(&name).cloned());
        if let Some(f) = f
            && f.is_callable()
        {
            let mut args = args;
            args.pos.insert(0, obj.clone());
            return self.call_value(f, args);
        }
        self.no_method(obj, name, span)
    }

    /// Calls `obj.name(args)`.
    pub fn call_method(&mut self, obj: &Value, name: Sym, args: Args, span: Span) -> R {
        match self.try_call_method(obj, name, args, span)? {
            Some(v) => Ok(v),
            None => self.no_method(obj, name, span),
        }
    }

    fn no_method(&mut self, obj: &Value, name: Sym, span: Span) -> R {
        let n = name.as_str();
        let (owner, candidates): (String, Vec<Rc<str>>) = match obj {
            Value::Instance(i) => (
                i.class.name.as_str().to_string(),
                i.class
                    .methods
                    .borrow()
                    .keys()
                    .map(|k| k.as_str())
                    .collect(),
            ),
            Value::Nil => {
                return Err(self.error(
                    "AttributeError",
                    format!("cannot call `{n}` on nil (use `?.` to make the call optional)"),
                    span,
                ));
            }
            other => (
                other.type_name(),
                crate::builtins::methods::method_names(other)
                    .iter()
                    .map(|s| Rc::from(*s))
                    .collect(),
            ),
        };
        let hint = chimera_core::text::suggest(&n, candidates.iter().map(|s| &**s));
        let msg = match hint {
            Some(h) => format!("`{owner}` has no method `{n}` (did you mean `{h}`?)"),
            None => format!("`{owner}` has no method `{n}`"),
        };
        Err(self.error("AttributeError", msg, span))
    }

    /// The method-call protocol; `None` means "no such method".
    pub fn try_call_method(
        &mut self,
        obj: &Value,
        name: Sym,
        args: Args,
        span: Span,
    ) -> R<Option<Value>> {
        match obj {
            Value::Instance(inst) => {
                if let Some(f) = inst.get(name) {
                    return self.call_value(f, args).map(Some);
                }
                if let Some(m) = inst.class.find_method(name) {
                    return match m {
                        Value::Func(c) => self.call_closure(&c, args, Some(obj.clone())).map(Some),
                        Value::Native(n) => {
                            let mut args = args;
                            args.pos.insert(0, obj.clone());
                            (n.f)(self, args).map(Some)
                        }
                        other => self.call_value(other, args).map(Some),
                    };
                }
                if let Some(g) = inst.class.find_getter(name) {
                    let bound = Value::Bound(Rc::new(Bound {
                        receiver: obj.clone(),
                        func: g,
                    }));
                    let f = self.call_value(bound, Args::new(Vec::new(), span))?;
                    return self.call_value(f, args).map(Some);
                }
                if let Some(s) = inst.class.find_static(name) {
                    return self.call_value(s, args).map(Some);
                }
                if is_ctor_name(name) {
                    // `super().__init__(…)` reaching a class without a constructor.
                    return Ok(Some(Value::Nil));
                }
                // Generic object methods every instance has.
                crate::builtins::methods::object_method(self, obj, &name.as_str(), args)
            }
            Value::Class(c) => {
                if let Some(s) = c.find_static(name) {
                    return self.call_value(s, args).map(Some);
                }
                if name.is("new") {
                    return self.instantiate(c, args).map(Some);
                }
                if let Some(m) = c.find_method(name) {
                    // `Base.method(self, …)` (explicit receiver, Python style).
                    let mut args = args;
                    if args.pos.is_empty() {
                        return self.call_value(m, args).map(Some);
                    }
                    let recv = args.pos.remove(0);
                    let bound = Value::Bound(Rc::new(Bound {
                        receiver: recv,
                        func: m,
                    }));
                    return self.call_value(bound, args).map(Some);
                }
                Ok(None)
            }
            Value::Module(m) => match m.get(name) {
                Some(f) => self.call_value(f, args).map(Some),
                None => Ok(None),
            },
            Value::Map(m) => {
                let field = m.borrow().get(&Value::Str(name.as_str()));
                if let Some(f) = field
                    && f.is_callable()
                {
                    return self.call_value(f, args).map(Some);
                }
                self.builtin_method(obj, name, args, span)
            }
            Value::Enum(e) => {
                if let Some(i) = e.variant_index(name) {
                    return self.call_value(variant_ctor(e, i), args).map(Some);
                }
                let st = e.statics.borrow().get(&name).cloned();
                if let Some(f) = st {
                    return self.call_value(f, args).map(Some);
                }
                Ok(None)
            }
            Value::Variant(v) => {
                let m = v.ty.methods.borrow().get(&name).cloned();
                if let Some(m) = m {
                    return match m {
                        Value::Func(c) => self.call_closure(&c, args, Some(obj.clone())).map(Some),
                        other => self.call_value(other, args).map(Some),
                    };
                }
                self.builtin_method(obj, name, args, span)
            }
            Value::Object(o) => {
                let this = obj.clone();
                match o.call_method(self, &name.as_str(), args, &this) {
                    Some(r) => r.map(Some),
                    None => Ok(None),
                }
            }
            Value::Type(t) => {
                let tag = *t;
                if let Some(f) = crate::builtins::type_attr(self, tag, &name.as_str(), span)? {
                    return self.call_value(f, args).map(Some);
                }
                let mut args = args;
                if args.pos.is_empty() {
                    return Ok(None);
                }
                let recv = args.pos.remove(0);
                self.try_call_method(&recv, name, args, span)
            }
            _ => self.builtin_method(obj, name, args, span),
        }
    }

    fn builtin_method(
        &mut self,
        obj: &Value,
        name: Sym,
        args: Args,
        span: Span,
    ) -> R<Option<Value>> {
        if let Some(tag) = crate::builtins::methods::tag_of(obj)
            && let Some(f) = self
                .extensions
                .get(&tag)
                .and_then(|m| m.get(&name))
                .cloned()
        {
            let bound = Value::Bound(Rc::new(Bound {
                receiver: obj.clone(),
                func: f,
            }));
            return self.call_value(bound, args).map(Some);
        }
        let _ = span;
        crate::builtins::methods::dispatch(self, obj, &name.as_str(), args)
    }

    // ----- super -------------------------------------------------------------

    /// `super.name` (bound to the current receiver).
    pub fn super_member(&mut self, name: Sym, span: Span) -> R {
        let Some((class, recv)) = self.current_method() else {
            return Err(self.error("TypeError", "`super` used outside of a method", span));
        };
        let Some(sup) = &class.superclass else {
            let n = class.name;
            return Err(self.error("TypeError", format!("`{n}` has no superclass"), span));
        };
        let name = if is_ctor_name(name) {
            sym("init")
        } else {
            name
        };
        match sup.find_method(name) {
            Some(m) => Ok(Value::Bound(Rc::new(Bound {
                receiver: recv,
                func: m,
            }))),
            None => {
                let (s, n) = (sup.name, name.as_str());
                Err(self.error(
                    "AttributeError",
                    format!("superclass `{s}` has no method `{n}`"),
                    span,
                ))
            }
        }
    }

    /// `super(args)` / `super.method(args)` / `super().method(args)`.
    pub fn call_super(&mut self, name: Option<Sym>, args: Args, span: Span) -> R {
        let Some((class, recv)) = self.current_method() else {
            return Err(self.error("TypeError", "`super` used outside of a method", span));
        };
        let Some(sup) = class.superclass.clone() else {
            return Ok(Value::Nil);
        };
        let name = name.map_or_else(
            || sym("init"),
            |n| if is_ctor_name(n) { sym("init") } else { n },
        );
        match sup.find_method(name) {
            Some(m) => {
                let bound = Value::Bound(Rc::new(Bound {
                    receiver: recv,
                    func: m,
                }));
                self.call_value(bound, args)
            }
            None if name.is("init") => {
                // Superclass without a constructor: assign declared fields.
                if let Value::Instance(inst) = &recv {
                    for (n, v) in sup.all_field_names().iter().zip(args.pos) {
                        inst.set(*n, v);
                    }
                }
                Ok(Value::Nil)
            }
            None => {
                let (s, n) = (sup.name, name.as_str());
                Err(self.error(
                    "AttributeError",
                    format!("superclass `{s}` has no method `{n}`"),
                    span,
                ))
            }
        }
    }

    // ----- indexing -------------------------------------------------------------

    /// Normalises a (possibly negative) index against `len`.
    pub fn list_index(&mut self, idx: &Value, len: usize, span: Span) -> R<usize> {
        let Some(i) = idx.as_int() else {
            let t = idx.type_name();
            return self.type_err(format!("indices must be integers, not {t}"), span);
        };
        let j = if i < 0 { i + len as i64 } else { i };
        if j < 0 || j >= len as i64 {
            return Err(self.error(
                "IndexError",
                format!("index {i} is out of range for length {len}"),
                span,
            ));
        }
        Ok(j as usize)
    }

    fn slice_bounds(
        &mut self,
        start: Option<&Expr>,
        end: Option<&Expr>,
        step: Option<&Expr>,
        env: &Rc<Env>,
    ) -> R<(Option<i64>, Option<i64>, i64)> {
        let get = |e: Option<&Expr>, interp: &mut Interp| -> R<Option<i64>> {
            match e {
                None => Ok(None),
                Some(e) => {
                    let v = interp.eval(e, env)?;
                    if matches!(v, Value::Nil) {
                        return Ok(None);
                    }
                    match v.as_int() {
                        Some(i) => Ok(Some(i)),
                        None => interp.type_err("slice bounds must be integers", e.span),
                    }
                }
            }
        };
        let s = get(start, self)?;
        let e = get(end, self)?;
        let st = get(step, self)?.unwrap_or(1);
        Ok((s, e, st))
    }

    /// `obj[index]`, including slices and multi-dimensional tensor indexing.
    pub fn eval_index(&mut self, obj: &Value, index: &Expr, env: &Rc<Env>, span: Span) -> R {
        // Ruby `h[:name]` parses like Python's `xs[:n]`; on a map it is a
        // symbol key.
        if let (
            Value::Map(m),
            ExprKind::Slice {
                start: None,
                end: Some(end),
                step: None,
            },
        ) = (obj, &index.kind)
            && let ExprKind::Name(n) = &end.kind
        {
            return Ok(m
                .borrow()
                .get(&Value::Str(n.as_str()))
                .unwrap_or(Value::Nil));
        }
        if let ExprKind::Slice { start, end, step } = &index.kind {
            let (s, e, st) =
                self.slice_bounds(start.as_deref(), end.as_deref(), step.as_deref(), env)?;
            if st == 0 {
                return self.value_err("slice step cannot be zero", span);
            }
            return self.slice_value(obj, s, e, st, span);
        }
        if let (Value::Tensor(t), ExprKind::Tuple(parts)) = (obj, &index.kind) {
            let mut specs = Vec::with_capacity(parts.len());
            for p in parts {
                specs.push(self.slice_spec(p, env)?);
            }
            return t
                .slice(&specs)
                .map(Value::Tensor)
                .map_err(|e| self.error("IndexError", e, span));
        }
        let idx = self.eval(index, env)?;
        self.index_value(obj, &idx, span)
    }

    fn slice_spec(&mut self, e: &Expr, env: &Rc<Env>) -> R<SliceSpec> {
        if let ExprKind::Slice { start, end, step } = &e.kind {
            let (s, en, st) =
                self.slice_bounds(start.as_deref(), end.as_deref(), step.as_deref(), env)?;
            return Ok(SliceSpec::Range(s, en, st));
        }
        let v = self.eval(e, env)?;
        match v.as_int() {
            Some(i) => Ok(SliceSpec::Index(i)),
            None => self.type_err("tensor indices must be integers or slices", e.span),
        }
    }

    /// Indexes with an already evaluated index value.
    pub fn index_value(&mut self, obj: &Value, idx: &Value, span: Span) -> R {
        if let Value::Range(r) = idx
            && !matches!(obj, Value::Map(_))
        {
            return self.slice_value(obj, Some(r.start), Some(r.end), r.step, span);
        }
        match obj {
            Value::List(l) => {
                let len = l.borrow().len();
                let i = self.list_index(idx, len, span)?;
                Ok(l.borrow()[i].clone())
            }
            Value::Tuple(t) => {
                let i = self.list_index(idx, t.len(), span)?;
                Ok(t[i].clone())
            }
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let i = self.list_index(idx, chars.len(), span)?;
                Ok(Value::str(&chars[i].to_string()))
            }
            Value::Map(m) => Ok(m.borrow().get(idx).unwrap_or(Value::Nil)),
            Value::Range(r) => {
                let i = self.list_index(idx, r.len(), span)?;
                Ok(Value::Int(r.get(i).unwrap_or(0)))
            }
            Value::Tensor(t) => {
                let Some(i) = idx.as_int() else {
                    return self.type_err("tensor indices must be integers", span);
                };
                let r = t
                    .slice(&[SliceSpec::Index(i)])
                    .map_err(|e| self.error("IndexError", e, span))?;
                if r.ndim() == 0 {
                    Ok(Value::Float(r.item()))
                } else {
                    Ok(Value::Tensor(r))
                }
            }
            Value::Variant(v) => {
                let i = self.list_index(idx, v.fields.len(), span)?;
                Ok(v.fields[i].clone())
            }
            Value::Instance(_) | Value::Object(_) => {
                match self.try_call_method(
                    obj,
                    sym("__getitem__"),
                    Args::new(vec![idx.clone()], span),
                    span,
                )? {
                    Some(v) => Ok(v),
                    None => match self.try_call_method(
                        obj,
                        sym("get"),
                        Args::new(vec![idx.clone()], span),
                        span,
                    )? {
                        Some(v) => Ok(v),
                        None => self.type_err(
                            "this object does not support indexing (define `__getitem__`)",
                            span,
                        ),
                    },
                }
            }
            Value::Nil => Err(self.error("TypeError", "cannot index into nil", span)),
            other => {
                let t = other.type_name();
                self.type_err(format!("`{t}` values cannot be indexed"), span)
            }
        }
    }

    fn slice_value(
        &mut self,
        obj: &Value,
        s: Option<i64>,
        e: Option<i64>,
        step: i64,
        span: Span,
    ) -> R {
        let positions = |len: usize| -> Vec<usize> {
            let n = len as i64;
            let norm = |v: i64| if v < 0 { v + n } else { v };
            let mut out = Vec::new();
            if step > 0 {
                let mut i = s.map_or(0, norm).clamp(0, n);
                let end = e.map_or(n, norm).clamp(0, n);
                while i < end {
                    out.push(i as usize);
                    i += step;
                }
            } else {
                let mut i = s.map_or(n - 1, norm).clamp(-1, n - 1);
                let end = e.map_or(-1, norm).clamp(-1, n - 1);
                while i > end {
                    out.push(i as usize);
                    i += step;
                }
            }
            out
        };
        match obj {
            Value::List(l) => {
                let items = l.borrow();
                Ok(Value::list(
                    positions(items.len())
                        .into_iter()
                        .map(|i| items[i].clone())
                        .collect(),
                ))
            }
            Value::Tuple(t) => Ok(Value::tuple(
                positions(t.len())
                    .into_iter()
                    .map(|i| t[i].clone())
                    .collect(),
            )),
            Value::Str(st) => {
                let chars: Vec<char> = st.chars().collect();
                let out: String = positions(chars.len())
                    .into_iter()
                    .map(|i| chars[i])
                    .collect();
                Ok(Value::str(&out))
            }
            Value::Range(r) => {
                let items: Vec<Value> = positions(r.len())
                    .into_iter()
                    .filter_map(|i| r.get(i))
                    .map(Value::Int)
                    .collect();
                Ok(Value::list(items))
            }
            Value::Tensor(t) => t
                .slice(&[SliceSpec::Range(s, e, step)])
                .map(Value::Tensor)
                .map_err(|err| self.error("IndexError", err, span)),
            other => {
                let t = other.type_name();
                self.type_err(format!("`{t}` values cannot be sliced"), span)
            }
        }
    }

    /// `obj[index] = value`.
    pub fn set_index(&mut self, obj: &Value, idx: Value, value: Value, span: Span) -> R<()> {
        match obj {
            Value::List(l) => {
                let len = l.borrow().len();
                // Appending at exactly `len` is allowed (JS/Lua style).
                if idx.as_int() == Some(len as i64) {
                    l.borrow_mut().push(value);
                    return Ok(());
                }
                let i = self.list_index(&idx, len, span)?;
                l.borrow_mut()[i] = value;
                Ok(())
            }
            Value::Map(m) => {
                if !m.borrow_mut().insert(idx.clone(), value) {
                    let t = idx.type_name();
                    return self.type_err(format!("unhashable key type: {t}"), span);
                }
                Ok(())
            }
            Value::Tensor(t) => {
                let Some(x) = value.as_f64() else {
                    return self.type_err("tensor elements must be numbers", span);
                };
                let Some(i) = idx.as_int() else {
                    return self.type_err("tensor indices must be integers", span);
                };
                if t.ndim() != 1 {
                    return self.type_err(
                        "item assignment needs a 1-D tensor (use slicing for more)",
                        span,
                    );
                }
                let len = t.numel();
                let i = self.list_index(&Value::Int(i), len, span)?;
                t.update(|d| d[i] = x);
                Ok(())
            }
            Value::Instance(_) => {
                match self.try_call_method(
                    obj,
                    sym("__setitem__"),
                    Args::new(vec![idx, value], span),
                    span,
                )? {
                    Some(_) => Ok(()),
                    None => self.type_err(
                        "this object does not support item assignment (define `__setitem__`)",
                        span,
                    ),
                }
            }
            other => {
                let t = other.type_name();
                self.type_err(format!("`{t}` does not support item assignment"), span)
            }
        }
    }
}

/// The value for enum variant `i`: the variant itself if it has no payload,
/// otherwise a constructor function.
pub fn variant_ctor(ty: &Rc<EnumType>, i: usize) -> Value {
    let info = &ty.variants[i];
    if info.arity == 0 {
        return Value::Variant(Rc::new(Variant {
            ty: ty.clone(),
            index: i,
            fields: Vec::new(),
        }));
    }
    let ty = ty.clone();
    let name = format!("{}::{}", ty.name, info.name);
    Value::native(&name, move |interp, args| {
        let info = &ty.variants[i];
        let mut fields = args.pos.clone();
        if !args.named.is_empty() {
            if info.field_names.is_empty() {
                return interp.type_err(
                    format!("`{}` takes positional fields", info.name),
                    args.span,
                );
            }
            fields = vec![Value::Nil; info.arity];
            for (k, v) in args.pos.iter().enumerate() {
                if k < fields.len() {
                    fields[k] = v.clone();
                }
            }
            for (n, v) in &args.named {
                match info.field_names.iter().position(|f| f == n) {
                    Some(p) => fields[p] = v.clone(),
                    None => {
                        return interp
                            .type_err(format!("`{}` has no field `{n}`", info.name), args.span);
                    }
                }
            }
        }
        if fields.len() != info.arity {
            return interp.type_err(
                format!(
                    "`{}` takes {} field(s) but {} were given",
                    info.name,
                    info.arity,
                    fields.len()
                ),
                args.span,
            );
        }
        Ok(Value::Variant(Rc::new(Variant {
            ty: ty.clone(),
            index: i,
            fields,
        })))
    })
}
