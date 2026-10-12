//! Runtime values.
//!
//! Chimera is dynamically typed. A [`Value`] is a small tagged enum: numbers,
//! booleans and `nil` are immediate; strings, collections, functions, classes
//! and instances are reference-counted handles. Collections use interior
//! mutability (`RefCell`) so aliasing behaves like Python/JS/Ruby: two
//! variables bound to one list see each other's mutations.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::{Rc, Weak};

use crate::ast::{ClassDef, FnDef};
use crate::bigint::BigInt;
use crate::interp::{Args, Env, Flow, Interp};
use crate::ml::tensor::Tensor;
use crate::symbol::Sym;

/// Result type of everything that can throw or unwind.
pub type R<T = Value> = Result<T, Flow>;

/// A Chimera value.
#[derive(Clone)]
pub enum Value {
    /// `nil`, `null`, `None`, `undefined`.
    Nil,
    /// A boolean.
    Bool(bool),
    /// A machine integer (promoted to [`Value::Big`] on overflow).
    Int(i64),
    /// A float.
    Float(f64),
    /// An arbitrary-precision integer.
    Big(Rc<BigInt>),
    /// An immutable string.
    Str(Rc<str>),
    /// A mutable list.
    List(Rc<RefCell<Vec<Value>>>),
    /// An immutable tuple.
    Tuple(Rc<[Value]>),
    /// A mutable, insertion-ordered dictionary.
    Map(Rc<RefCell<Dict>>),
    /// A mutable, insertion-ordered set.
    Set(Rc<RefCell<Dict>>),
    /// A lazy integer range.
    Range(Rc<RangeVal>),
    /// A user-defined function or lambda.
    Func(Rc<Closure>),
    /// A built-in function.
    Native(Rc<NativeFn>),
    /// A method bound to its receiver.
    Bound(Rc<Bound>),
    /// A class.
    Class(Rc<Class>),
    /// An instance of a class.
    Instance(Rc<Instance>),
    /// An enum / algebraic data type.
    Enum(Rc<EnumType>),
    /// A value of an enum.
    Variant(Rc<Variant>),
    /// A module (built-in or imported file).
    Module(Rc<Module>),
    /// A built-in type, usable as a conversion function and in type tests.
    Type(TypeTag),
    /// An n-dimensional array with autograd support.
    Tensor(Tensor),
    /// A dual number for forward-mode differentiation.
    Dual(Rc<crate::ml::dual::Dual>),
    /// A native object (models, optimizers, streams, AI brains, …).
    Object(Rc<dyn Object>),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", crate::interp::format::repr_plain(self))
    }
}

/// Built-in types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TypeTag {
    /// `nil`
    Nil,
    /// `bool`
    Bool,
    /// `int`
    Int,
    /// `float`
    Float,
    /// `number` (int or float)
    Number,
    /// `str`
    Str,
    /// `list`
    List,
    /// `tuple`
    Tuple,
    /// `dict`
    Map,
    /// `set`
    Set,
    /// `range`
    Range,
    /// functions of any kind
    Func,
    /// `tensor`
    Tensor,
    /// `type`
    Type,
    /// `object` / `any`
    Any,
}

impl TypeTag {
    /// The canonical name.
    pub fn name(self) -> &'static str {
        match self {
            TypeTag::Nil => "nil",
            TypeTag::Bool => "bool",
            TypeTag::Int => "int",
            TypeTag::Float => "float",
            TypeTag::Number => "number",
            TypeTag::Str => "str",
            TypeTag::List => "list",
            TypeTag::Tuple => "tuple",
            TypeTag::Map => "dict",
            TypeTag::Set => "set",
            TypeTag::Range => "range",
            TypeTag::Func => "function",
            TypeTag::Tensor => "tensor",
            TypeTag::Type => "type",
            TypeTag::Any => "object",
        }
    }

    /// Resolves a type name from any of the supported languages.
    pub fn from_name(name: &str) -> Option<TypeTag> {
        Some(match name {
            "nil" | "None" | "NoneType" | "void" | "Void" | "unit" | "Unit" | "null"
            | "undefined" => TypeTag::Nil,
            "bool" | "Bool" | "boolean" | "Boolean" => TypeTag::Bool,
            "int" | "Int" | "integer" | "Integer" | "long" | "Long" | "short" | "byte" | "i8"
            | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
            | "isize" | "uint" | "size_t" | "BigInt" | "Fixnum" | "Bignum" | "int8" | "int16"
            | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" | "uintptr"
            | "ulong" | "Int32" | "Int64" => TypeTag::Int,
            "float" | "Float" | "double" | "Double" | "f32" | "f64" | "real" | "decimal"
            | "Decimal" | "float32" | "float64" | "Float32" | "Float64" => TypeTag::Float,
            "number" | "Number" | "Numeric" | "num" => TypeTag::Number,
            "str" | "string" | "String" | "char" | "Char" | "Character" | "rune" | "Symbol" => {
                TypeTag::Str
            }
            "list" | "List" | "array" | "Array" | "Vec" | "vector" | "ArrayList" | "slice" => {
                TypeTag::List
            }
            "tuple" | "Tuple" => TypeTag::Tuple,
            "dict" | "Dict" | "map" | "Map" | "HashMap" | "Hash" | "object" | "Object"
            | "dictionary" | "Dictionary" | "table" | "BTreeMap" => TypeTag::Map,
            "set" | "Set" | "HashSet" | "frozenset" | "BTreeSet" => TypeTag::Set,
            "range" | "Range" => TypeTag::Range,
            "function" | "Function" | "fn" | "Fn" | "callable" | "Callable" | "func" | "lambda"
            | "Proc" => TypeTag::Func,
            "tensor" | "Tensor" | "ndarray" | "matrix" | "Matrix" => TypeTag::Tensor,
            "type" | "Type" | "class" | "Class" => TypeTag::Type,
            "any" | "Any" | "auto" | "dynamic" => TypeTag::Any,
            _ => return None,
        })
    }
}

/// A lazy integer range `start..end` (end exclusive) stepping by `step`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeVal {
    /// First value.
    pub start: i64,
    /// Exclusive bound.
    pub end: i64,
    /// Step (never zero).
    pub step: i64,
}

impl RangeVal {
    /// Number of elements.
    pub fn len(&self) -> usize {
        if self.step > 0 && self.end > self.start {
            ((self.end - self.start - 1) / self.step + 1) as usize
        } else if self.step < 0 && self.end < self.start {
            ((self.start - self.end - 1) / -self.step + 1) as usize
        } else {
            0
        }
    }

    /// Whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`th element.
    pub fn get(&self, i: usize) -> Option<i64> {
        (i < self.len()).then(|| self.start + self.step * i as i64)
    }

    /// Whether `x` is one of the range's elements.
    pub fn contains(&self, x: i64) -> bool {
        let in_bounds = if self.step > 0 {
            x >= self.start && x < self.end
        } else {
            x <= self.start && x > self.end
        };
        in_bounds && (x - self.start) % self.step == 0
    }
}

/// A closure: a function definition plus the environment it captured.
pub struct Closure {
    /// The definition.
    pub def: Rc<FnDef>,
    /// Captured environment.
    pub env: Rc<Env>,
    /// The class a method was defined in (for `super`).
    pub class: RefCell<Option<Weak<Class>>>,
}

impl fmt::Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.name())
    }
}

impl Closure {
    /// The function's name or `<lambda>`.
    pub fn name(&self) -> Rc<str> {
        self.def
            .name
            .map_or_else(|| Rc::from("<lambda>"), Sym::as_str)
    }

    /// Number of parameters without defaults.
    pub fn required_params(&self) -> usize {
        self.def
            .params
            .iter()
            .filter(|p| p.default.is_none() && p.kind == crate::ast::ParamKind::Normal)
            .count()
    }
}

/// Signature of native functions.
pub type NativeImpl = dyn Fn(&mut Interp, Args) -> R;

/// A built-in function.
pub struct NativeFn {
    /// Display name.
    pub name: Rc<str>,
    /// Implementation.
    pub f: Box<NativeImpl>,
}

impl fmt::Debug for NativeFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<builtin {}>", self.name)
    }
}

/// A method bound to a receiver.
#[derive(Debug)]
pub struct Bound {
    /// The receiver (`self`).
    pub receiver: Value,
    /// The function (a closure or native).
    pub func: Value,
}

/// A class.
pub struct Class {
    /// Class name.
    pub name: Sym,
    /// Parent class.
    pub superclass: Option<Rc<Class>>,
    /// Instance methods by name.
    pub methods: RefCell<HashMap<Sym, Value>>,
    /// Getter methods (`@property`, JS `get x()`).
    pub getters: RefCell<HashMap<Sym, Value>>,
    /// Static/class methods.
    pub statics: RefCell<HashMap<Sym, Value>>,
    /// Class attributes (shared, Python class variables).
    pub attrs: RefCell<HashMap<Sym, Value>>,
    /// Declaration (fields with default expressions), if user-defined.
    pub def: Option<Rc<ClassDef>>,
    /// Environment the class was defined in (for field defaults).
    pub env: Option<Rc<Env>>,
    /// Field names for positional construction and pattern matching.
    pub field_names: Vec<Sym>,
}

impl fmt::Debug for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<class {}>", self.name)
    }
}

impl Class {
    /// Creates a class with no members.
    pub fn new(name: &str, superclass: Option<Rc<Class>>) -> Class {
        Class {
            name: Sym::new(name),
            superclass,
            methods: RefCell::new(HashMap::new()),
            getters: RefCell::new(HashMap::new()),
            statics: RefCell::new(HashMap::new()),
            attrs: RefCell::new(HashMap::new()),
            def: None,
            env: None,
            field_names: Vec::new(),
        }
    }

    /// Looks up an instance method, walking the superclass chain.
    pub fn find_method(&self, name: Sym) -> Option<Value> {
        if let Some(m) = self.methods.borrow().get(&name) {
            return Some(m.clone());
        }
        self.superclass.as_ref().and_then(|s| s.find_method(name))
    }

    /// Looks up a getter, walking the superclass chain.
    pub fn find_getter(&self, name: Sym) -> Option<Value> {
        if let Some(m) = self.getters.borrow().get(&name) {
            return Some(m.clone());
        }
        self.superclass.as_ref().and_then(|s| s.find_getter(name))
    }

    /// Looks up a static method or class attribute, walking the chain.
    pub fn find_static(&self, name: Sym) -> Option<Value> {
        if let Some(m) = self.statics.borrow().get(&name) {
            return Some(m.clone());
        }
        if let Some(m) = self.attrs.borrow().get(&name) {
            return Some(m.clone());
        }
        self.superclass.as_ref().and_then(|s| s.find_static(name))
    }

    /// Whether this class is `other` or inherits from it.
    pub fn is_subclass_of(&self, other: &Class) -> bool {
        if std::ptr::eq(self, other) {
            return true;
        }
        self.superclass
            .as_ref()
            .is_some_and(|s| s.is_subclass_of(other))
    }

    /// Whether this class or an ancestor has the given name.
    pub fn has_ancestor_named(&self, name: &str) -> bool {
        self.name.is(name)
            || self
                .superclass
                .as_ref()
                .is_some_and(|s| s.has_ancestor_named(name))
    }

    /// All field names, ancestors first.
    pub fn all_field_names(&self) -> Vec<Sym> {
        let mut names = self
            .superclass
            .as_ref()
            .map(|s| s.all_field_names())
            .unwrap_or_default();
        for n in &self.field_names {
            if !names.contains(n) {
                names.push(*n);
            }
        }
        names
    }
}

/// An instance of a class.
pub struct Instance {
    /// Its class.
    pub class: Rc<Class>,
    /// Fields in insertion order.
    pub fields: RefCell<Vec<(Sym, Value)>>,
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{} instance>", self.class.name)
    }
}

impl Instance {
    /// Reads a field.
    pub fn get(&self, name: Sym) -> Option<Value> {
        self.fields
            .borrow()
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.clone())
    }

    /// Writes a field, creating it if needed.
    pub fn set(&self, name: Sym, value: Value) {
        let mut fields = self.fields.borrow_mut();
        if let Some(slot) = fields.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            fields.push((name, value));
        }
    }

    /// Removes a field.
    pub fn remove(&self, name: Sym) -> bool {
        let mut fields = self.fields.borrow_mut();
        let before = fields.len();
        fields.retain(|(n, _)| *n != name);
        fields.len() != before
    }
}

/// An enum / algebraic data type.
pub struct EnumType {
    /// Type name.
    pub name: Sym,
    /// Variants.
    pub variants: Vec<VariantInfo>,
    /// Methods (from the enum body or `impl` blocks).
    pub methods: RefCell<HashMap<Sym, Value>>,
    /// Associated functions (`Color::from_hex`, `Shape.unit()`).
    pub statics: RefCell<HashMap<Sym, Value>>,
}

impl fmt::Debug for EnumType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<enum {}>", self.name)
    }
}

impl EnumType {
    /// Finds a variant index by name.
    pub fn variant_index(&self, name: Sym) -> Option<usize> {
        self.variants.iter().position(|v| v.name == name)
    }
}

/// Metadata for one variant.
#[derive(Debug, Clone)]
pub struct VariantInfo {
    /// Name.
    pub name: Sym,
    /// Number of fields.
    pub arity: usize,
    /// Field names (struct-like variants).
    pub field_names: Vec<Sym>,
    /// Explicit discriminant.
    pub value: Option<Value>,
}

/// A value of an enum.
pub struct Variant {
    /// The enum.
    pub ty: Rc<EnumType>,
    /// Which variant.
    pub index: usize,
    /// Payload.
    pub fields: Vec<Value>,
}

impl fmt::Debug for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.info().name)
    }
}

impl Variant {
    /// The variant's metadata.
    pub fn info(&self) -> &VariantInfo {
        &self.ty.variants[self.index]
    }

    /// The variant's name.
    pub fn name(&self) -> Sym {
        self.info().name
    }
}

/// A module.
pub struct Module {
    /// Module name.
    pub name: Rc<str>,
    /// Members.
    pub members: RefCell<HashMap<Sym, Value>>,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<module {}>", self.name)
    }
}

impl Module {
    /// Creates an empty module.
    pub fn new(name: &str) -> Module {
        Module {
            name: Rc::from(name),
            members: RefCell::new(HashMap::new()),
        }
    }

    /// Reads a member.
    pub fn get(&self, name: Sym) -> Option<Value> {
        self.members.borrow().get(&name).cloned()
    }
}

/// Native objects exposed to Chimera code.
///
/// Implement this to add a new built-in type: models, optimizers, output
/// streams and AI helpers are all `Object`s.
pub trait Object: Any + fmt::Debug {
    /// Type name shown by `type(x)` and in error messages.
    fn type_name(&self) -> &str;

    /// Reads an attribute (`obj.name`).
    fn get_attr(&self, _interp: &mut Interp, _name: &str) -> Option<R> {
        None
    }

    /// Writes an attribute; returns `false` if unsupported.
    fn set_attr(&self, _interp: &mut Interp, _name: &str, _value: Value) -> bool {
        false
    }

    /// Calls a method (`obj.name(args)`); `this` is the object's own value.
    fn call_method(
        &self,
        _interp: &mut Interp,
        _name: &str,
        _args: Args,
        _this: &Value,
    ) -> Option<R> {
        None
    }

    /// Calls the object itself (`model(x)`).
    fn call(&self, _interp: &mut Interp, _args: Args, _this: &Value) -> Option<R> {
        None
    }

    /// How the object prints.
    fn display(&self) -> String {
        format!("<{}>", self.type_name())
    }

    /// Downcasting support.
    fn as_any(&self) -> &dyn Any;
}

/// A hashable dictionary key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// `nil`
    Nil,
    /// booleans
    Bool(bool),
    /// integers (and integral floats)
    Int(i64),
    /// big integers, by decimal text
    Big(Rc<str>),
    /// non-integral floats, by bits
    Float(u64),
    /// strings
    Str(Rc<str>),
    /// tuples of keys
    Tuple(Rc<[Key]>),
    /// identity of a reference value
    Ptr(usize),
}

/// Computes the dictionary key for a value, or `None` if it is unhashable.
pub fn key_of(v: &Value) -> Option<Key> {
    Some(match v {
        Value::Nil => Key::Nil,
        Value::Bool(b) => Key::Bool(*b),
        Value::Int(i) => Key::Int(*i),
        Value::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 9.0e18 {
                Key::Int(*f as i64)
            } else if *f == 0.0 {
                Key::Int(0)
            } else {
                Key::Float(f.to_bits())
            }
        }
        Value::Big(b) => match b.to_i64() {
            Some(i) => Key::Int(i),
            None => Key::Big(Rc::from(b.to_string())),
        },
        Value::Str(s) => Key::Str(s.clone()),
        Value::Tuple(items) => {
            Key::Tuple(items.iter().map(key_of).collect::<Option<Vec<_>>>()?.into())
        }
        Value::List(_) | Value::Map(_) | Value::Set(_) => return None,
        Value::Variant(v) if v.fields.is_empty() => Key::Tuple(Rc::from(vec![
            Key::Ptr(Rc::as_ptr(&v.ty) as *const u8 as usize),
            Key::Int(v.index as i64),
        ])),
        Value::Variant(v) => {
            let mut parts = vec![
                Key::Ptr(Rc::as_ptr(&v.ty) as *const u8 as usize),
                Key::Int(v.index as i64),
            ];
            for f in &v.fields {
                parts.push(key_of(f)?);
            }
            Key::Tuple(parts.into())
        }
        Value::Range(r) => Key::Tuple(Rc::from(vec![
            Key::Int(r.start),
            Key::Int(r.end),
            Key::Int(r.step),
        ])),
        other => Key::Ptr(ptr_of(other)),
    })
}

/// Address of a reference value (for identity).
pub fn ptr_of(v: &Value) -> usize {
    match v {
        Value::Str(s) => Rc::as_ptr(s) as *const u8 as usize,
        Value::List(l) => Rc::as_ptr(l) as *const u8 as usize,
        Value::Tuple(t) => Rc::as_ptr(t) as *const u8 as usize,
        Value::Map(m) | Value::Set(m) => Rc::as_ptr(m) as *const u8 as usize,
        Value::Range(r) => Rc::as_ptr(r) as *const u8 as usize,
        Value::Func(f) => Rc::as_ptr(f) as *const u8 as usize,
        Value::Native(f) => Rc::as_ptr(f) as *const u8 as usize,
        Value::Bound(b) => Rc::as_ptr(b) as *const u8 as usize,
        Value::Class(c) => Rc::as_ptr(c) as *const u8 as usize,
        Value::Instance(i) => Rc::as_ptr(i) as *const u8 as usize,
        Value::Enum(e) => Rc::as_ptr(e) as *const u8 as usize,
        Value::Variant(v) => Rc::as_ptr(v) as *const u8 as usize,
        Value::Module(m) => Rc::as_ptr(m) as *const u8 as usize,
        Value::Big(b) => Rc::as_ptr(b) as *const u8 as usize,
        Value::Tensor(t) => t.ptr(),
        Value::Dual(d) => Rc::as_ptr(d) as *const u8 as usize,
        Value::Object(o) => Rc::as_ptr(o) as *const u8 as usize,
        Value::Type(t) => *t as usize,
        Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) => 0,
    }
}

/// An insertion-ordered hash map from values to values.
#[derive(Clone, Default)]
pub struct Dict {
    entries: Vec<Option<(Value, Value)>>,
    index: HashMap<Key, usize>,
}

impl fmt::Debug for Dict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.iter().map(|(k, v)| (k.clone(), v.clone())))
            .finish()
    }
}

impl Dict {
    /// An empty dictionary.
    pub fn new() -> Dict {
        Dict::default()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Looks up a value by key.
    pub fn get(&self, key: &Value) -> Option<Value> {
        let k = key_of(key)?;
        self.get_key(&k)
    }

    /// Looks up by precomputed key.
    pub fn get_key(&self, k: &Key) -> Option<Value> {
        let &i = self.index.get(k)?;
        self.entries[i].as_ref().map(|(_, v)| v.clone())
    }

    /// Looks up by string key.
    pub fn get_str(&self, s: &str) -> Option<Value> {
        self.get_key(&Key::Str(Rc::from(s)))
    }

    /// Whether `key` is present.
    pub fn contains(&self, key: &Value) -> bool {
        key_of(key).is_some_and(|k| self.index.contains_key(&k))
    }

    /// Inserts or replaces; returns `false` if the key is unhashable.
    pub fn insert(&mut self, key: Value, value: Value) -> bool {
        let Some(k) = key_of(&key) else {
            return false;
        };
        match self.index.get(&k) {
            Some(&i) => {
                if let Some(entry) = self.entries[i].as_mut() {
                    entry.1 = value;
                }
            }
            None => {
                self.index.insert(k, self.entries.len());
                self.entries.push(Some((key, value)));
            }
        }
        true
    }

    /// Removes a key, returning its value.
    pub fn remove(&mut self, key: &Value) -> Option<Value> {
        let k = key_of(key)?;
        let i = self.index.remove(&k)?;
        let old = self.entries[i].take().map(|(_, v)| v);
        if self.entries.len() > 32 && self.index.len() * 2 < self.entries.len() {
            self.compact();
        }
        old
    }

    fn compact(&mut self) {
        let entries: Vec<_> = self.entries.drain(..).flatten().collect();
        self.index.clear();
        for (i, (k, _)) in entries.iter().enumerate() {
            if let Some(key) = key_of(k) {
                self.index.insert(key, i);
            }
        }
        self.entries = entries.into_iter().map(Some).collect();
    }

    /// Iterates entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.entries.iter().flatten().map(|(k, v)| (k, v))
    }

    /// Keys in insertion order.
    pub fn keys(&self) -> Vec<Value> {
        self.iter().map(|(k, _)| k.clone()).collect()
    }

    /// Values in insertion order.
    pub fn values(&self) -> Vec<Value> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    /// Removes everything.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
    }
}

impl Value {
    /// Makes a string value.
    pub fn str(s: &str) -> Value {
        Value::Str(Rc::from(s))
    }

    /// Makes a list value.
    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(RefCell::new(items)))
    }

    /// Makes a tuple value.
    pub fn tuple(items: Vec<Value>) -> Value {
        Value::Tuple(Rc::from(items))
    }

    /// Makes a map from string keys.
    pub fn map_from(entries: Vec<(&str, Value)>) -> Value {
        let mut d = Dict::new();
        for (k, v) in entries {
            d.insert(Value::str(k), v);
        }
        Value::Map(Rc::new(RefCell::new(d)))
    }

    /// Makes a native function.
    pub fn native(name: &str, f: impl Fn(&mut Interp, Args) -> R + 'static) -> Value {
        Value::Native(Rc::new(NativeFn {
            name: Rc::from(name),
            f: Box::new(f),
        }))
    }

    /// Normalises a big integer back to `Int` when it fits.
    pub fn from_big(b: BigInt) -> Value {
        match b.to_i64() {
            Some(i) => Value::Int(i),
            None => Value::Big(Rc::new(b)),
        }
    }

    /// Python-style truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Big(b) => !b.is_zero(),
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.borrow().is_empty(),
            Value::Tuple(t) => !t.is_empty(),
            Value::Map(m) | Value::Set(m) => !m.borrow().is_empty(),
            Value::Range(r) => !r.is_empty(),
            Value::Tensor(t) => t.truthy(),
            Value::Dual(d) => d.re.truthy(),
            _ => true,
        }
    }

    /// The value's type name, as `type(x)` reports it.
    pub fn type_name(&self) -> String {
        match self {
            Value::Nil => "nil".into(),
            Value::Bool(_) => "bool".into(),
            Value::Int(_) | Value::Big(_) => "int".into(),
            Value::Float(_) => "float".into(),
            Value::Str(_) => "str".into(),
            Value::List(_) => "list".into(),
            Value::Tuple(_) => "tuple".into(),
            Value::Map(_) => "dict".into(),
            Value::Set(_) => "set".into(),
            Value::Range(_) => "range".into(),
            Value::Func(_) | Value::Native(_) | Value::Bound(_) => "function".into(),
            Value::Class(_) | Value::Type(_) | Value::Enum(_) => "type".into(),
            Value::Instance(i) => i.class.name.as_str().to_string(),
            Value::Variant(v) => v.ty.name.as_str().to_string(),
            Value::Module(_) => "module".into(),
            Value::Tensor(_) => "tensor".into(),
            Value::Dual(_) => "dual".into(),
            Value::Object(o) => o.type_name().to_owned(),
        }
    }

    /// The integer value, if this is an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e18 => Some(*f as i64),
            _ => None,
        }
    }

    /// The numeric value as a float, if numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Big(b) => Some(b.to_f64()),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            Value::Tensor(t) if t.numel() == 1 => Some(t.item()),
            Value::Dual(d) => d.re.as_f64(),
            _ => None,
        }
    }

    /// Whether the value can be called.
    pub fn is_callable(&self) -> bool {
        matches!(
            self,
            Value::Func(_) | Value::Native(_) | Value::Bound(_) | Value::Class(_) | Value::Type(_)
        ) || matches!(self, Value::Object(_))
    }

    /// Whether the value is a number (int, big int or float).
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Int(_) | Value::Float(_) | Value::Big(_))
    }
}
