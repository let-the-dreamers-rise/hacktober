//! The core abstract syntax tree.
//!
//! Every surface dialect Chimera accepts — Python, JavaScript, C, Java, Go,
//! Rust, Ruby, Lua, OCaml/Haskell, SQL, English — is parsed into this one
//! small tree. The interpreter, the `explain` command and the dialect
//! converter only ever see this form, which is what makes "write it in any
//! syntax" tractable: the parser is wide, the core is narrow.

use std::rc::Rc;

use chimera_core::Span;

use crate::bigint::BigInt;
use crate::symbol::Sym;

/// Index of a source file registered with the engine.
pub type FileId = u32;

/// A sequence of statements. The value of a block is the value of its last
/// expression statement (Ruby/Rust semantics).
#[derive(Clone, Debug, Default)]
pub struct Block {
    /// The statements.
    pub stmts: Vec<Stmt>,
    /// Whether the block directly declares variables (`let`, `var`, …) and so
    /// needs its own scope.
    pub scoped: bool,
}

impl Block {
    /// A block holding a single expression.
    pub fn of_expr(e: Expr) -> Block {
        let span = e.span;
        Block {
            stmts: vec![Stmt {
                kind: StmtKind::Expr(e),
                span,
            }],
            scoped: false,
        }
    }

    /// Builds a block, computing whether it needs a scope.
    pub fn new(stmts: Vec<Stmt>) -> Block {
        let scoped = stmts.iter().any(|s| matches!(s.kind, StmtKind::Let { .. }));
        Block { stmts, scoped }
    }
}

/// A statement.
#[derive(Clone, Debug)]
pub struct Stmt {
    /// What kind of statement.
    pub kind: StmtKind,
    /// Source range.
    pub span: Span,
}

/// Statement kinds.
#[derive(Clone, Debug)]
pub enum StmtKind {
    /// An expression evaluated for its value or effect.
    Expr(Expr),
    /// `let`/`var`/`const`/`int x = …`/… declaration.
    Let {
        /// What to bind.
        pattern: Pattern,
        /// Optional annotation, checked at runtime.
        ty: Option<TypeExpr>,
        /// Initialiser (`nil` when absent).
        value: Option<Expr>,
    },
    /// A named function declaration (hoisted to the start of its scope).
    Fn(Rc<FnDef>),
    /// A class/struct/record/data class.
    Class(Rc<ClassDef>),
    /// Methods added to an existing type (`impl`, Go receivers, `extension`).
    Impl {
        /// The type being extended.
        target: Expr,
        /// Instance methods.
        methods: Vec<Rc<FnDef>>,
        /// Methods without a receiver.
        statics: Vec<Rc<FnDef>>,
    },
    /// An enum or algebraic data type.
    Enum(Rc<EnumDef>),
    /// `while`, `until`, and `do … while`.
    While {
        /// Loop condition.
        cond: Expr,
        /// Loop body.
        body: Block,
        /// `until`: loop while the condition is false.
        negate: bool,
        /// `do … while`: test after the body.
        post: bool,
    },
    /// `loop { … }` / `forever`.
    Loop(Block),
    /// `repeat N times` / `N.times`.
    Repeat {
        /// Number of iterations.
        count: Expr,
        /// Loop body.
        body: Block,
    },
    /// `for x in xs`, `for (const x of xs)`, `for i = 1, 10`, …
    For {
        /// Binding for each element.
        pattern: Pattern,
        /// The iterable.
        iter: Expr,
        /// Loop body.
        body: Block,
        /// Whether each iteration needs a fresh scope (closures capture it).
        fresh_scope: bool,
    },
    /// C-style `for (init; cond; step)`.
    CFor {
        /// Initialiser.
        init: Option<Box<Stmt>>,
        /// Condition (absent = forever).
        cond: Option<Expr>,
        /// Step expression.
        step: Option<Expr>,
        /// Loop body.
        body: Block,
    },
    /// `return`, `give back`.
    Return(Option<Expr>),
    /// `break` / `last`.
    Break,
    /// `continue` / `next`.
    Continue,
    /// `throw`/`raise`.
    Throw(Expr),
    /// `try`/`begin` with handlers.
    Try {
        /// Protected block.
        body: Block,
        /// Handlers in order.
        catches: Vec<Catch>,
        /// Python's `try … else`.
        else_block: Option<Block>,
        /// `finally`/`ensure`.
        finally: Option<Block>,
    },
    /// C/Java/Go `switch`; `break` inside exits the switch.
    Switch {
        /// Value switched on.
        subject: Expr,
        /// Cases in order; the first match runs.
        cases: Vec<SwitchCase>,
        /// `default:` body.
        default: Option<Block>,
    },
    /// `import`, `from … import`, `use`, `require`, `#include`-likes.
    Import(Import),
    /// Go/Swift `defer`: runs when the enclosing function exits.
    Defer(Block),
    /// `del x`, `del d[k]`.
    Delete(Expr),
    /// `model Name: dense 2 -> 8, relu …`.
    Model(Rc<ModelDef>),
    /// `train net on X, y for 100 epochs using adam …`.
    Train(Rc<TrainSpec>),
    /// `pass`, `...`, ignored foreign declarations.
    Pass,
}

/// One `catch`/`except`/`rescue` clause.
#[derive(Clone, Debug)]
pub struct Catch {
    /// Exception classes to match (empty = catch everything).
    pub classes: Vec<Expr>,
    /// Name bound to the exception.
    pub binding: Option<Sym>,
    /// Handler body.
    pub body: Block,
}

/// One `case` of a C-style switch.
#[derive(Clone, Debug)]
pub struct SwitchCase {
    /// Values compared against the subject (case equality).
    pub values: Vec<Expr>,
    /// Statements to run.
    pub body: Block,
}

/// An import statement.
#[derive(Clone, Debug)]
pub struct Import {
    /// Module path segments (`a.b.c`, `a::b`), or a file path.
    pub path: Vec<Sym>,
    /// Explicit file path (`import "utils.chi"`).
    pub file: Option<String>,
    /// `as` alias for the module.
    pub alias: Option<Sym>,
    /// `from m import a, b as c` — `None` imports the module itself.
    pub names: Option<Vec<(Sym, Option<Sym>)>>,
    /// `from m import *`.
    pub wildcard: bool,
}

/// An expression.
#[derive(Clone, Debug)]
pub struct Expr {
    /// What kind of expression.
    pub kind: ExprKind,
    /// Source range.
    pub span: Span,
}

impl Expr {
    /// Wraps a kind with a span.
    pub fn new(kind: ExprKind, span: Span) -> Expr {
        Expr { kind, span }
    }

    /// Whether this expression can be assigned to.
    pub fn is_assignable(&self) -> bool {
        match &self.kind {
            ExprKind::Name(_) | ExprKind::Member { .. } | ExprKind::Index { .. } => true,
            ExprKind::Tuple(items) | ExprKind::List(items) => items.iter().all(|e| {
                e.is_assignable()
                    || matches!(&e.kind, ExprKind::Spread(inner) if inner.is_assignable())
            }),
            _ => false,
        }
    }
}

/// Expression kinds.
#[derive(Clone, Debug)]
pub enum ExprKind {
    /// `nil`, `null`, `None`, `undefined`.
    Nil,
    /// Booleans.
    Bool(bool),
    /// Machine integer literal.
    Int(i64),
    /// Big integer literal.
    BigInt(Rc<BigInt>),
    /// Float literal.
    Float(f64),
    /// String literal without interpolation.
    Str(Rc<str>),
    /// Interpolated string.
    Interp(Vec<InterpPart>),
    /// Variable reference.
    Name(Sym),
    /// `self` / `this`.
    SelfRef,
    /// `super`.
    Super,
    /// List literal (elements may be [`ExprKind::Spread`]).
    List(Vec<Expr>),
    /// Tuple literal.
    Tuple(Vec<Expr>),
    /// Map literal; a `Spread` key merges another map.
    Map(Vec<(Expr, Expr)>),
    /// Set literal `{1, 2}`.
    Set(Vec<Expr>),
    /// `...xs` / `*xs` inside a literal or call.
    Spread(Box<Expr>),
    /// `a..b`, `a..=b`, `a to b by s`.
    Range {
        /// Start (default 0).
        start: Option<Box<Expr>>,
        /// End (absent = unbounded, only for slicing).
        end: Option<Box<Expr>>,
        /// Whether `end` is included.
        inclusive: bool,
        /// Step.
        step: Option<Box<Expr>>,
    },
    /// Prefix operator.
    Unary(UnOp, Box<Expr>),
    /// Arithmetic/bitwise binary operator.
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Chained comparison `a < b <= c`.
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    /// Short-circuit `and`.
    And(Box<Expr>, Box<Expr>),
    /// Short-circuit `or`.
    Or(Box<Expr>, Box<Expr>),
    /// `a ?? b`: `a` unless it is nil.
    Coalesce(Box<Expr>, Box<Expr>),
    /// Function/method call.
    Call {
        /// What is called.
        callee: Box<Expr>,
        /// Arguments.
        args: Vec<Arg>,
    },
    /// `obj.name`, `obj?.name`, `Mod::name`.
    Member {
        /// Receiver.
        object: Box<Expr>,
        /// Attribute name.
        name: Sym,
        /// `?.`: yield nil when the receiver is nil.
        optional: bool,
    },
    /// `obj[index]` (index may be a [`ExprKind::Slice`] or a tuple of them).
    Index {
        /// Receiver.
        object: Box<Expr>,
        /// Index expression.
        index: Box<Expr>,
    },
    /// `start:end:step` inside an index.
    Slice {
        /// Start.
        start: Option<Box<Expr>>,
        /// End.
        end: Option<Box<Expr>>,
        /// Step.
        step: Option<Box<Expr>>,
    },
    /// Anonymous function.
    Lambda(Rc<FnDef>),
    /// `if`/`unless`/ternary/`a if c else b`.
    If {
        /// Condition.
        cond: Box<Expr>,
        /// Taken branch.
        then: Block,
        /// Other branch.
        otherwise: Option<Block>,
    },
    /// `match`, `case … of`, `when`, `case/when`.
    Match {
        /// Value matched (absent for Kotlin's subject-less `when`).
        subject: Option<Box<Expr>>,
        /// Arms in order.
        arms: Vec<MatchArm>,
    },
    /// A block used as an expression (`do … end`, `begin … end`, `{ … }`).
    Block(Block),
    /// List/map/set comprehension.
    Comprehension(Box<Comprehension>),
    /// `target = value` (also `:=`, `<-`).
    Assign {
        /// What is assigned (name, member, index or destructuring).
        target: Box<Expr>,
        /// New value.
        value: Box<Expr>,
    },
    /// `target op= value`.
    CompoundAssign {
        /// Operator.
        op: BinOp,
        /// Target.
        target: Box<Expr>,
        /// Right-hand side.
        value: Box<Expr>,
    },
    /// `x++`, `--x`.
    IncDec {
        /// Target.
        target: Box<Expr>,
        /// +1 or -1.
        delta: i64,
        /// Whether the new value is returned (prefix form).
        prefix: bool,
    },
    /// SQL / LINQ query or DML statement.
    Query(Rc<Query>),
    /// `f′`, `f″`: derivative via forward-mode automatic differentiation.
    Derivative {
        /// The function differentiated.
        func: Box<Expr>,
        /// 1 for `′`, 2 for `″`.
        order: u32,
    },
    /// `∇f`: gradient via reverse-mode automatic differentiation.
    Gradient(Box<Expr>),
    /// Struct literal `Point { x: 1, y: 2 }`.
    StructLit {
        /// The type.
        ty: Box<Expr>,
        /// Field initialisers.
        fields: Vec<(Sym, Expr)>,
    },
    /// A natural-language request: `please sort xs descending`.
    Please(Rc<str>),
    /// `let pattern = value` as a condition (`if let Some(x) = f()`,
    /// `while let`, Swift `if let x = optional`): binds on success.
    Let {
        /// The pattern tried.
        pattern: Box<Pattern>,
        /// The value matched.
        value: Box<Expr>,
    },
    /// `yield`, `yield a, b`, `yield from xs` / `yield* xs`.
    Yield {
        /// Yielded values (several for Ruby's `yield a, b`).
        values: Vec<Expr>,
        /// `yield from` / `yield*`: delegate to another iterable.
        delegate: bool,
    },
    /// A jump used as an expression: `x ?? throw Error()`,
    /// `None => return 0`, `Err(e) => raise e`.
    Jump(Box<Stmt>),
}

/// Prefix operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// `-x`
    Neg,
    /// `+x`
    Pos,
    /// `not x`, `!x`
    Not,
    /// `~x`
    BitNot,
    /// `√x`
    Sqrt,
    /// `typeof x`
    TypeOf,
    /// Postfix `x!` (factorial for integers, force-unwrap otherwise).
    Bang,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (true division)
    Div,
    /// `div` (floor division)
    FloorDiv,
    /// `%`, `mod`
    Mod,
    /// `**`, `^`
    Pow,
    /// `@` (matrix multiplication)
    MatMul,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `xor`
    BitXor,
    /// `<<` (also stream output)
    Shl,
    /// `>>`
    Shr,
    /// `++` (concatenation)
    Concat,
    /// `|>`: `x |> f` is `f(x)`
    Pipe,
}

/// Comparison operators (chainable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// `==`, `equals`, `is equal to`
    Eq,
    /// `!=`, `<>`
    Ne,
    /// `===` (same type and value)
    StrictEq,
    /// `!==`
    StrictNe,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `in`, `∈`
    In,
    /// `not in`
    NotIn,
    /// `is` (identity, or instance-of when the right side is a type)
    Is,
    /// `is not`
    IsNot,
    /// `≈` (approximately equal)
    Approx,
    /// `<=>` (three-way comparison; not chainable)
    Spaceship,
}

/// A call argument.
#[derive(Clone, Debug)]
pub enum Arg {
    /// Positional.
    Pos(Expr),
    /// `name: value` / `name = value`.
    Named(Sym, Expr),
    /// `*xs` / `...xs`.
    Spread(Expr),
    /// `**kwargs`.
    KwSpread(Expr),
}

/// A piece of an interpolated string.
#[derive(Clone, Debug)]
pub enum InterpPart {
    /// Literal text.
    Lit(Rc<str>),
    /// Interpolated expression with an optional format spec.
    Expr(Expr, Option<Rc<str>>),
}

/// A function definition (named, method or lambda).
#[derive(Clone, Debug)]
pub struct FnDef {
    /// Name, if any.
    pub name: Option<Sym>,
    /// Parameters (excluding an explicit receiver).
    pub params: Vec<Param>,
    /// Body; the last expression is the implicit return value.
    pub body: Block,
    /// Source range.
    pub span: Span,
    /// File the function was defined in.
    pub file: FileId,
    /// Declared return type, checked at runtime.
    pub ret: Option<TypeExpr>,
    /// The explicit receiver name (`self`, `this`, Go's `p`), if declared.
    pub receiver: Option<Sym>,
    /// Whether the function is a static/class method.
    pub is_static: bool,
    /// Block lambda with implicit `it`/`$0` parameters.
    pub implicit_params: bool,
    /// Decorators applied (innermost last).
    pub decorators: Vec<Expr>,
    /// Declared without a parameter list (Ruby `def speak`): referencing it
    /// calls it, so `dog.speak` works without parentheses.
    pub bare: bool,
    /// Source offsets of the `yield`s that belong to this function. A
    /// function with yields is a generator (or, called with a trailing
    /// block, a Ruby method that yields to it).
    pub yields: Vec<usize>,
}

impl FnDef {
    /// Whether the function contains `yield`.
    pub fn is_generator(&self) -> bool {
        !self.yields.is_empty()
    }

    /// Whether a statement of this function contains one of its yields.
    pub fn yields_in(&self, span: Span) -> bool {
        self.yields.iter().any(|&o| o >= span.start && o < span.end)
    }
}

/// A function parameter.
#[derive(Clone, Debug)]
pub struct Param {
    /// Binding pattern (usually a plain name).
    pub pattern: Pattern,
    /// Default value.
    pub default: Option<Expr>,
    /// Declared type, checked at runtime.
    pub ty: Option<TypeExpr>,
    /// Ordinary, `*rest` or `**kwargs`.
    pub kind: ParamKind,
}

impl Param {
    /// The bound name when the pattern is a plain binding.
    pub fn name(&self) -> Option<Sym> {
        match &self.pattern {
            Pattern::Bind(s) => Some(*s),
            _ => None,
        }
    }
}

/// Parameter kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Ordinary.
    Normal,
    /// Collects remaining positional arguments.
    Rest,
    /// Collects remaining named arguments.
    KwRest,
}

/// A class/struct/record declaration.
#[derive(Clone, Debug)]
pub struct ClassDef {
    /// Class name.
    pub name: Sym,
    /// Superclass expression.
    pub superclass: Option<Expr>,
    /// Instance methods (constructors included under their own names).
    pub methods: Vec<Rc<FnDef>>,
    /// Static/class methods.
    pub statics: Vec<Rc<FnDef>>,
    /// Declared instance fields with defaults.
    pub fields: Vec<FieldDef>,
    /// Class-level attributes (Python class variables).
    pub class_attrs: Vec<(Sym, Expr)>,
    /// Whether a positional/named constructor is generated from `fields`.
    pub auto_init: bool,
    /// Source range.
    pub span: Span,
}

/// A declared field.
#[derive(Clone, Debug)]
pub struct FieldDef {
    /// Field name.
    pub name: Sym,
    /// Default value.
    pub default: Option<Expr>,
    /// Declared type.
    pub ty: Option<TypeExpr>,
}

/// An enum / algebraic data type.
#[derive(Clone, Debug)]
pub struct EnumDef {
    /// Type name.
    pub name: Sym,
    /// Variants in declaration order.
    pub variants: Vec<VariantDef>,
    /// Methods declared inside the enum body.
    pub methods: Vec<Rc<FnDef>>,
    /// Source range.
    pub span: Span,
}

/// One variant of an enum.
#[derive(Clone, Debug)]
pub struct VariantDef {
    /// Variant name.
    pub name: Sym,
    /// Number of payload fields.
    pub arity: usize,
    /// Field names for struct-like variants.
    pub field_names: Vec<Sym>,
    /// Explicit discriminant (`Green = 2`).
    pub value: Option<Expr>,
}

/// One arm of a `match`.
#[derive(Clone, Debug)]
pub struct MatchArm {
    /// Pattern tested against the subject.
    pub pattern: Pattern,
    /// Extra condition.
    pub guard: Option<Expr>,
    /// Result.
    pub body: Block,
}

/// Patterns for `match`, destructuring `let`, `for` and parameters.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// `_`, `else`, `default`.
    Wildcard,
    /// Binds the value to a name.
    Bind(Sym),
    /// Matches by equality with a constant.
    Literal(Expr),
    /// `(a, b)`.
    Tuple(Vec<Pattern>),
    /// `[a, b, ...rest]`.
    List {
        /// Patterns before (and after) the rest marker.
        items: Vec<Pattern>,
        /// Position of the rest marker and its optional binding.
        rest: Option<(usize, Option<Sym>)>,
    },
    /// `{name: n, age, ...rest}`.
    Map(Vec<(Expr, Pattern)>, Option<Sym>),
    /// `x = default` inside a destructuring pattern: used when the value
    /// is missing or nil.
    Default(Box<Pattern>, Expr),
    /// `Some(x)`, `Point(x, y)`, `Shape::Circle(r)`, `Rect { w, h }`, `Circle r`.
    Ctor {
        /// Constructor path (last segment is the variant/class name).
        path: Vec<Sym>,
        /// Positional sub-patterns.
        args: Vec<Pattern>,
        /// Named sub-patterns.
        fields: Vec<(Sym, Pattern)>,
        /// Whether the pattern had an argument list at all.
        has_args: bool,
    },
    /// `p1 | p2`.
    Or(Vec<Pattern>),
    /// `1..5`, `'a'..='z'`.
    Range {
        /// Low bound.
        lo: Expr,
        /// High bound.
        hi: Expr,
        /// Whether `hi` is included.
        inclusive: bool,
    },
    /// Ruby/Kotlin/C case-equality: equality, range membership or type test.
    CaseEq(Expr),
    /// Kotlin `in collection`.
    In(Expr),
    /// `is Type` / `x: Type`.
    Type(TypeExpr, Option<Box<Pattern>>),
    /// `pat as name`, `name @ pat`.
    As(Box<Pattern>, Sym),
}

/// A (runtime-checked) type annotation.
#[derive(Clone, Debug)]
pub enum TypeExpr {
    /// A named type with optional generic arguments.
    Named(Sym, Vec<TypeExpr>),
    /// `T?`, `Optional[T]`, `Option<T>`.
    Optional(Box<TypeExpr>),
    /// `A | B`.
    Union(Vec<TypeExpr>),
    /// `T[]`, `[T]`.
    List(Box<TypeExpr>),
    /// `(A, B)`.
    Tuple(Vec<TypeExpr>),
    /// Anything (also used for unparseable foreign types).
    Any,
}

/// A comprehension.
#[derive(Clone, Debug)]
pub struct Comprehension {
    /// What is built.
    pub kind: CompKind,
    /// Element expression (key for maps).
    pub element: Expr,
    /// Value expression for map comprehensions.
    pub value: Option<Expr>,
    /// `for`/`if` clauses, outermost first.
    pub clauses: Vec<CompClause>,
}

/// Comprehension result kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompKind {
    /// `[ … ]`
    List,
    /// `{k: v …}`
    Map,
    /// `{x …}`
    Set,
}

/// A comprehension clause.
#[derive(Clone, Debug)]
pub enum CompClause {
    /// `for pat in iter` / `pat <- iter`.
    For(Pattern, Expr),
    /// `if cond` / `, cond` (Haskell guard).
    If(Expr),
}

/// SQL / LINQ operations over in-memory collections.
#[derive(Clone, Debug)]
pub enum Query {
    /// `SELECT …` / `from x in xs … select …`.
    Select(SelectQuery),
    /// `INSERT INTO t [(cols)] VALUES (…), (…)`.
    Insert {
        /// Target collection.
        target: Expr,
        /// Column names (absent = positional into existing keys / values).
        columns: Vec<Sym>,
        /// Rows of values.
        rows: Vec<Vec<Expr>>,
    },
    /// `UPDATE t SET a = e WHERE …`.
    Update {
        /// Target collection.
        target: Expr,
        /// Alias for rows.
        alias: Option<Sym>,
        /// Assignments.
        sets: Vec<(Sym, Expr)>,
        /// Row filter.
        filter: Option<Expr>,
    },
    /// `DELETE FROM t WHERE …`.
    Delete {
        /// Target collection.
        target: Expr,
        /// Alias for rows.
        alias: Option<Sym>,
        /// Row filter.
        filter: Option<Expr>,
    },
}

/// A `SELECT` query.
#[derive(Clone, Debug, Default)]
pub struct SelectQuery {
    /// Output columns (empty with `star` = all columns).
    pub columns: Vec<SelectItem>,
    /// `SELECT DISTINCT`.
    pub distinct: bool,
    /// Data sources joined left to right.
    pub sources: Vec<Source>,
    /// `WHERE`.
    pub filter: Option<Expr>,
    /// `GROUP BY`.
    pub group_by: Vec<Expr>,
    /// `HAVING`.
    pub having: Option<Expr>,
    /// `ORDER BY expr [DESC]`.
    pub order_by: Vec<(Expr, bool)>,
    /// `LIMIT`.
    pub limit: Option<Expr>,
    /// `OFFSET`.
    pub offset: Option<Expr>,
    /// LINQ `select expr`: produce bare values instead of rows.
    pub value: Option<Expr>,
}

/// One output column of a `SELECT`.
#[derive(Clone, Debug)]
pub enum SelectItem {
    /// `*` or `alias.*`.
    Star(Option<Sym>),
    /// `expr [AS alias]`.
    Expr(Expr, Option<Sym>),
}

/// A `FROM`/`JOIN` source.
#[derive(Clone, Debug)]
pub struct Source {
    /// Collection expression.
    pub expr: Expr,
    /// Row alias (defaults to the collection's variable name).
    pub alias: Option<Sym>,
    /// How it joins to the sources before it.
    pub join: JoinKind,
    /// Join condition.
    pub on: Option<Expr>,
}

/// Join kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinKind {
    /// First source / comma / `CROSS JOIN`.
    Cross,
    /// `[INNER] JOIN … ON`.
    Inner,
    /// `LEFT [OUTER] JOIN … ON`.
    Left,
}

/// A neural-network model declaration.
#[derive(Clone, Debug)]
pub struct ModelDef {
    /// Model (class) name.
    pub name: Sym,
    /// Layers in order.
    pub layers: Vec<LayerSpec>,
    /// Source range.
    pub span: Span,
}

/// One layer line: `dense 2 -> 8, relu` / `dropout 0.1` / `relu`.
#[derive(Clone, Debug)]
pub struct LayerSpec {
    /// Layer kind (`dense`, `linear`, `relu`, …).
    pub kind: Sym,
    /// Arguments.
    pub args: Vec<Arg>,
    /// Source range.
    pub span: Span,
}

/// A `train` statement.
#[derive(Clone, Debug)]
pub struct TrainSpec {
    /// The model.
    pub model: Expr,
    /// Inputs.
    pub inputs: Expr,
    /// Targets.
    pub targets: Expr,
    /// Extra options (`epochs`, `optimizer`, `loss`, `lr`, `report`, `batch`).
    pub options: Vec<(Sym, Expr)>,
    /// Source range.
    pub span: Span,
}
