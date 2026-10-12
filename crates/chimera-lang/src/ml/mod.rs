//! Machine learning built into the language: tensors with autograd,
//! forward-mode derivatives (`f′`), gradients (`∇f`), neural-network layers,
//! optimizers, losses, the `model`/`train` statements, and NumPy / PyTorch /
//! Keras / scikit-learn flavoured APIs over the same engine.

pub mod autodiff;
pub mod dual;
pub mod nn;
pub mod tensor;

use std::rc::Rc;

use chimera_core::Span;

use crate::builtins::objects::{builtin, module as make_module};
use crate::builtins::{arg, constant, def, int_arg};
use crate::interp::{Args, Interp, sym};
use crate::value::*;
use tensor::{Tensor, Unary};

/// `tensor(data, requires_grad=…)` / `np.array(data)`.
pub fn make_tensor(interp: &mut Interp, args: Args) -> R {
    let data = args.get(0).cloned().unwrap_or(Value::list(Vec::new()));
    let t = interp.to_tensor(&data, args.span)?.detach();
    let grad = args
        .named("requires_grad")
        .or_else(|| args.named("grad"))
        .or_else(|| args.named("param"))
        .is_some_and(|v| v.truthy());
    if grad {
        t.set_requires_grad(true);
    }
    Ok(Value::Tensor(t))
}

/// Converts a tensor to nested lists.
pub fn tensor_to_list(t: &Tensor) -> Value {
    fn build(d: &[f64], shape: &[usize], pos: &mut usize) -> Value {
        if shape.is_empty() {
            let v = Value::Float(d[*pos]);
            *pos += 1;
            return v;
        }
        let mut out = Vec::with_capacity(shape[0]);
        for _ in 0..shape[0] {
            out.push(build(d, &shape[1..], pos));
        }
        Value::list(out)
    }
    let d = t.data();
    if t.ndim() == 0 {
        return Value::Float(d.first().copied().unwrap_or(0.0));
    }
    build(&d, t.shape(), &mut 0)
}

fn shape_args(interp: &mut Interp, args: &Args, name: &str) -> R<Vec<usize>> {
    let mut dims = Vec::new();
    for v in &args.pos {
        match v {
            Value::Tuple(_) | Value::List(_) => {
                for d in interp.collect(v, args.span)? {
                    dims.push(d.as_int().unwrap_or(0).max(0) as usize);
                }
            }
            other => match other.as_int() {
                Some(d) => dims.push(d.max(0) as usize),
                None => {
                    return interp.type_err(format!("`{name}` needs integer dimensions"), args.span);
                }
            },
        }
    }
    if dims.iter().product::<usize>() > 50_000_000 {
        return interp.value_err(
            format!("`{name}` would allocate too many elements"),
            args.span,
        );
    }
    Ok(dims)
}

fn terr<T>(interp: &mut Interp, e: String, span: Span) -> R<T> {
    Err(interp.error("ValueError", e, span))
}

fn axis_arg(args: &Args, i: usize) -> Option<i64> {
    args.arg(i, "axis")
        .or_else(|| args.named("dim"))
        .and_then(|v| v.as_int())
}

/// Properties of tensors (`t.shape`, `t.T`, `t.grad`, …).
pub fn tensor_property(
    interp: &mut Interp,
    t: &Tensor,
    name: &str,
    span: Span,
) -> R<Option<Value>> {
    let _ = (interp, span);
    Ok(Some(match name {
        "shape" | "size_tuple" => {
            Value::tuple(t.shape().iter().map(|&d| Value::Int(d as i64)).collect())
        }
        "ndim" | "rank" | "dim" => Value::Int(t.ndim() as i64),
        "size" | "numel" | "length" | "len" => Value::Int(t.numel() as i64),
        "T" | "mT" => Value::Tensor(t.transpose()),
        "grad" => t.grad().map_or(Value::Nil, Value::Tensor),
        "requires_grad" => Value::Bool(t.requires_grad()),
        "data" => Value::Tensor(t.detach()),
        "dtype" => Value::str("float64"),
        "is_leaf" => Value::Bool(true),
        _ => return Ok(None),
    }))
}

/// Methods of tensors.
pub fn tensor_method(interp: &mut Interp, t: &Tensor, name: &str, args: &Args) -> R<Option<Value>> {
    let span = args.span;
    let wrap = |interp: &mut Interp, r: Result<Tensor, String>| -> R {
        r.map(Value::Tensor)
            .map_err(|e| interp.error("ValueError", e, span))
    };
    let other = |interp: &mut Interp| -> R<Tensor> {
        let v = arg(interp, args, 0, name)?;
        interp.to_tensor(&v, span)
    };
    let out = match name {
        "item" | "to_f" | "__float__" => {
            if t.numel() != 1 {
                return terr(
                    interp,
                    format!(
                        "item() needs a single-element tensor, this one has {}",
                        t.numel()
                    ),
                    span,
                );
            }
            Value::Float(t.item())
        }
        "tolist" | "to_list" | "to_a" | "numpy" | "cpu" | "to_vec" => {
            if name == "numpy" || name == "cpu" {
                Value::Tensor(t.clone())
            } else {
                tensor_to_list(t)
            }
        }
        "backward" => {
            if t.numel() != 1 {
                return terr(
                    interp,
                    "backward() needs a scalar (one-element) tensor such as a loss".into(),
                    span,
                );
            }
            t.backward();
            Value::Nil
        }
        "zero_grad" | "zero_grad_" => {
            t.zero_grad();
            Value::Nil
        }
        "detach" | "clone" | "copy" => Value::Tensor(t.detach()),
        "requires_grad_" => {
            t.set_requires_grad(args.get(0).is_none_or(Value::truthy));
            Value::Tensor(t.clone())
        }
        "reshape" | "view" | "resize" => {
            let mut dims = Vec::new();
            for v in &args.pos {
                match v {
                    Value::Tuple(_) | Value::List(_) => {
                        for d in interp.collect(v, span)? {
                            dims.push(d.as_int().unwrap_or(-1));
                        }
                    }
                    other => dims.push(other.as_int().unwrap_or(-1)),
                }
            }
            wrap(interp, t.reshape(&dims))?
        }
        "flatten" | "ravel" => wrap(interp, t.reshape(&[-1]))?,
        "transpose" | "t" | "permute" => Value::Tensor(t.transpose()),
        "unsqueeze" | "expand_dims" => {
            let a = args.get(0).and_then(Value::as_int).unwrap_or(0);
            let a = if a < 0 {
                (a + t.ndim() as i64 + 1).max(0)
            } else {
                a
            } as usize;
            Value::Tensor(t.unsqueeze(a))
        }
        "squeeze" => {
            let dims: Vec<i64> = t
                .shape()
                .iter()
                .filter(|&&d| d != 1)
                .map(|&d| d as i64)
                .collect();
            wrap(interp, t.reshape(&dims))?
        }
        "sum" => wrap(interp, t.sum(axis_arg(args, 0)))?,
        "mean" | "average" => wrap(interp, t.mean(axis_arg(args, 0)))?,
        "max" | "amax" => {
            if let Some(o) = args.get(0).filter(|v| matches!(v, Value::Tensor(_))) {
                let o = interp.to_tensor(&o.clone(), span)?;
                wrap(interp, t.binary_map(&o, f64::max))?
            } else {
                wrap(interp, t.max(axis_arg(args, 0)))?
            }
        }
        "min" | "amin" => {
            let neg = t.neg();
            let m = neg.max(axis_arg(args, 0)).map(|m| m.neg());
            wrap(interp, m)?
        }
        "argmax" => wrap(interp, t.argmax(axis_arg(args, 0)))?,
        "argmin" => wrap(interp, t.neg().argmax(axis_arg(args, 0)))?,
        "softmax" => wrap(interp, t.softmax(axis_arg(args, 0)))?,
        "log_softmax" => wrap(interp, t.log_softmax(axis_arg(args, 0)))?,
        "exp" => Value::Tensor(t.unary(Unary::Exp)),
        "log" => Value::Tensor(t.unary(Unary::Log)),
        "sqrt" => Value::Tensor(t.unary(Unary::Sqrt)),
        "abs" => Value::Tensor(t.unary(Unary::Abs)),
        "relu" => Value::Tensor(t.unary(Unary::Relu)),
        "sigmoid" => Value::Tensor(t.unary(Unary::Sigmoid)),
        "tanh" => Value::Tensor(t.unary(Unary::Tanh)),
        "sin" => Value::Tensor(t.unary(Unary::Sin)),
        "cos" => Value::Tensor(t.unary(Unary::Cos)),
        "neg" => Value::Tensor(t.neg()),
        "pow" => {
            let o = other(interp)?;
            wrap(interp, t.pow(&o))?
        }
        "dot" | "matmul" | "mm" => {
            let o = other(interp)?;
            wrap(interp, t.matmul(&o))?
        }
        "add" => {
            let o = other(interp)?;
            wrap(interp, t.add(&o))?
        }
        "sub" => {
            let o = other(interp)?;
            wrap(interp, t.sub(&o))?
        }
        "mul" => {
            let o = other(interp)?;
            wrap(interp, t.mul(&o))?
        }
        "div" => {
            let o = other(interp)?;
            wrap(interp, t.div(&o))?
        }
        "round" => {
            let p = 10f64.powi(args.get(0).and_then(Value::as_int).unwrap_or(0) as i32);
            Value::Tensor(t.map(|v| (v * p).round() / p))
        }
        "clip" | "clamp" => {
            let lo = args
                .get(0)
                .and_then(Value::as_f64)
                .unwrap_or(f64::NEG_INFINITY);
            let hi = args.get(1).and_then(Value::as_f64).unwrap_or(f64::INFINITY);
            Value::Tensor(t.map(|v| v.clamp(lo, hi)))
        }
        "std" | "var" => {
            let d = t.data();
            let n = d.len().max(1) as f64;
            let mean = d.iter().sum::<f64>() / n;
            let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            Value::Float(if name == "std" { var.sqrt() } else { var })
        }
        "norm" => Value::Float(t.with_data(|d| d.iter().map(|x| x * x).sum::<f64>().sqrt())),
        "all" => Value::Bool(t.with_data(|d| d.iter().all(|&x| x != 0.0))),
        "any" => Value::Bool(t.with_data(|d| d.iter().any(|&x| x != 0.0))),
        "astype" | "to" | "float" | "double" => Value::Tensor(t.clone()),
        "fill_" | "fill" => {
            let v = args.get(0).and_then(Value::as_f64).unwrap_or(0.0);
            t.update(|d| d.fill(v));
            Value::Tensor(t.clone())
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Installs tensor constructors, math, nn and autodiff built-ins.
pub fn install(interp: &mut Interp) {
    def(
        interp,
        &[
            "tensor",
            "array",
            "Tensor",
            "matrix",
            "vector",
            "asarray",
            "from_numpy",
        ],
        make_tensor,
    );
    def(
        interp,
        &["param", "parameter", "Parameter", "variable", "Variable"],
        |interp, args| {
            let v = arg(interp, &args, 0, "param")?;
            let t = interp.to_tensor(&v, args.span)?.detach();
            t.set_requires_grad(true);
            Ok(Value::Tensor(t))
        },
    );
    def(interp, &["zeros", "zeros_like_shape"], |interp, args| {
        Ok(Value::Tensor(Tensor::zeros(&shape_args(
            interp, &args, "zeros",
        )?)))
    });
    def(interp, &["ones"], |interp, args| {
        Ok(Value::Tensor(Tensor::ones(&shape_args(
            interp, &args, "ones",
        )?)))
    });
    def(interp, &["full"], |interp, args| {
        let shape_v = arg(interp, &args, 0, "full")?;
        let v = args.get(1).and_then(Value::as_f64).unwrap_or(0.0);
        let shape = shape_args(interp, &Args::new(vec![shape_v], args.span), "full")?;
        Ok(Value::Tensor(Tensor::full(&shape, v)))
    });
    def(interp, &["eye", "identity_matrix"], |interp, args| {
        let n = int_arg(interp, &args, 0, "eye")?.clamp(0, 5000) as usize;
        Ok(Value::Tensor(Tensor::eye(n)))
    });
    def(
        interp,
        &["randn", "random_normal_tensor"],
        |interp, args| {
            let shape = shape_args(interp, &args, "randn")?;
            let n: usize = shape.iter().product();
            let data = (0..n).map(|_| interp.rng.normal(0.0, 1.0)).collect();
            Tensor::new(data, shape)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span))
        },
    );
    def(
        interp,
        &["rand_tensor", "uniform_tensor"],
        |interp, args| {
            let shape = shape_args(interp, &args, "rand")?;
            let n: usize = shape.iter().product();
            let data = (0..n).map(|_| interp.rng.next_f64()).collect();
            Tensor::new(data, shape)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span))
        },
    );
    def(interp, &["arange"], |interp, args| {
        let nums: Vec<f64> = args.pos.iter().filter_map(Value::as_f64).collect();
        let (start, stop, step) = match nums.as_slice() {
            [n] => (0.0, *n, 1.0),
            [a, b] => (*a, *b, 1.0),
            [a, b, s, ..] => (*a, *b, *s),
            [] => return interp.type_err("arange needs a stop value", args.span),
        };
        if step == 0.0 {
            return interp.value_err("arange step cannot be zero", args.span);
        }
        let n = ((stop - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f64> = (0..n.min(50_000_000))
            .map(|i| start + step * i as f64)
            .collect();
        let len = data.len();
        Ok(Value::Tensor(
            Tensor::new(data, vec![len]).unwrap_or_else(|_| Tensor::zeros(&[0])),
        ))
    });
    def(interp, &["linspace"], |interp, args| {
        let a = crate::builtins::float_arg(interp, &args, 0, "linspace")?;
        let b = crate::builtins::float_arg(interp, &args, 1, "linspace")?;
        let n = args
            .get(2)
            .and_then(Value::as_int)
            .unwrap_or(50)
            .clamp(0, 50_000_000) as usize;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                if n == 1 {
                    a
                } else {
                    a + (b - a) * i as f64 / (n - 1) as f64
                }
            })
            .collect();
        Ok(Value::Tensor(
            Tensor::new(data, vec![n]).unwrap_or_else(|_| Tensor::zeros(&[0])),
        ))
    });
    def(interp, &["stack", "vstack"], |interp, args| {
        let xs = arg(interp, &args, 0, "stack")?;
        let mut parts = Vec::new();
        for x in interp.collect(&xs, args.span)? {
            parts.push(interp.to_tensor(&x, args.span)?);
        }
        Tensor::stack(&parts)
            .map(Value::Tensor)
            .map_err(|e| interp.error("ValueError", e, args.span))
    });
    def(
        interp,
        &["cat", "concatenate", "hstack_rows"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "cat")?;
            let mut parts = Vec::new();
            for x in interp.collect(&xs, args.span)? {
                parts.push(interp.to_tensor(&x, args.span)?);
            }
            Tensor::concat(&parts)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span))
        },
    );
    def(interp, &["dot", "matmul", "mm", "inner"], |interp, args| {
        let a = arg(interp, &args, 0, "dot")?;
        let b = arg(interp, &args, 1, "dot")?;
        let (x, y) = (
            interp.to_tensor(&a, args.span)?,
            interp.to_tensor(&b, args.span)?,
        );
        let r = x
            .matmul(&y)
            .map_err(|e| interp.error("ValueError", e, args.span))?;
        Ok(if r.ndim() == 0 && !matches!(a, Value::Tensor(_)) {
            Value::Float(r.item())
        } else {
            Value::Tensor(r)
        })
    });
    def(interp, &["softmax"], |interp, args| {
        let v = arg(interp, &args, 0, "softmax")?;
        let t = interp.to_tensor(&v, args.span)?;
        t.softmax(axis_arg(&args, 1))
            .map(Value::Tensor)
            .map_err(|e| interp.error("ValueError", e, args.span))
    });
    def(interp, &["argmax"], |interp, args| {
        let v = arg(interp, &args, 0, "argmax")?;
        if let Value::List(_) = &v {
            let items = interp.collect(&v, args.span)?;
            let mut best = 0;
            for i in 1..items.len() {
                if interp
                    .compare_values(&items[i], &items[best], args.span)?
                    .is_gt()
                {
                    best = i;
                }
            }
            return Ok(Value::Int(best as i64));
        }
        let t = interp.to_tensor(&v, args.span)?;
        let r = t
            .argmax(axis_arg(&args, 1))
            .map_err(|e| interp.error("ValueError", e, args.span))?;
        Ok(if r.ndim() == 0 {
            Value::Int(r.item() as i64)
        } else {
            Value::Tensor(r)
        })
    });
    def(interp, &["no_grad_eval"], |interp, args| {
        let f = arg(interp, &args, 0, "no_grad")?;
        tensor::no_grad(|| crate::builtins::call(interp, &f, Vec::new(), args.span))
    });
    def(
        interp,
        &["derivative", "diff", "deriv", "D"],
        |interp, args| {
            let f = arg(interp, &args, 0, "derivative")?;
            let order = args
                .arg(2, "order")
                .and_then(|o| o.as_int())
                .unwrap_or(1)
                .clamp(1, 8) as u32;
            let d = autodiff::derivative_fn(f, order);
            match args.get(1) {
                Some(x) => crate::builtins::call(interp, &d, vec![x.clone()], args.span),
                None => Ok(d),
            }
        },
    );
    def(
        interp,
        &["grad", "gradient", "nabla", "∇"],
        |interp, args| {
            let f = arg(interp, &args, 0, "grad")?;
            let g = autodiff::gradient_fn(f);
            if args.len() > 1 {
                return crate::builtins::call(interp, &g, args.pos[1..].to_vec(), args.span);
            }
            Ok(g)
        },
    );
    constant(
        interp,
        &["float32", "float64", "int64", "int32"],
        Value::str("float64"),
    );
    nn::install(interp);
}

/// `import numpy as np`, `import torch`, `from torch import nn`, …
pub fn module(interp: &mut Interp, name: &str) -> Option<Value> {
    let b = |n: &str| builtin(interp, n);
    Some(match name {
        "numpy" | "np" | "nd" => {
            let random = make_module(
                "random",
                vec![
                    ("rand", b("rand_tensor")),
                    ("randn", b("randn")),
                    ("seed", b("seed")),
                    ("random", b("rand_tensor")),
                    ("randint", b("randint")),
                    ("choice", b("choice")),
                    ("shuffle", b("shuffle")),
                    ("normal", b("randn")),
                    ("uniform", b("uniform")),
                ],
            );
            let linalg = make_module(
                "linalg",
                vec![(
                    "norm",
                    Value::native("norm", |interp, args| {
                        let v = arg(interp, &args, 0, "norm")?;
                        let t = interp.to_tensor(&v, args.span)?;
                        Ok(Value::Float(t.with_data(|d| {
                            d.iter().map(|x| x * x).sum::<f64>().sqrt()
                        })))
                    }),
                )],
            );
            let mut members = vec![
                ("array", b("tensor")),
                ("asarray", b("tensor")),
                ("zeros", b("zeros")),
                ("ones", b("ones")),
                ("full", b("full")),
                ("eye", b("eye")),
                ("identity", b("eye")),
                ("arange", b("arange")),
                ("linspace", b("linspace")),
                ("dot", b("dot")),
                ("matmul", b("dot")),
                ("stack", b("stack")),
                ("vstack", b("stack")),
                ("concatenate", b("cat")),
                ("argmax", b("argmax")),
                ("exp", b("exp")),
                ("log", b("log")),
                ("sqrt", b("sqrt")),
                ("abs", b("abs")),
                ("tanh", b("tanh")),
                ("sin", b("sin")),
                ("cos", b("cos")),
                ("round", b("round")),
                ("pi", b("pi")),
                ("e", b("e")),
                ("inf", b("inf")),
                ("nan", b("nan")),
                ("float64", Value::str("float64")),
                ("float32", Value::str("float32")),
                ("random", random),
                ("linalg", linalg),
            ];
            for (n, op) in [
                ("sum", "sum"),
                ("mean", "mean"),
                ("max", "max"),
                ("min", "min"),
                ("std", "std"),
                ("var", "var"),
                ("transpose", "transpose"),
                ("reshape", "reshape"),
                ("clip", "clip"),
                ("argmin", "argmin"),
            ] {
                members.push((
                    n,
                    Value::native(n, move |interp, mut args| {
                        let v = arg(interp, &args, 0, op)?;
                        let t = interp.to_tensor(&v, args.span)?;
                        args.pos.remove(0);
                        let span = args.span;
                        match tensor_method(interp, &t, op, &args)? {
                            Some(r) => Ok(r),
                            None => interp.type_err(format!("np.{op} is not supported"), span),
                        }
                    }),
                ));
            }
            make_module("numpy", members)
        }
        "torch"
        | "torch.nn"
        | "nn"
        | "keras"
        | "tensorflow"
        | "tf"
        | "tensorflow.keras"
        | "keras.layers"
        | "keras.models"
        | "sklearn"
        | "sklearn.linear_model"
        | "sklearn.cluster"
        | "sklearn.model_selection"
        | "sklearn.metrics" => nn::framework_module(interp, name),
        _ => return None,
    })
}

#[allow(dead_code)]
fn unused(_: Rc<()>) {
    let _ = sym("_");
}
