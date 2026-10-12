//! Neural networks: layers, models, optimizers, losses, training, the
//! `model`/`train` statements, PyTorch/Keras-style constructors and a few
//! scikit-learn style estimators — all on top of [`Tensor`] autograd.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use chimera_core::Span;

use super::tensor::{Tensor, Unary, no_grad};
use crate::ast::{Arg, ModelDef, TrainSpec};
use crate::builtins::objects::module as make_module;
use crate::builtins::{arg, call, def};
use crate::interp::{Args, Env, Interp, sym};
use crate::symbol::Sym;
use crate::value::*;

// ----- activations ---------------------------------------------------------

/// Activation functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// max(0, x)
    Relu,
    /// tanh
    Tanh,
    /// logistic sigmoid
    Sigmoid,
    /// softmax over the last axis
    Softmax,
    /// GELU
    Gelu,
    /// SiLU / swish
    Silu,
    /// softplus
    Softplus,
    /// leaky ReLU
    LeakyRelu(f64),
    /// no-op
    Identity,
}

impl Act {
    fn parse(name: &str) -> Option<Act> {
        Some(match name.to_ascii_lowercase().as_str() {
            "relu" => Act::Relu,
            "tanh" => Act::Tanh,
            "sigmoid" | "logistic" => Act::Sigmoid,
            "softmax" => Act::Softmax,
            "gelu" => Act::Gelu,
            "silu" | "swish" => Act::Silu,
            "softplus" => Act::Softplus,
            "leaky_relu" | "leakyrelu" => Act::LeakyRelu(0.01),
            "linear" | "identity" | "none" => Act::Identity,
            _ => return None,
        })
    }

    fn apply(self, x: &Tensor) -> Result<Tensor, String> {
        Ok(match self {
            Act::Relu => x.unary(Unary::Relu),
            Act::Tanh => x.unary(Unary::Tanh),
            Act::Sigmoid => x.unary(Unary::Sigmoid),
            Act::Softmax => x.softmax(None)?,
            Act::Gelu => x.unary(Unary::Gelu),
            Act::Silu => x.unary(Unary::Silu),
            Act::Softplus => x.unary(Unary::Softplus),
            Act::LeakyRelu(a) => x.unary(Unary::LeakyRelu(a)),
            Act::Identity => x.clone(),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Act::Relu => "ReLU",
            Act::Tanh => "Tanh",
            Act::Sigmoid => "Sigmoid",
            Act::Softmax => "Softmax",
            Act::Gelu => "GELU",
            Act::Silu => "SiLU",
            Act::Softplus => "Softplus",
            Act::LeakyRelu(_) => "LeakyReLU",
            Act::Identity => "Identity",
        }
    }
}

// ----- layers -------------------------------------------------------------------

#[derive(Debug)]
enum LayerKind {
    Linear {
        w: RefCell<Option<Tensor>>,
        b: RefCell<Option<Tensor>>,
        out_f: usize,
        act: Act,
    },
    Activation(Act),
    Dropout(f64),
    Flatten,
}

/// A network layer.
#[derive(Debug)]
pub struct Layer {
    kind: LayerKind,
    training: Cell<bool>,
}

fn xavier(interp: &mut Interp, fan_in: usize, fan_out: usize) -> Tensor {
    let limit = (6.0 / (fan_in + fan_out).max(1) as f64).sqrt();
    let data = (0..fan_in * fan_out)
        .map(|_| interp.rng.range_f64(-limit, limit))
        .collect();
    let w = Tensor::new(data, vec![fan_in, fan_out])
        .unwrap_or_else(|_| Tensor::zeros(&[fan_in, fan_out]));
    w.set_requires_grad(true);
    w
}

impl Layer {
    fn linear(interp: &mut Interp, in_f: Option<usize>, out_f: usize, act: Act) -> Layer {
        let (w, b) = match in_f {
            Some(i) => {
                let w = xavier(interp, i, out_f);
                let b = Tensor::zeros(&[out_f]);
                b.set_requires_grad(true);
                (Some(w), Some(b))
            }
            None => (None, None),
        };
        Layer {
            kind: LayerKind::Linear {
                w: RefCell::new(w),
                b: RefCell::new(b),
                out_f,
                act,
            },
            training: Cell::new(false),
        }
    }

    fn simple(kind: LayerKind) -> Layer {
        Layer {
            kind,
            training: Cell::new(false),
        }
    }

    fn forward(&self, interp: &mut Interp, x: &Tensor) -> Result<Tensor, String> {
        match &self.kind {
            LayerKind::Linear { w, b, out_f, act } => {
                if w.borrow().is_none() {
                    // Keras-style lazy shape inference on the first call.
                    let in_f = *x
                        .shape()
                        .last()
                        .ok_or("Dense layer needs at least 1-D input")?;
                    *w.borrow_mut() = Some(xavier(interp, in_f, *out_f));
                    let bias = Tensor::zeros(&[*out_f]);
                    bias.set_requires_grad(true);
                    *b.borrow_mut() = Some(bias);
                }
                let wt = w.borrow().clone().ok_or("uninitialised layer")?;
                let bt = b.borrow().clone().ok_or("uninitialised layer")?;
                let expected = wt.shape()[0];
                if x.shape().last() != Some(&expected) {
                    return Err(format!(
                        "this layer expects {expected} input features but got shape {:?}",
                        x.shape()
                    ));
                }
                act.apply(&x.matmul(&wt)?.add(&bt)?)
            }
            LayerKind::Activation(a) => a.apply(x),
            LayerKind::Dropout(p) => {
                if !self.training.get() || *p <= 0.0 {
                    return Ok(x.clone());
                }
                let keep = 1.0 - p;
                let mask: Vec<f64> = (0..x.numel())
                    .map(|_| {
                        if interp.rng.chance(keep) {
                            1.0 / keep
                        } else {
                            0.0
                        }
                    })
                    .collect();
                x.mul(&Tensor::new(mask, x.shape().to_vec())?)
            }
            LayerKind::Flatten => {
                if x.ndim() <= 1 {
                    Ok(x.clone())
                } else {
                    x.reshape(&[x.shape()[0] as i64, -1])
                }
            }
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match &self.kind {
            LayerKind::Linear { w, b, .. } => w
                .borrow()
                .iter()
                .chain(b.borrow().iter())
                .cloned()
                .collect(),
            _ => Vec::new(),
        }
    }

    fn describe(&self) -> (String, String) {
        match &self.kind {
            LayerKind::Linear { w, out_f, act, .. } => {
                let in_f = w
                    .borrow()
                    .as_ref()
                    .map_or("?".to_string(), |w| w.shape()[0].to_string());
                let act = if *act == Act::Identity {
                    String::new()
                } else {
                    format!(", {}", act.name())
                };
                (
                    format!("Dense({in_f} → {out_f}{act})"),
                    format!("(*, {out_f})"),
                )
            }
            LayerKind::Activation(a) => (a.name().to_string(), "same".into()),
            LayerKind::Dropout(p) => (format!("Dropout({p})"), "same".into()),
            LayerKind::Flatten => ("Flatten".into(), "(*, -1)".into()),
        }
    }
}

impl Object for Layer {
    fn type_name(&self) -> &str {
        "Layer"
    }

    fn call(&self, interp: &mut Interp, args: Args, _this: &Value) -> Option<R> {
        Some((|| {
            let x = arg(interp, &args, 0, "layer")?;
            let t = interp.to_tensor(&x, args.span)?;
            self.forward(interp, &t)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span))
        })())
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        match name {
            "forward" | "__call__" => self.call(interp, args, this),
            "parameters" => Some(Ok(Value::list(
                self.parameters().into_iter().map(Value::Tensor).collect(),
            ))),
            _ => None,
        }
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match (&self.kind, name) {
            (LayerKind::Linear { w, .. }, "weight" | "W" | "kernel") => {
                Some(Ok(w.borrow().clone().map_or(Value::Nil, Value::Tensor)))
            }
            (LayerKind::Linear { b, .. }, "bias" | "b") => {
                Some(Ok(b.borrow().clone().map_or(Value::Nil, Value::Tensor)))
            }
            _ => None,
        }
    }

    fn display(&self) -> String {
        self.describe().0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ----- models ---------------------------------------------------------------------

/// A sequential model (also what `model Name:` declarations build).
#[derive(Debug)]
pub struct Sequential {
    name: String,
    layers: RefCell<Vec<Rc<Layer>>>,
    config: RefCell<Vec<(Sym, Value)>>,
}

impl Sequential {
    fn forward(&self, interp: &mut Interp, x: &Tensor) -> Result<Tensor, String> {
        let layers = self.layers.borrow().clone();
        let mut h = x.clone();
        for l in &layers {
            h = l.forward(interp, &h)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .borrow()
            .iter()
            .flat_map(|l| l.parameters())
            .collect()
    }

    fn set_training(&self, on: bool) {
        for l in self.layers.borrow().iter() {
            l.training.set(on);
        }
    }

    fn last_activation(&self) -> Option<Act> {
        self.layers
            .borrow()
            .iter()
            .rev()
            .find_map(|l| match &l.kind {
                LayerKind::Linear { act, .. } if *act != Act::Identity => Some(*act),
                LayerKind::Activation(a) => Some(*a),
                LayerKind::Linear { .. } => Some(Act::Identity),
                _ => None,
            })
    }
}

impl Object for Sequential {
    fn type_name(&self) -> &str {
        "Model"
    }

    fn call(&self, interp: &mut Interp, args: Args, _this: &Value) -> Option<R> {
        Some((|| {
            let x = arg(interp, &args, 0, "model")?;
            let t = interp.to_tensor(&x, args.span)?;
            self.forward(interp, &t)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span))
        })())
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let span = args.span;
        Some((|| -> R {
            match name {
                "forward" | "__call__" => self.call(interp, args, this).unwrap_or(Ok(Value::Nil)),
                "predict" | "infer" | "predict_proba" => {
                    let x = arg(interp, &args, 0, "predict")?;
                    let t = interp.to_tensor(&x, span)?;
                    self.set_training(false);
                    let r = no_grad(|| self.forward(interp, &t));
                    r.map(Value::Tensor)
                        .map_err(|e| interp.error("ValueError", e, span))
                }
                "parameters" | "params" | "trainable_variables" | "weights" => Ok(Value::list(
                    self.parameters().into_iter().map(Value::Tensor).collect(),
                )),
                "zero_grad" => {
                    for p in self.parameters() {
                        p.zero_grad();
                    }
                    Ok(Value::Nil)
                }
                "train" => {
                    self.set_training(args.get(0).is_none_or(Value::truthy));
                    Ok(this.clone())
                }
                "eval" => {
                    self.set_training(false);
                    Ok(this.clone())
                }
                "add" | "append" => {
                    let l = arg(interp, &args, 0, "add")?;
                    let layer = as_layer(interp, &l, span)?;
                    self.layers.borrow_mut().push(layer);
                    Ok(Value::Nil)
                }
                "compile" => {
                    let mut cfg = self.config.borrow_mut();
                    for (k, v) in &args.named {
                        cfg.push((*k, v.clone()));
                    }
                    if let Some(o) = args.get(0) {
                        cfg.push((sym("optimizer"), o.clone()));
                    }
                    if let Some(l) = args.get(1) {
                        cfg.push((sym("loss"), l.clone()));
                    }
                    Ok(Value::Nil)
                }
                "fit" | "train_on" | "learn" => {
                    let x = arg(interp, &args, 0, "fit")?;
                    let y = arg(interp, &args, 1, "fit")?;
                    let mut options: Vec<(Sym, Value)> = self.config.borrow().clone();
                    for (k, v) in &args.named {
                        let key = match &*k.as_str() {
                            "batch_size" => sym("batch"),
                            "learning_rate" => sym("lr"),
                            "verbose" => sym("report"),
                            _ => *k,
                        };
                        let v = if key.is("report") && v.as_int() == Some(1) {
                            Value::Int(0)
                        } else if key.is("report") && v.as_int() == Some(0) {
                            Value::Int(-1)
                        } else {
                            v.clone()
                        };
                        options.push((key, v));
                    }
                    let losses = train(interp, this, self, &x, &y, &options, span)?;
                    Ok(Value::map_from(vec![(
                        "loss",
                        Value::list(losses.into_iter().map(Value::Float).collect()),
                    )]))
                }
                "evaluate" | "score_loss" => {
                    let x = arg(interp, &args, 0, "evaluate")?;
                    let y = arg(interp, &args, 1, "evaluate")?;
                    let xt = interp.to_tensor(&x, span)?;
                    let yt = interp.to_tensor(&y, span)?;
                    let loss_name = self
                        .config
                        .borrow()
                        .iter()
                        .find(|(k, _)| k.is("loss"))
                        .map(|(_, v)| v.clone());
                    let loss =
                        resolve_loss(interp, loss_name.as_ref(), self.last_activation(), span)?;
                    self.set_training(false);
                    let pred = no_grad(|| self.forward(interp, &xt))
                        .map_err(|e| interp.error("ValueError", e, span))?;
                    let l = no_grad(|| loss.apply(interp, &pred, &yt, span))?;
                    Ok(Value::Float(l.item()))
                }
                "summary" => {
                    let mut t = chimera_core::Table::new(["#", "layer", "output", "params"]);
                    let mut total = 0;
                    for (i, l) in self.layers.borrow().iter().enumerate() {
                        let (desc, shape) = l.describe();
                        let n: usize = l.parameters().iter().map(Tensor::numel).sum();
                        total += n;
                        t.add_row([i.to_string(), desc, shape, n.to_string()]);
                    }
                    let text = format!(
                        "Model \"{}\"\n{}total parameters: {total}\n",
                        self.name,
                        t.render()
                    );
                    interp.write_out(&text);
                    Ok(Value::Nil)
                }
                "save_weights" | "state_dict" => {
                    let ps: Vec<Value> = self
                        .parameters()
                        .iter()
                        .map(super::tensor_to_list)
                        .collect();
                    Ok(Value::list(ps))
                }
                "__iter__" => Ok(Value::list(
                    self.layers
                        .borrow()
                        .iter()
                        .map(|l| Value::Object(l.clone() as Rc<dyn Object>))
                        .collect(),
                )),
                _ => interp.type_err(format!("models have no method `{name}`"), span),
            }
        })())
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match name {
            "layers" => Some(Ok(Value::list(
                self.layers
                    .borrow()
                    .iter()
                    .map(|l| Value::Object(l.clone() as Rc<dyn Object>))
                    .collect(),
            ))),
            "name" => Some(Ok(Value::str(&self.name))),
            _ => None,
        }
    }

    fn display(&self) -> String {
        let layers: Vec<String> = self
            .layers
            .borrow()
            .iter()
            .map(|l| l.describe().0)
            .collect();
        format!("{}({})", self.name, layers.join(" → "))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn as_layer(interp: &mut Interp, v: &Value, span: Span) -> R<Rc<Layer>> {
    // Layers are shared by reference so their weights stay in sync.
    if let Value::Object(o) = v
        && o.as_any().is::<Layer>()
        && let Ok(layer) = rc_to_any(o.clone()).downcast::<Layer>()
    {
        return Ok(layer);
    }
    if let Value::Str(s) = v
        && let Some(a) = Act::parse(s)
    {
        return Ok(Rc::new(Layer::simple(LayerKind::Activation(a))));
    }
    let t = v.type_name();
    interp.type_err(format!("expected a layer, got {t}"), span)
}

/// Upcasts an `Rc<dyn Object>` to `Rc<dyn Any>` for downcasting.
fn rc_to_any(rc: Rc<dyn Object>) -> Rc<dyn Any> {
    rc
}

fn new_model(name: &str, layers: Vec<Rc<Layer>>) -> Value {
    Value::Object(Rc::new(Sequential {
        name: name.to_owned(),
        layers: RefCell::new(layers),
        config: RefCell::new(Vec::new()),
    }))
}

fn model_ref(v: &Value) -> Option<Rc<Sequential>> {
    let Value::Object(o) = v else { return None };
    let any: Rc<dyn Any> = rc_to_any(o.clone());
    any.downcast::<Sequential>().ok()
}

// ----- losses ---------------------------------------------------------------------

/// A loss function.
#[derive(Clone, Debug)]
pub enum Loss {
    /// Mean squared error.
    Mse,
    /// Mean absolute error.
    Mae,
    /// Binary cross-entropy on probabilities.
    Bce,
    /// Binary cross-entropy on logits.
    BceLogits,
    /// Categorical cross-entropy on logits (targets: indices or one-hot).
    CrossEntropy,
    /// A user-supplied function `(pred, target) -> loss`.
    Custom(Value),
}

impl Loss {
    fn parse(name: &str) -> Option<Loss> {
        Some(
            match name
                .to_ascii_lowercase()
                .replace(['_', '-', ' '], "")
                .as_str()
            {
                "mse" | "meansquarederror" | "mseloss" | "l2" | "squarederror" => Loss::Mse,
                "mae" | "meanabsoluteerror" | "l1" | "l1loss" => Loss::Mae,
                "bce" | "binarycrossentropy" | "bceloss" | "logloss" => Loss::Bce,
                "bcewithlogits" | "bcewithlogitsloss" => Loss::BceLogits,
                "ce"
                | "crossentropy"
                | "categoricalcrossentropy"
                | "sparsecategoricalcrossentropy"
                | "crossentropyloss"
                | "nll" => Loss::CrossEntropy,
                _ => return None,
            },
        )
    }

    /// Computes the loss.
    pub fn apply(
        &self,
        interp: &mut Interp,
        pred: &Tensor,
        target: &Tensor,
        span: Span,
    ) -> R<Tensor> {
        let err = |interp: &mut Interp, e: String| interp.error("ValueError", e, span);
        let target = if target.shape() != pred.shape()
            && target.numel() == pred.numel()
            && !matches!(self, Loss::CrossEntropy)
        {
            target
                .reshape(&pred.shape().iter().map(|&d| d as i64).collect::<Vec<_>>())
                .map_err(|e| err(interp, e))?
        } else {
            target.clone()
        };
        let r = match self {
            Loss::Mse => pred
                .sub(&target)
                .and_then(|d| d.mul(&d))
                .and_then(|s| s.mean(None)),
            Loss::Mae => pred
                .sub(&target)
                .map(|d| d.unary(Unary::Abs))
                .and_then(|a| a.mean(None)),
            Loss::Bce => {
                let eps = Tensor::scalar(1e-12);
                let one = Tensor::scalar(1.0);
                (|| {
                    let a = target.mul(&pred.add(&eps)?.unary(Unary::Log))?;
                    let b = one
                        .sub(&target)?
                        .mul(&one.sub(pred)?.add(&eps)?.unary(Unary::Log))?;
                    a.add(&b)?.mean(None).map(|m| m.neg())
                })()
            }
            Loss::BceLogits => (|| {
                pred.unary(Unary::Softplus)
                    .sub(&target.mul(pred)?)?
                    .mean(None)
            })(),
            Loss::CrossEntropy => (|| {
                let logp = pred.log_softmax(None)?;
                let onehot = if target.shape() == pred.shape() {
                    target.clone()
                } else {
                    let classes = *pred
                        .shape()
                        .last()
                        .ok_or("cross entropy needs class scores")?;
                    let rows = target.numel();
                    let mut d = vec![0.0; rows * classes];
                    for (i, c) in target.data().iter().enumerate() {
                        let c = *c as usize;
                        if c >= classes {
                            return Err(format!(
                                "class index {c} is out of range for {classes} classes"
                            ));
                        }
                        d[i * classes + c] = 1.0;
                    }
                    Tensor::new(d, logp.shape().to_vec())?
                };
                let rows = if logp.ndim() >= 2 {
                    logp.shape()[0] as f64
                } else {
                    1.0
                };
                onehot
                    .mul(&logp)?
                    .sum(None)?
                    .neg()
                    .div(&Tensor::scalar(rows))
            })(),
            Loss::Custom(f) => {
                let v = call(
                    interp,
                    f,
                    vec![Value::Tensor(pred.clone()), Value::Tensor(target.clone())],
                    span,
                )?;
                return interp.to_tensor(&v, span);
            }
        };
        r.map_err(|e| err(interp, e))
    }
}

fn resolve_loss(
    interp: &mut Interp,
    spec: Option<&Value>,
    last: Option<Act>,
    span: Span,
) -> R<Loss> {
    match spec {
        Some(Value::Str(s)) => {
            Loss::parse(s).map_or_else(|| interp.value_err(format!("unknown loss `{s}`"), span), Ok)
        }
        Some(Value::Object(o)) if o.type_name() == "Loss" => Ok(o
            .as_any()
            .downcast_ref::<LossObj>()
            .map(|l| l.0.clone())
            .unwrap_or(Loss::Mse)),
        Some(f) if f.is_callable() => Ok(Loss::Custom(f.clone())),
        _ => Ok(match last {
            Some(Act::Sigmoid) => Loss::Bce,
            Some(Act::Softmax) => Loss::Mse,
            _ => Loss::Mse,
        }),
    }
}

/// A loss as a callable object (`nn.MSELoss()`).
#[derive(Debug)]
pub struct LossObj(Loss);

impl Object for LossObj {
    fn type_name(&self) -> &str {
        "Loss"
    }

    fn call(&self, interp: &mut Interp, args: Args, _this: &Value) -> Option<R> {
        Some((|| {
            let p = arg(interp, &args, 0, "loss")?;
            let y = arg(interp, &args, 1, "loss")?;
            let (pt, yt) = (
                interp.to_tensor(&p, args.span)?,
                interp.to_tensor(&y, args.span)?,
            );
            self.0.apply(interp, &pt, &yt, args.span).map(Value::Tensor)
        })())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ----- optimizers -------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum OptKind {
    Sgd { momentum: f64 },
    Adam { b1: f64, b2: f64, eps: f64 },
    RmsProp { decay: f64, eps: f64 },
}

/// A gradient-descent optimizer.
#[derive(Debug)]
pub struct Optimizer {
    kind: OptKind,
    lr: Cell<f64>,
    params: RefCell<Vec<Tensor>>,
    state1: RefCell<Vec<Vec<f64>>>,
    state2: RefCell<Vec<Vec<f64>>>,
    t: Cell<u64>,
}

impl Optimizer {
    fn new(kind: OptKind, lr: f64, params: Vec<Tensor>) -> Optimizer {
        Optimizer {
            kind,
            lr: Cell::new(lr),
            params: RefCell::new(params),
            state1: RefCell::new(Vec::new()),
            state2: RefCell::new(Vec::new()),
            t: Cell::new(0),
        }
    }

    fn parse(name: &str) -> Option<OptKind> {
        Some(match name.to_ascii_lowercase().as_str() {
            "sgd" | "gd" | "gradient_descent" => OptKind::Sgd { momentum: 0.0 },
            "momentum" => OptKind::Sgd { momentum: 0.9 },
            "adam" | "adamw" => OptKind::Adam {
                b1: 0.9,
                b2: 0.999,
                eps: 1e-8,
            },
            "rmsprop" => OptKind::RmsProp {
                decay: 0.9,
                eps: 1e-8,
            },
            _ => return None,
        })
    }

    fn default_lr(kind: OptKind) -> f64 {
        match kind {
            OptKind::Sgd { .. } => 0.1,
            OptKind::Adam { .. } => 0.01,
            OptKind::RmsProp { .. } => 0.01,
        }
    }

    /// Applies one update using the current gradients.
    pub fn step(&self) {
        let params = self.params.borrow();
        let mut s1 = self.state1.borrow_mut();
        let mut s2 = self.state2.borrow_mut();
        if s1.len() != params.len() {
            *s1 = params.iter().map(|p| vec![0.0; p.numel()]).collect();
            *s2 = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        }
        self.t.set(self.t.get() + 1);
        let t = self.t.get() as f64;
        let lr = self.lr.get();
        for (i, p) in params.iter().enumerate() {
            let Some(g) = p.grad_data() else { continue };
            let (m, v) = (&mut s1[i], &mut s2[i]);
            p.update(|d| match self.kind {
                OptKind::Sgd { momentum } => {
                    for j in 0..d.len() {
                        m[j] = momentum * m[j] + g[j];
                        d[j] -= lr * m[j];
                    }
                }
                OptKind::Adam { b1, b2, eps } => {
                    for j in 0..d.len() {
                        m[j] = b1 * m[j] + (1.0 - b1) * g[j];
                        v[j] = b2 * v[j] + (1.0 - b2) * g[j] * g[j];
                        let mh = m[j] / (1.0 - b1.powf(t));
                        let vh = v[j] / (1.0 - b2.powf(t));
                        d[j] -= lr * mh / (vh.sqrt() + eps);
                    }
                }
                OptKind::RmsProp { decay, eps } => {
                    for j in 0..d.len() {
                        v[j] = decay * v[j] + (1.0 - decay) * g[j] * g[j];
                        d[j] -= lr * g[j] / (v[j].sqrt() + eps);
                    }
                }
            });
        }
    }

    fn zero_grad(&self) {
        for p in self.params.borrow().iter() {
            p.zero_grad();
        }
    }
}

impl Object for Optimizer {
    fn type_name(&self) -> &str {
        "Optimizer"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, _this: &Value) -> Option<R> {
        let _ = (interp, &args);
        match name {
            "step" | "update" | "apply_gradients" => {
                self.step();
                Some(Ok(Value::Nil))
            }
            "zero_grad" | "clear_grad" => {
                self.zero_grad();
                Some(Ok(Value::Nil))
            }
            _ => None,
        }
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match name {
            "lr" | "learning_rate" => Some(Ok(Value::Float(self.lr.get()))),
            _ => None,
        }
    }

    fn set_attr(&self, _interp: &mut Interp, name: &str, value: Value) -> bool {
        match (name, value.as_f64()) {
            ("lr" | "learning_rate", Some(v)) => {
                self.lr.set(v);
                true
            }
            _ => false,
        }
    }

    fn display(&self) -> String {
        format!("{:?}(lr={})", self.kind, self.lr.get())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn optimizer_ref(v: &Value) -> Option<Rc<Optimizer>> {
    let Value::Object(o) = v else { return None };
    rc_to_any(o.clone()).downcast::<Optimizer>().ok()
}

fn params_from(interp: &mut Interp, v: &Value, span: Span) -> R<Vec<Tensor>> {
    if let Some(m) = model_ref(v) {
        return Ok(m.parameters());
    }
    let mut out = Vec::new();
    for p in interp.collect(v, span)? {
        if let Value::Tensor(t) = p {
            out.push(t);
        }
    }
    Ok(out)
}

fn make_optimizer(interp: &mut Interp, kind: OptKind, args: &Args) -> R {
    let params = match args.get(0) {
        Some(v) if !v.is_number() => params_from(interp, v, args.span)?,
        _ => Vec::new(),
    };
    let lr = args
        .named("lr")
        .or_else(|| args.named("learning_rate"))
        .or_else(|| args.pos.iter().find(|v| v.is_number()).cloned())
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| Optimizer::default_lr(kind));
    let kind = match (kind, args.named("momentum").and_then(|m| m.as_f64())) {
        (OptKind::Sgd { .. }, Some(m)) => OptKind::Sgd { momentum: m },
        (k, _) => k,
    };
    Ok(Value::Object(Rc::new(Optimizer::new(kind, lr, params))))
}

// ----- training -----------------------------------------------------------------------

fn option<'a>(opts: &'a [(Sym, Value)], name: &str) -> Option<&'a Value> {
    opts.iter().rev().find(|(k, _)| k.is(name)).map(|(_, v)| v)
}

/// Trains `model` on `(x, y)`; returns the loss per epoch.
pub fn train(
    interp: &mut Interp,
    model_value: &Value,
    model: &Sequential,
    x: &Value,
    y: &Value,
    opts: &[(Sym, Value)],
    span: Span,
) -> R<Vec<f64>> {
    let xt = interp.to_tensor(x, span)?;
    let yt = interp.to_tensor(y, span)?;
    let epochs = option(opts, "epochs")
        .and_then(Value::as_int)
        .unwrap_or(100)
        .clamp(0, 10_000_000) as usize;
    let loss = resolve_loss(interp, option(opts, "loss"), model.last_activation(), span)?;
    // Initialise lazy layers before collecting parameters.
    model.set_training(false);
    no_grad(|| model.forward(interp, &xt)).map_err(|e| interp.error("ValueError", e, span))?;
    let params = model.parameters();
    let opt = match option(opts, "optimizer") {
        Some(v) if optimizer_ref(v).is_some() => {
            let o =
                optimizer_ref(v).ok_or_else(|| interp.error("TypeError", "bad optimizer", span))?;
            if o.params.borrow().is_empty() {
                *o.params.borrow_mut() = params.clone();
            }
            if let Some(lr) = option(opts, "lr").and_then(Value::as_f64) {
                o.lr.set(lr);
            }
            o
        }
        other => {
            let kind = match other {
                Some(Value::Str(s)) => match Optimizer::parse(s) {
                    Some(k) => k,
                    None => return interp.value_err(format!("unknown optimizer `{s}`"), span),
                },
                _ => OptKind::Adam {
                    b1: 0.9,
                    b2: 0.999,
                    eps: 1e-8,
                },
            };
            let lr = option(opts, "lr")
                .and_then(Value::as_f64)
                .unwrap_or_else(|| Optimizer::default_lr(kind));
            Rc::new(Optimizer::new(kind, lr, params.clone()))
        }
    };
    let n = xt.shape().first().copied().unwrap_or(1);
    let batch = option(opts, "batch")
        .and_then(Value::as_int)
        .map_or(n, |b| (b.max(1) as usize).min(n.max(1)));
    let report = option(opts, "report").and_then(Value::as_int);
    let every = match report {
        Some(0) => (epochs / 10).max(1),
        Some(k) if k > 0 => k as usize,
        _ => usize::MAX,
    };
    let _ = model_value;
    let mut history = Vec::with_capacity(epochs);
    let mut order: Vec<usize> = (0..n).collect();
    model.set_training(true);
    for epoch in 1..=epochs {
        if batch < n {
            interp.rng.shuffle(&mut order);
        }
        let mut total = 0.0;
        let mut batches = 0;
        for chunk in order.chunks(batch.max(1)) {
            let (xb, yb) = if batch >= n {
                (xt.clone(), yt.clone())
            } else {
                let idx: Vec<i64> = chunk.iter().map(|&i| i as i64).collect();
                (gather_rows(&xt, &idx), gather_rows(&yt, &idx))
            };
            opt.zero_grad();
            let pred = model
                .forward(interp, &xb)
                .map_err(|e| interp.error("ValueError", e, span))?;
            let l = loss.apply(interp, &pred, &yb, span)?;
            l.backward();
            opt.step();
            total += l.item();
            batches += 1;
        }
        let epoch_loss = total / batches.max(1) as f64;
        history.push(epoch_loss);
        if epoch % every == 0
            || (report.is_some_and(|r| r >= 0) && epoch == epochs && every != usize::MAX)
        {
            let line = format!("epoch {epoch:>5}  loss {epoch_loss:.6}\n");
            interp.write_out(&line);
        }
    }
    model.set_training(false);
    opt.zero_grad();
    Ok(history)
}

fn gather_rows(t: &Tensor, rows: &[i64]) -> Tensor {
    let parts: Vec<Tensor> = rows
        .iter()
        .filter_map(|&r| t.slice(&[super::tensor::SliceSpec::Index(r)]).ok())
        .collect();
    no_grad(|| Tensor::stack(&parts)).unwrap_or_else(|_| t.clone())
}

/// `train net on X, y for N epochs …`.
pub fn train_statement(interp: &mut Interp, spec: &TrainSpec, env: &Rc<Env>) -> R {
    let model_value = interp.eval(&spec.model, env)?;
    let x = interp.eval(&spec.inputs, env)?;
    let y = interp.eval(&spec.targets, env)?;
    let mut opts = Vec::with_capacity(spec.options.len());
    for (k, e) in &spec.options {
        opts.push((*k, interp.eval(e, env)?));
    }
    let Some(model) = model_ref(&model_value) else {
        // Any object with a `fit` method trains too.
        let span = spec.span;
        let named: Vec<(Sym, Value)> = opts;
        return interp.call_method(
            &model_value,
            sym("fit"),
            Args {
                pos: vec![x, y],
                named,
                span,
            },
            span,
        );
    };
    // The value is the per-epoch loss history.
    let history = train(interp, &model_value, &model, &x, &y, &opts, spec.span)?;
    Ok(Value::list(history.into_iter().map(Value::Float).collect()))
}

/// `model Name: dense 2 -> 8, relu …` defines a constructor.
pub fn define_model(interp: &mut Interp, def: &Rc<ModelDef>, env: &Rc<Env>) -> R {
    let def = def.clone();
    let env = env.clone();
    let name = def.name.as_str().to_string();
    let _ = interp;
    Ok(Value::native(&name.clone(), move |interp, _args| {
        let mut layers = Vec::with_capacity(def.layers.len());
        for spec in &def.layers {
            let mut a = Args {
                pos: Vec::new(),
                named: Vec::new(),
                span: spec.span,
            };
            for arg in &spec.args {
                match arg {
                    Arg::Pos(e) => a.pos.push(interp.eval(e, &env)?),
                    Arg::Named(n, e) => {
                        let v = interp.eval(e, &env)?;
                        a.named.push((*n, v));
                    }
                    _ => {}
                }
            }
            layers.push(build_layer(interp, &spec.kind.as_str(), &a)?);
        }
        Ok(new_model(&name, layers))
    }))
}

fn build_layer(interp: &mut Interp, kind: &str, a: &Args) -> R<Rc<Layer>> {
    let act = |interp: &mut Interp| -> R<Act> {
        match a.named("activation") {
            Some(Value::Str(s)) => Act::parse(&s).map_or_else(
                || interp.value_err(format!("unknown activation `{s}`"), a.span),
                Ok,
            ),
            Some(Value::Nil) | None => Ok(Act::Identity),
            Some(other) => {
                let t = other.type_name();
                interp.type_err(format!("activation must be a name, got {t}"), a.span)
            }
        }
    };
    let lower = kind.to_ascii_lowercase();
    Ok(Rc::new(match lower.as_str() {
        "dense" | "linear" | "fc" | "fullyconnected" => {
            let nums: Vec<usize> = a
                .pos
                .iter()
                .filter_map(Value::as_int)
                .map(|n| n.max(1) as usize)
                .collect();
            let input = a
                .named("input_dim")
                .or_else(|| a.named("in_features"))
                .or_else(|| {
                    a.named("input_shape").and_then(|s| match s {
                        Value::Tuple(t) => t.first().cloned(),
                        Value::List(l) => l.borrow().first().cloned(),
                        other => Some(other),
                    })
                })
                .and_then(|v| v.as_int())
                .map(|n| n.max(1) as usize);
            let (in_f, out_f) = match nums.as_slice() {
                [o] => (input, *o),
                [i, o, ..] => (Some(*i), *o),
                [] => match a
                    .named("units")
                    .or_else(|| a.named("out_features"))
                    .and_then(|v| v.as_int())
                {
                    Some(o) => (input, o.max(1) as usize),
                    None => {
                        return interp
                            .value_err("a dense layer needs an output size", a.span)
                            .map(|_: Value| unreachable!());
                    }
                },
            };
            let act = act(interp)?;
            Layer::linear(interp, in_f, out_f, act)
        }
        "dropout" => Layer::simple(LayerKind::Dropout(
            a.get(0)
                .and_then(Value::as_f64)
                .unwrap_or(0.5)
                .clamp(0.0, 0.99),
        )),
        "flatten" => Layer::simple(LayerKind::Flatten),
        "activation" => {
            let name = a
                .get(0)
                .map(|v| crate::interp::format::display_plain(v, false))
                .unwrap_or_default();
            match Act::parse(&name) {
                Some(x) => Layer::simple(LayerKind::Activation(x)),
                None => {
                    return interp
                        .value_err(format!("unknown activation `{name}`"), a.span)
                        .map(|_: Value| unreachable!());
                }
            }
        }
        "leakyrelu" | "leaky_relu" => Layer::simple(LayerKind::Activation(Act::LeakyRelu(
            a.get(0).and_then(Value::as_f64).unwrap_or(0.01),
        ))),
        other => match Act::parse(other) {
            Some(x) => Layer::simple(LayerKind::Activation(x)),
            None => {
                return interp
                    .value_err(format!("unknown layer type `{kind}`"), a.span)
                    .map(|_: Value| unreachable!());
            }
        },
    }))
}

fn layer_value(l: Layer) -> Value {
    Value::Object(Rc::new(l))
}

// ----- scikit-learn style estimators ---------------------------------------------------

#[derive(Debug)]
enum EstimatorKind {
    LinearRegression,
    LogisticRegression { lr: f64, epochs: usize },
    KMeans { k: usize, iters: usize },
}

/// A classical ML model with `fit`/`predict`.
#[derive(Debug)]
pub struct Estimator {
    kind: EstimatorKind,
    weights: RefCell<Option<Tensor>>,
    bias: Cell<f64>,
    centroids: RefCell<Option<Tensor>>,
    labels: RefCell<Vec<i64>>,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let (top, bottom) = a.split_at_mut(row);
            for (x, p) in bottom[0][col..n].iter_mut().zip(&top[col][col..n]) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

fn as_matrix(t: &Tensor) -> (usize, usize, Vec<f64>) {
    match t.ndim() {
        0 => (1, 1, t.data()),
        1 => (t.shape()[0], 1, t.data()),
        _ => (t.shape()[0], t.numel() / t.shape()[0], t.data()),
    }
}

impl Estimator {
    fn new(kind: EstimatorKind) -> Estimator {
        Estimator {
            kind,
            weights: RefCell::new(None),
            bias: Cell::new(0.0),
            centroids: RefCell::new(None),
            labels: RefCell::new(Vec::new()),
        }
    }

    fn fit(&self, interp: &mut Interp, x: &Tensor, y: Option<&Tensor>, span: Span) -> R<()> {
        let (n, d, xs) = as_matrix(x);
        match &self.kind {
            EstimatorKind::LinearRegression => {
                let Some(y) = y else {
                    return interp.type_err("fit(X, y) needs targets", span);
                };
                let ys = y.data();
                // Normal equations with a bias column: (AᵀA) w = Aᵀy.
                let m = d + 1;
                let mut ata = vec![vec![0.0; m]; m];
                let mut aty = vec![0.0; m];
                for i in 0..n {
                    let row: Vec<f64> = (0..d)
                        .map(|j| xs[i * d + j])
                        .chain(std::iter::once(1.0))
                        .collect();
                    for p in 0..m {
                        aty[p] += row[p] * ys[i];
                        for q in 0..m {
                            ata[p][q] += row[p] * row[q];
                        }
                    }
                }
                for (p, row) in ata.iter_mut().enumerate() {
                    row[p] += 1e-10;
                }
                let Some(w) = solve(ata, aty) else {
                    return interp
                        .value_err("the training data is degenerate (singular matrix)", span);
                };
                self.bias.set(w[d]);
                *self.weights.borrow_mut() = Tensor::new(w[..d].to_vec(), vec![d]).ok();
            }
            EstimatorKind::LogisticRegression { lr, epochs } => {
                let Some(y) = y else {
                    return interp.type_err("fit(X, y) needs targets", span);
                };
                let w = Tensor::zeros(&[d]);
                w.set_requires_grad(true);
                let b = Tensor::zeros(&[1]);
                b.set_requires_grad(true);
                let xm =
                    Tensor::new(xs, vec![n, d]).map_err(|e| interp.error("ValueError", e, span))?;
                let opt = Optimizer::new(
                    OptKind::Adam {
                        b1: 0.9,
                        b2: 0.999,
                        eps: 1e-8,
                    },
                    *lr,
                    vec![w.clone(), b.clone()],
                );
                let yv = y
                    .reshape(&[n as i64])
                    .map_err(|e| interp.error("ValueError", e, span))?;
                for _ in 0..*epochs {
                    opt.zero_grad();
                    let z = xm
                        .matmul(&w)
                        .and_then(|z| z.add(&b))
                        .map_err(|e| interp.error("ValueError", e, span))?;
                    let l = Loss::BceLogits.apply(interp, &z, &yv, span)?;
                    l.backward();
                    opt.step();
                }
                self.bias.set(b.item());
                *self.weights.borrow_mut() = Some(w.detach());
            }
            EstimatorKind::KMeans { k, iters } => {
                let k = (*k).min(n).max(1);
                // k-means++ style seeding with the interpreter's RNG.
                let mut centers: Vec<Vec<f64>> = vec![xs[interp.rng.index(n) * d..][..d].to_vec()];
                while centers.len() < k {
                    let dists: Vec<f64> = (0..n)
                        .map(|i| {
                            centers
                                .iter()
                                .map(|c| {
                                    (0..d).map(|j| (xs[i * d + j] - c[j]).powi(2)).sum::<f64>()
                                })
                                .fold(f64::INFINITY, f64::min)
                        })
                        .collect();
                    let total: f64 = dists.iter().sum();
                    let mut r = interp.rng.next_f64() * total;
                    let mut pick = n - 1;
                    for (i, dd) in dists.iter().enumerate() {
                        r -= dd;
                        if r <= 0.0 {
                            pick = i;
                            break;
                        }
                    }
                    centers.push(xs[pick * d..][..d].to_vec());
                }
                let mut labels = vec![0i64; n];
                for _ in 0..*iters {
                    let mut changed = false;
                    for i in 0..n {
                        let best = (0..k)
                            .min_by(|&a, &b| {
                                let da: f64 = (0..d)
                                    .map(|j| (xs[i * d + j] - centers[a][j]).powi(2))
                                    .sum();
                                let db: f64 = (0..d)
                                    .map(|j| (xs[i * d + j] - centers[b][j]).powi(2))
                                    .sum();
                                da.total_cmp(&db)
                            })
                            .unwrap_or(0) as i64;
                        if labels[i] != best {
                            labels[i] = best;
                            changed = true;
                        }
                    }
                    for (c, center) in centers.iter_mut().enumerate() {
                        let members: Vec<usize> =
                            (0..n).filter(|&i| labels[i] == c as i64).collect();
                        if members.is_empty() {
                            continue;
                        }
                        for j in 0..d {
                            center[j] = members.iter().map(|&i| xs[i * d + j]).sum::<f64>()
                                / members.len() as f64;
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                *self.labels.borrow_mut() = labels;
                *self.centroids.borrow_mut() = Tensor::new(centers.concat(), vec![k, d]).ok();
            }
        }
        Ok(())
    }

    fn predict(&self, interp: &mut Interp, x: &Tensor, proba: bool, span: Span) -> R {
        let (n, d, xs) = as_matrix(x);
        match &self.kind {
            EstimatorKind::LinearRegression | EstimatorKind::LogisticRegression { .. } => {
                let Some(w) = self.weights.borrow().clone() else {
                    return interp.value_err("call fit() before predict()", span);
                };
                let wd = w.data();
                if wd.len() != d {
                    return interp
                        .value_err(format!("expected {} features, got {d}", wd.len()), span);
                }
                let out: Vec<f64> = (0..n)
                    .map(|i| {
                        let z: f64 =
                            (0..d).map(|j| xs[i * d + j] * wd[j]).sum::<f64>() + self.bias.get();
                        match (&self.kind, proba) {
                            (EstimatorKind::LogisticRegression { .. }, true) => {
                                1.0 / (1.0 + (-z).exp())
                            }
                            (EstimatorKind::LogisticRegression { .. }, false) => {
                                f64::from(u8::from(z > 0.0))
                            }
                            _ => z,
                        }
                    })
                    .collect();
                Ok(Value::Tensor(
                    Tensor::new(out, vec![n]).map_err(|e| interp.error("ValueError", e, span))?,
                ))
            }
            EstimatorKind::KMeans { .. } => {
                let Some(c) = self.centroids.borrow().clone() else {
                    return interp.value_err("call fit() before predict()", span);
                };
                let k = c.shape()[0];
                let cd = c.data();
                let labels: Vec<Value> = (0..n)
                    .map(|i| {
                        let best = (0..k)
                            .min_by(|&a, &b| {
                                let da: f64 = (0..d)
                                    .map(|j| (xs[i * d + j] - cd[a * d + j]).powi(2))
                                    .sum();
                                let db: f64 = (0..d)
                                    .map(|j| (xs[i * d + j] - cd[b * d + j]).powi(2))
                                    .sum();
                                da.total_cmp(&db)
                            })
                            .unwrap_or(0);
                        Value::Int(best as i64)
                    })
                    .collect();
                Ok(Value::list(labels))
            }
        }
    }
}

impl Object for Estimator {
    fn type_name(&self) -> &str {
        match self.kind {
            EstimatorKind::LinearRegression => "LinearRegression",
            EstimatorKind::LogisticRegression { .. } => "LogisticRegression",
            EstimatorKind::KMeans { .. } => "KMeans",
        }
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let span = args.span;
        Some((|| -> R {
            match name {
                "fit" => {
                    let x = arg(interp, &args, 0, "fit")?;
                    let xt = interp.to_tensor(&x, span)?;
                    let yt = match args.get(1) {
                        Some(y) => Some(interp.to_tensor(y, span)?),
                        None => None,
                    };
                    self.fit(interp, &xt, yt.as_ref(), span)?;
                    Ok(this.clone())
                }
                "predict" | "predict_proba" | "transform" => {
                    let x = arg(interp, &args, 0, "predict")?;
                    let xt = interp.to_tensor(&x, span)?;
                    self.predict(interp, &xt, name == "predict_proba", span)
                }
                "fit_predict" => {
                    let x = arg(interp, &args, 0, "fit_predict")?;
                    let xt = interp.to_tensor(&x, span)?;
                    self.fit(interp, &xt, None, span)?;
                    self.predict(interp, &xt, false, span)
                }
                "score" => {
                    let x = arg(interp, &args, 0, "score")?;
                    let y = arg(interp, &args, 1, "score")?;
                    let xt = interp.to_tensor(&x, span)?;
                    let pred = self.predict(interp, &xt, false, span)?;
                    let p = interp.to_tensor(&pred, span)?.data();
                    let t = interp.to_tensor(&y, span)?.data();
                    Ok(Value::Float(match self.kind {
                        EstimatorKind::LinearRegression => {
                            let mean = t.iter().sum::<f64>() / t.len().max(1) as f64;
                            let ss_res: f64 = p.iter().zip(&t).map(|(a, b)| (a - b).powi(2)).sum();
                            let ss_tot: f64 = t.iter().map(|b| (b - mean).powi(2)).sum();
                            1.0 - ss_res / ss_tot.max(1e-300)
                        }
                        _ => {
                            p.iter()
                                .zip(&t)
                                .filter(|(a, b)| (*a - *b).abs() < 1e-9)
                                .count() as f64
                                / t.len().max(1) as f64
                        }
                    }))
                }
                _ => interp.type_err(
                    format!("`{}` has no method `{name}`", self.type_name()),
                    span,
                ),
            }
        })())
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        Some(Ok(match name {
            "coef_" | "coef" | "weights" => self
                .weights
                .borrow()
                .clone()
                .map_or(Value::Nil, Value::Tensor),
            "intercept_" | "intercept" | "bias" => Value::Float(self.bias.get()),
            "cluster_centers_" | "centroids" => self
                .centroids
                .borrow()
                .clone()
                .map_or(Value::Nil, Value::Tensor),
            "labels_" | "labels" => Value::list(
                self.labels
                    .borrow()
                    .iter()
                    .map(|&l| Value::Int(l))
                    .collect(),
            ),
            _ => return None,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ----- registration ----------------------------------------------------------------------

/// Installs layer/model/optimizer/loss constructors.
pub fn install(interp: &mut Interp) {
    def(
        interp,
        &["Dense", "Linear", "dense", "linear", "fc"],
        |interp, args| {
            let l = build_layer(interp, "dense", &args)?;
            Ok(Value::Object(l))
        },
    );
    for (names, act) in [
        (&["ReLU", "Relu"][..], Act::Relu),
        (&["Tanh"], Act::Tanh),
        (&["Sigmoid"], Act::Sigmoid),
        (&["Softmax"], Act::Softmax),
        (&["GELU", "Gelu"], Act::Gelu),
        (&["SiLU", "Swish"], Act::Silu),
        (&["Softplus"], Act::Softplus),
        (&["LeakyReLU"], Act::LeakyRelu(0.01)),
    ] {
        def(interp, names, move |_interp, _args| {
            Ok(layer_value(Layer::simple(LayerKind::Activation(act))))
        });
    }
    def(interp, &["Dropout"], |_interp, args| {
        let p = args
            .get(0)
            .or_else(|| args.named.first().map(|(_, v)| v))
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        Ok(layer_value(Layer::simple(LayerKind::Dropout(
            p.clamp(0.0, 0.99),
        ))))
    });
    def(interp, &["Flatten"], |_interp, _args| {
        Ok(layer_value(Layer::simple(LayerKind::Flatten)))
    });
    def(interp, &["Activation"], |interp, args| {
        let l = build_layer(interp, "activation", &args)?;
        Ok(Value::Object(l))
    });
    def(
        interp,
        &["Sequential", "sequential", "Model", "MLP"],
        |interp, args| {
            let items = match args.pos.as_slice() {
                [single @ Value::List(_)] => interp.collect(single, args.span)?,
                other => other.to_vec(),
            };
            let mut layers = Vec::with_capacity(items.len());
            for item in &items {
                layers.push(as_layer(interp, item, args.span)?);
            }
            let name = args
                .named("name")
                .map(|n| crate::interp::format::display_plain(&n, false))
                .unwrap_or_else(|| "Sequential".into());
            Ok(new_model(&name, layers))
        },
    );
    for (names, kind) in [
        (&["sgd", "SGD"][..], OptKind::Sgd { momentum: 0.0 }),
        (
            &["adam", "Adam", "AdamW", "adamw"],
            OptKind::Adam {
                b1: 0.9,
                b2: 0.999,
                eps: 1e-8,
            },
        ),
        (
            &["rmsprop", "RMSprop"],
            OptKind::RmsProp {
                decay: 0.9,
                eps: 1e-8,
            },
        ),
        (&["momentum"], OptKind::Sgd { momentum: 0.9 }),
    ] {
        def(interp, names, move |interp, args| {
            make_optimizer(interp, kind, &args)
        });
    }
    for (names, loss) in [
        (
            &["mse", "mse_loss", "MSELoss", "mean_squared_error"][..],
            Loss::Mse,
        ),
        (
            &["mae", "l1_loss", "L1Loss", "mean_absolute_error"],
            Loss::Mae,
        ),
        (
            &[
                "bce",
                "bce_loss",
                "BCELoss",
                "binary_crossentropy",
                "binary_cross_entropy",
            ],
            Loss::Bce,
        ),
        (&["BCEWithLogitsLoss", "bce_with_logits"], Loss::BceLogits),
        (
            &[
                "cross_entropy",
                "CrossEntropyLoss",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
            ],
            Loss::CrossEntropy,
        ),
    ] {
        let l = loss.clone();
        def(interp, names, move |interp, args| {
            if args.len() >= 2 {
                let p = interp.to_tensor(&args.pos[0].clone(), args.span)?;
                let y = interp.to_tensor(&args.pos[1].clone(), args.span)?;
                return l.apply(interp, &p, &y, args.span).map(Value::Tensor);
            }
            Ok(Value::Object(Rc::new(LossObj(l.clone()))))
        });
    }
    def(interp, &["train", "fit_model"], |interp, args| {
        let m = arg(interp, &args, 0, "train")?;
        let x = arg(interp, &args, 1, "train")?;
        let y = arg(interp, &args, 2, "train")?;
        let Some(model) = model_ref(&m) else {
            let span = args.span;
            return interp.call_method(
                &m,
                sym("fit"),
                Args {
                    pos: vec![x, y],
                    named: args.named.clone(),
                    span,
                },
                span,
            );
        };
        let h = train(interp, &m, &model, &x, &y, &args.named, args.span)?;
        Ok(h.last().map_or(Value::Nil, |&l| Value::Float(l)))
    });
    def(interp, &["LinearRegression"], |_interp, _args| {
        Ok(Value::Object(Rc::new(Estimator::new(
            EstimatorKind::LinearRegression,
        ))))
    });
    def(interp, &["LogisticRegression"], |_interp, args| {
        let lr = args.named("lr").and_then(|v| v.as_f64()).unwrap_or(0.1);
        let epochs = args
            .named("max_iter")
            .or_else(|| args.named("epochs"))
            .and_then(|v| v.as_int())
            .unwrap_or(500)
            .max(1) as usize;
        Ok(Value::Object(Rc::new(Estimator::new(
            EstimatorKind::LogisticRegression { lr, epochs },
        ))))
    });
    def(interp, &["KMeans", "kmeans"], |_interp, args| {
        let k = args
            .arg(0, "n_clusters")
            .or_else(|| args.named("k"))
            .and_then(|v| v.as_int())
            .unwrap_or(8)
            .max(1) as usize;
        let iters = args
            .named("max_iter")
            .and_then(|v| v.as_int())
            .unwrap_or(100)
            .max(1) as usize;
        Ok(Value::Object(Rc::new(Estimator::new(
            EstimatorKind::KMeans { k, iters },
        ))))
    });
    def(interp, &["train_test_split"], |interp, args| {
        let x = arg(interp, &args, 0, "train_test_split")?;
        let y = args.get(1).cloned();
        let test = args
            .named("test_size")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let xs = interp.collect(&x, args.span)?;
        let ys = match &y {
            Some(y) => interp.collect(y, args.span)?,
            None => Vec::new(),
        };
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        if args.named("shuffle").is_none_or(|s| s.truthy()) {
            interp.rng.shuffle(&mut idx);
        }
        let n_test = ((xs.len() as f64) * test).round() as usize;
        let (test_idx, train_idx) = idx.split_at(n_test.min(idx.len()));
        let pick =
            |src: &[Value], ix: &[usize]| Value::list(ix.iter().map(|&i| src[i].clone()).collect());
        let mut out = vec![pick(&xs, train_idx), pick(&xs, test_idx)];
        if y.is_some() {
            out.push(pick(&ys, train_idx));
            out.push(pick(&ys, test_idx));
        }
        Ok(Value::tuple(out))
    });
    def(interp, &["accuracy", "accuracy_score"], |interp, args| {
        let a = arg(interp, &args, 0, "accuracy")?;
        let b = arg(interp, &args, 1, "accuracy")?;
        let (ta, tb) = (
            interp.to_tensor(&a, args.span)?.data(),
            interp.to_tensor(&b, args.span)?.data(),
        );
        if ta.len() != tb.len() || ta.is_empty() {
            return interp.value_err(
                "accuracy needs two equally sized, non-empty sequences",
                args.span,
            );
        }
        let hits = ta
            .iter()
            .zip(&tb)
            .filter(|(x, y)| (**x - **y).abs() < 1e-9)
            .count();
        Ok(Value::Float(hits as f64 / ta.len() as f64))
    });
    def(interp, &["one_hot", "to_categorical"], |interp, args| {
        let v = arg(interp, &args, 0, "one_hot")?;
        let t = interp.to_tensor(&v, args.span)?.data();
        let classes = args.get(1).and_then(Value::as_int).map_or_else(
            || t.iter().fold(0.0f64, |a, &b| a.max(b)) as usize + 1,
            |c| c.max(1) as usize,
        );
        let mut d = vec![0.0; t.len() * classes];
        for (i, &c) in t.iter().enumerate() {
            let c = c as usize;
            if c < classes {
                d[i * classes + c] = 1.0;
            }
        }
        Tensor::new(d, vec![t.len(), classes])
            .map(Value::Tensor)
            .map_err(|e| interp.error("ValueError", e, args.span))
    });
}

/// `torch`, `torch.nn`, `keras`, `sklearn` namespaces.
pub fn framework_module(interp: &mut Interp, name: &str) -> Value {
    let b = |n: &str| crate::builtins::objects::builtin(interp, n);
    let nn_members = vec![
        ("Linear", b("Linear")),
        ("Sequential", b("Sequential")),
        ("ReLU", b("ReLU")),
        ("Tanh", b("Tanh")),
        ("Sigmoid", b("Sigmoid")),
        ("Softmax", b("Softmax")),
        ("GELU", b("GELU")),
        ("LeakyReLU", b("LeakyReLU")),
        ("Dropout", b("Dropout")),
        ("Flatten", b("Flatten")),
        ("MSELoss", b("MSELoss")),
        ("L1Loss", b("L1Loss")),
        ("BCELoss", b("BCELoss")),
        ("BCEWithLogitsLoss", b("BCEWithLogitsLoss")),
        ("CrossEntropyLoss", b("CrossEntropyLoss")),
        ("Parameter", b("param")),
        (
            "functional",
            make_module(
                "functional",
                vec![
                    ("relu", b("relu")),
                    ("sigmoid", b("sigmoid")),
                    ("softmax", b("softmax")),
                    ("mse_loss", b("mse_loss")),
                    ("cross_entropy", b("cross_entropy")),
                ],
            ),
        ),
    ];
    let optim = make_module(
        "optim",
        vec![
            ("SGD", b("SGD")),
            ("Adam", b("Adam")),
            ("AdamW", b("Adam")),
            ("RMSprop", b("RMSprop")),
        ],
    );
    match name {
        "torch.nn" | "nn" => make_module("nn", nn_members),
        "torch" => make_module(
            "torch",
            vec![
                ("tensor", b("tensor")),
                ("zeros", b("zeros")),
                ("ones", b("ones")),
                ("randn", b("randn")),
                ("rand", b("rand_tensor")),
                ("eye", b("eye")),
                ("arange", b("arange")),
                ("linspace", b("linspace")),
                ("stack", b("stack")),
                ("cat", b("cat")),
                ("matmul", b("dot")),
                ("relu", b("relu")),
                ("sigmoid", b("sigmoid")),
                ("tanh", b("tanh")),
                ("softmax", b("softmax")),
                ("exp", b("exp")),
                ("log", b("log")),
                ("manual_seed", b("seed")),
                ("no_grad", b("no_grad_eval")),
                ("nn", make_module("nn", nn_members)),
                ("optim", optim),
                ("float32", Value::str("float32")),
            ],
        ),
        "keras" | "tensorflow" | "tf" | "tensorflow.keras" | "keras.layers" | "keras.models" => {
            let layers = make_module(
                "layers",
                vec![
                    ("Dense", b("Dense")),
                    ("Dropout", b("Dropout")),
                    ("Flatten", b("Flatten")),
                    ("Activation", b("Activation")),
                ],
            );
            let models = make_module("models", vec![("Sequential", b("Sequential"))]);
            let optimizers = make_module(
                "optimizers",
                vec![
                    ("SGD", b("SGD")),
                    ("Adam", b("Adam")),
                    ("RMSprop", b("RMSprop")),
                ],
            );
            let keras = make_module(
                "keras",
                vec![
                    ("layers", layers.clone()),
                    ("models", models.clone()),
                    ("optimizers", optimizers),
                    ("Sequential", b("Sequential")),
                ],
            );
            match name {
                "keras.layers" => layers,
                "keras.models" => models,
                "tensorflow" | "tf" => make_module(
                    "tensorflow",
                    vec![
                        ("keras", keras),
                        ("constant", b("tensor")),
                        ("Variable", b("param")),
                    ],
                ),
                _ => keras,
            }
        }
        _ => {
            let sk = vec![
                ("LinearRegression", b("LinearRegression")),
                ("LogisticRegression", b("LogisticRegression")),
                ("KMeans", b("KMeans")),
                ("train_test_split", b("train_test_split")),
                ("accuracy_score", b("accuracy_score")),
                ("mean_squared_error", b("mean_squared_error")),
            ];
            make_module(name, sk)
        }
    }
}

#[allow(dead_code)]
fn unused(_: Rc<()>) {}
