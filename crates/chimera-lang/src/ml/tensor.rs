//! Tensors with reverse-mode automatic differentiation.
//!
//! A [`Tensor`] is a reference-counted n-dimensional array of `f64` plus an
//! optional *autograd* record: the operation that produced it and its inputs.
//! Calling [`Tensor::backward`] walks that graph in reverse topological order
//! (iteratively — no recursion, so deep graphs are fine) and accumulates
//! gradients into every input that `requires_grad`.
//!
//! Elementwise operations broadcast with NumPy's rules; their backward pass
//! sums gradients over broadcast dimensions. Indexing and slicing are all
//! expressed as a *gather* of flat indices, whose backward is a scatter-add,
//! so every way of slicing is differentiable for free.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

thread_local! {
    static NO_GRAD: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` without recording autograd history (inference, optimizer steps).
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    NO_GRAD.with(|c| c.set(c.get() + 1));
    let r = f();
    NO_GRAD.with(|c| c.set(c.get() - 1));
    r
}

fn grad_enabled() -> bool {
    NO_GRAD.with(|c| c.get() == 0)
}

/// How a tensor was computed (for the backward pass).
#[derive(Clone)]
enum Op {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    Pow(Tensor, Tensor),
    Neg(Tensor),
    MatMul(Tensor, Tensor),
    Unary(Tensor, Unary),
    Sum(Tensor, Option<usize>),
    Mean(Tensor, Option<usize>),
    Max(Tensor, Rc<Vec<usize>>),
    Transpose(Tensor),
    Reshape(Tensor),
    Gather(Tensor, Rc<Vec<usize>>),
    Concat(Vec<Tensor>),
}

/// Elementwise unary functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unary {
    /// e^x
    Exp,
    /// ln x
    Log,
    /// tanh x
    Tanh,
    /// 1 / (1 + e^-x)
    Sigmoid,
    /// max(0, x)
    Relu,
    /// x if x > 0 else αx
    LeakyRelu(f64),
    /// √x
    Sqrt,
    /// |x|
    Abs,
    /// sin x
    Sin,
    /// cos x
    Cos,
    /// x·σ(x)
    Silu,
    /// ln(1 + e^x)
    Softplus,
    /// GELU (tanh approximation)
    Gelu,
}

impl Unary {
    fn apply(self, x: f64) -> f64 {
        match self {
            Unary::Exp => x.exp(),
            Unary::Log => x.ln(),
            Unary::Tanh => x.tanh(),
            Unary::Sigmoid => sigmoid(x),
            Unary::Relu => x.max(0.0),
            Unary::LeakyRelu(a) => {
                if x > 0.0 {
                    x
                } else {
                    a * x
                }
            }
            Unary::Sqrt => x.sqrt(),
            Unary::Abs => x.abs(),
            Unary::Sin => x.sin(),
            Unary::Cos => x.cos(),
            Unary::Silu => x * sigmoid(x),
            Unary::Softplus => softplus(x),
            Unary::Gelu => gelu(x),
        }
    }

    /// d(f(x))/dx given x and y = f(x).
    fn derivative(self, x: f64, y: f64) -> f64 {
        match self {
            Unary::Exp => y,
            Unary::Log => 1.0 / x,
            Unary::Tanh => 1.0 - y * y,
            Unary::Sigmoid => y * (1.0 - y),
            Unary::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Unary::LeakyRelu(a) => {
                if x > 0.0 {
                    1.0
                } else {
                    a
                }
            }
            Unary::Sqrt => 0.5 / y,
            Unary::Abs => x.signum(),
            Unary::Sin => x.cos(),
            Unary::Cos => -x.sin(),
            Unary::Silu => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            Unary::Softplus => sigmoid(x),
            Unary::Gelu => {
                let c = (2.0 / std::f64::consts::PI).sqrt();
                let inner = c * (x + 0.044_715 * x.powi(3));
                let t = inner.tanh();
                0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044_715 * x * x)
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}

fn gelu(x: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x.powi(3))).tanh())
}

struct Node {
    data: RefCell<Vec<f64>>,
    shape: Vec<usize>,
    grad: RefCell<Option<Vec<f64>>>,
    requires_grad: Cell<bool>,
    op: Option<Op>,
}

/// An n-dimensional array of `f64` with autograd.
#[derive(Clone)]
pub struct Tensor(Rc<Node>);

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// A slicing instruction for one axis.
#[derive(Clone, Copy, Debug)]
pub enum SliceSpec {
    /// A single index (removes the axis).
    Index(i64),
    /// `start:stop:step` (keeps the axis); `None` bounds mean the ends.
    Range(Option<i64>, Option<i64>, i64),
}

/// Errors from tensor operations (shape mismatches and the like).
pub type TResult<T> = Result<T, String>;

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn shape_str(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// NumPy broadcasting of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> TResult<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let db = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return Err(format!(
                "shapes {} and {} cannot be broadcast together",
                shape_str(a),
                shape_str(b)
            ));
        };
    }
    Ok(out)
}

/// For each flat index in `out_shape`, the flat index in a broadcast input.
fn broadcast_index_map(input: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let n = out_shape.len();
    let in_strides = strides(input);
    // Stride of each output axis in the input (0 where broadcast).
    let mut eff = vec![0usize; n];
    for (i, e) in eff.iter_mut().enumerate() {
        if i + input.len() >= n {
            let j = i + input.len() - n;
            if input[j] != 1 {
                *e = in_strides[j];
            }
        }
    }
    let total = numel(out_shape);
    let mut map = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    let mut flat = 0usize;
    for _ in 0..total {
        map.push(flat);
        // Increment the multi-index.
        for ax in (0..n).rev() {
            idx[ax] += 1;
            flat += eff[ax];
            if idx[ax] < out_shape[ax] {
                break;
            }
            flat -= eff[ax] * idx[ax];
            idx[ax] = 0;
        }
    }
    map
}

/// Sums a gradient of shape `from` down to `to` (undoing broadcasting).
fn reduce_to(grad: &[f64], from: &[usize], to: &[usize]) -> Vec<f64> {
    if from == to {
        return grad.to_vec();
    }
    let map = broadcast_index_map(to, from);
    let mut out = vec![0.0; numel(to)];
    for (g, &i) in grad.iter().zip(&map) {
        out[i] += g;
    }
    out
}

impl Tensor {
    fn make(data: Vec<f64>, shape: Vec<usize>, op: Option<Op>, requires_grad: bool) -> Tensor {
        debug_assert_eq!(data.len(), numel(&shape));
        Tensor(Rc::new(Node {
            data: RefCell::new(data),
            shape,
            grad: RefCell::new(None),
            requires_grad: Cell::new(requires_grad),
            op,
        }))
    }

    /// A tensor with the given data and shape (no autograd).
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> TResult<Tensor> {
        if data.len() != numel(&shape) {
            return Err(format!(
                "{} values do not fill shape {}",
                data.len(),
                shape_str(&shape)
            ));
        }
        Ok(Tensor::make(data, shape, None, false))
    }

    /// A 0-dimensional tensor.
    pub fn scalar(x: f64) -> Tensor {
        Tensor::make(vec![x], Vec::new(), None, false)
    }

    /// All zeros.
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::make(vec![0.0; numel(shape)], shape.to_vec(), None, false)
    }

    /// All ones.
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::make(vec![1.0; numel(shape)], shape.to_vec(), None, false)
    }

    /// Filled with `v`.
    pub fn full(shape: &[usize], v: f64) -> Tensor {
        Tensor::make(vec![v; numel(shape)], shape.to_vec(), None, false)
    }

    /// Identity matrix.
    pub fn eye(n: usize) -> Tensor {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        Tensor::make(d, vec![n, n], None, false)
    }

    /// Identity of the underlying allocation (for `is`).
    pub fn ptr(&self) -> usize {
        Rc::as_ptr(&self.0) as *const u8 as usize
    }

    /// The shape.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        numel(&self.0.shape)
    }

    /// A copy of the data.
    pub fn data(&self) -> Vec<f64> {
        self.0.data.borrow().clone()
    }

    /// Runs `f` over the data without copying.
    pub fn with_data<T>(&self, f: impl FnOnce(&[f64]) -> T) -> T {
        f(&self.0.data.borrow())
    }

    /// Overwrites the data in place (optimizer updates).
    pub fn update(&self, f: impl FnOnce(&mut [f64])) {
        f(&mut self.0.data.borrow_mut());
    }

    /// The single value of a one-element tensor.
    pub fn item(&self) -> f64 {
        self.0.data.borrow().first().copied().unwrap_or(f64::NAN)
    }

    /// Truthiness: non-empty and not all zeros.
    pub fn truthy(&self) -> bool {
        self.0.data.borrow().iter().any(|&x| x != 0.0)
    }

    /// Whether gradients are tracked.
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.get()
    }

    /// Turns gradient tracking on or off (makes this a leaf parameter).
    pub fn set_requires_grad(&self, on: bool) {
        self.0.requires_grad.set(on);
    }

    /// The accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.0
            .grad
            .borrow()
            .as_ref()
            .map(|g| Tensor::make(g.clone(), self.0.shape.clone(), None, false))
    }

    /// The gradient as raw data.
    pub fn grad_data(&self) -> Option<Vec<f64>> {
        self.0.grad.borrow().clone()
    }

    /// Clears the gradient.
    pub fn zero_grad(&self) {
        *self.0.grad.borrow_mut() = None;
    }

    /// A copy that shares no history.
    pub fn detach(&self) -> Tensor {
        Tensor::make(self.data(), self.0.shape.clone(), None, false)
    }

    fn result(&self, data: Vec<f64>, shape: Vec<usize>, op: Op, inputs: &[&Tensor]) -> Tensor {
        let track = grad_enabled() && inputs.iter().any(|t| t.requires_grad());
        Tensor::make(data, shape, track.then_some(op), track)
    }

    // ----- elementwise --------------------------------------------------

    fn binary(
        &self,
        other: &Tensor,
        f: impl Fn(f64, f64) -> f64,
    ) -> TResult<(Vec<f64>, Vec<usize>)> {
        let a = self.0.data.borrow();
        let b = other.0.data.borrow();
        if self.0.shape == other.0.shape {
            return Ok((
                a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect(),
                self.0.shape.clone(),
            ));
        }
        let out = broadcast_shape(&self.0.shape, &other.0.shape)?;
        let ma = broadcast_index_map(&self.0.shape, &out);
        let mb = broadcast_index_map(&other.0.shape, &out);
        Ok((
            ma.iter().zip(&mb).map(|(&i, &j)| f(a[i], b[j])).collect(),
            out,
        ))
    }

    /// Elementwise `self + other` with broadcasting.
    pub fn add(&self, other: &Tensor) -> TResult<Tensor> {
        let (d, s) = self.binary(other, |x, y| x + y)?;
        Ok(self.result(d, s, Op::Add(self.clone(), other.clone()), &[self, other]))
    }

    /// Elementwise `self - other`.
    pub fn sub(&self, other: &Tensor) -> TResult<Tensor> {
        let (d, s) = self.binary(other, |x, y| x - y)?;
        Ok(self.result(d, s, Op::Sub(self.clone(), other.clone()), &[self, other]))
    }

    /// Elementwise `self * other`.
    pub fn mul(&self, other: &Tensor) -> TResult<Tensor> {
        let (d, s) = self.binary(other, |x, y| x * y)?;
        Ok(self.result(d, s, Op::Mul(self.clone(), other.clone()), &[self, other]))
    }

    /// Elementwise `self / other`.
    pub fn div(&self, other: &Tensor) -> TResult<Tensor> {
        let (d, s) = self.binary(other, |x, y| x / y)?;
        Ok(self.result(d, s, Op::Div(self.clone(), other.clone()), &[self, other]))
    }

    /// Elementwise `self ** other`.
    pub fn pow(&self, other: &Tensor) -> TResult<Tensor> {
        let (d, s) = self.binary(other, f64::powf)?;
        Ok(self.result(d, s, Op::Pow(self.clone(), other.clone()), &[self, other]))
    }

    /// Elementwise comparison producing 0/1 (not differentiable).
    pub fn compare(&self, other: &Tensor, f: impl Fn(f64, f64) -> bool) -> TResult<Tensor> {
        let (d, s) = self.binary(other, |x, y| f64::from(u8::from(f(x, y))))?;
        Ok(Tensor::make(d, s, None, false))
    }

    /// `-self`.
    pub fn neg(&self) -> Tensor {
        let d = self.0.data.borrow().iter().map(|x| -x).collect();
        self.result(d, self.0.shape.clone(), Op::Neg(self.clone()), &[self])
    }

    /// Applies an elementwise function.
    pub fn unary(&self, u: Unary) -> Tensor {
        let d = self.0.data.borrow().iter().map(|&x| u.apply(x)).collect();
        self.result(d, self.0.shape.clone(), Op::Unary(self.clone(), u), &[self])
    }

    /// Applies an arbitrary function without autograd (rounding, clipping…).
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let d = self.0.data.borrow().iter().map(|&x| f(x)).collect();
        Tensor::make(d, self.0.shape.clone(), None, false)
    }

    // ----- linear algebra ---------------------------------------------------

    /// Matrix product with NumPy's 1-D/2-D rules.
    pub fn matmul(&self, other: &Tensor) -> TResult<Tensor> {
        let (a_shape, b_shape) = (self.shape().to_vec(), other.shape().to_vec());
        let (m, k1, a2) = match a_shape.len() {
            1 => (1, a_shape[0], false),
            2 => (a_shape[0], a_shape[1], true),
            _ => return Err("matmul supports 1-D and 2-D tensors".into()),
        };
        let (k2, n, b2) = match b_shape.len() {
            1 => (b_shape[0], 1, false),
            2 => (b_shape[0], b_shape[1], true),
            _ => return Err("matmul supports 1-D and 2-D tensors".into()),
        };
        if k1 != k2 {
            return Err(format!(
                "matmul shape mismatch: {} @ {}",
                shape_str(&a_shape),
                shape_str(&b_shape)
            ));
        }
        let a = self.0.data.borrow();
        let b = other.0.data.borrow();
        let out = matmul_raw(&a, &b, m, k1, n);
        let shape = match (a2, b2) {
            (true, true) => vec![m, n],
            (true, false) => vec![m],
            (false, true) => vec![n],
            (false, false) => Vec::new(),
        };
        drop(a);
        drop(b);
        Ok(self.result(
            out,
            shape,
            Op::MatMul(self.clone(), other.clone()),
            &[self, other],
        ))
    }

    /// Transpose (reverses the axes of a 2-D tensor; 1-D is unchanged).
    pub fn transpose(&self) -> Tensor {
        let s = self.shape();
        if s.len() < 2 {
            return self.reshape_unchecked(s.to_vec());
        }
        let (r, c) = (s[0], s[1..].iter().product::<usize>());
        let d = self.0.data.borrow();
        let mut out = vec![0.0; d.len()];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = d[i * c + j];
            }
        }
        drop(d);
        let mut shape: Vec<usize> = s.to_vec();
        shape.reverse();
        if s.len() > 2 {
            shape = vec![c, r];
        }
        self.result(out, shape, Op::Transpose(self.clone()), &[self])
    }

    fn reshape_unchecked(&self, shape: Vec<usize>) -> Tensor {
        self.result(self.data(), shape, Op::Reshape(self.clone()), &[self])
    }

    /// Reshapes; one dimension may be `-1` (inferred).
    pub fn reshape(&self, dims: &[i64]) -> TResult<Tensor> {
        let total = self.numel();
        let known: i64 = dims.iter().filter(|&&d| d >= 0).product();
        let mut shape = Vec::with_capacity(dims.len());
        for &d in dims {
            if d < 0 {
                if known == 0 || total as i64 % known != 0 {
                    return Err(format!("cannot reshape {} values", total));
                }
                shape.push((total as i64 / known) as usize);
            } else {
                shape.push(d as usize);
            }
        }
        if numel(&shape) != total {
            return Err(format!(
                "cannot reshape tensor of shape {} into {}",
                shape_str(self.shape()),
                shape_str(&shape)
            ));
        }
        Ok(self.reshape_unchecked(shape))
    }

    // ----- reductions ---------------------------------------------------------

    fn axis_layout(&self, axis: usize) -> (usize, usize, usize) {
        let s = self.shape();
        let outer: usize = s[..axis].iter().product();
        let inner: usize = s[axis + 1..].iter().product();
        (outer, s[axis], inner)
    }

    fn check_axis(&self, axis: i64) -> TResult<usize> {
        let n = self.ndim() as i64;
        let a = if axis < 0 { axis + n } else { axis };
        if a < 0 || a >= n {
            return Err(format!("axis {axis} out of range for a {n}-D tensor"));
        }
        Ok(a as usize)
    }

    /// Sum of all elements, or along `axis`.
    pub fn sum(&self, axis: Option<i64>) -> TResult<Tensor> {
        match axis {
            None => {
                let s: f64 = self.0.data.borrow().iter().sum();
                Ok(self.result(vec![s], Vec::new(), Op::Sum(self.clone(), None), &[self]))
            }
            Some(a) => {
                let a = self.check_axis(a)?;
                let (outer, len, inner) = self.axis_layout(a);
                let d = self.0.data.borrow();
                let mut out = vec![0.0; outer * inner];
                for o in 0..outer {
                    for l in 0..len {
                        for i in 0..inner {
                            out[o * inner + i] += d[(o * len + l) * inner + i];
                        }
                    }
                }
                drop(d);
                let mut shape = self.shape().to_vec();
                shape.remove(a);
                Ok(self.result(out, shape, Op::Sum(self.clone(), Some(a)), &[self]))
            }
        }
    }

    /// Mean of all elements, or along `axis`.
    pub fn mean(&self, axis: Option<i64>) -> TResult<Tensor> {
        match axis {
            None => {
                let n = self.numel().max(1) as f64;
                let s: f64 = self.0.data.borrow().iter().sum();
                Ok(self.result(
                    vec![s / n],
                    Vec::new(),
                    Op::Mean(self.clone(), None),
                    &[self],
                ))
            }
            Some(a) => {
                let a = self.check_axis(a)?;
                let len = self.shape()[a] as f64;
                let summed = no_grad(|| self.sum(Some(a as i64)))?;
                let data = summed.data().into_iter().map(|x| x / len).collect();
                Ok(self.result(
                    data,
                    summed.shape().to_vec(),
                    Op::Mean(self.clone(), Some(a)),
                    &[self],
                ))
            }
        }
    }

    /// Maximum of all elements, or along `axis` (gradient flows to the max).
    pub fn max(&self, axis: Option<i64>) -> TResult<Tensor> {
        let d = self.0.data.borrow();
        if d.is_empty() {
            return Err("max of an empty tensor".into());
        }
        match axis {
            None => {
                let (i, &m) = d
                    .iter()
                    .enumerate()
                    .fold((0, &f64::NEG_INFINITY), |acc, (i, x)| {
                        if *x > *acc.1 { (i, x) } else { acc }
                    });
                drop(d);
                Ok(self.result(
                    vec![m],
                    Vec::new(),
                    Op::Max(self.clone(), Rc::new(vec![i])),
                    &[self],
                ))
            }
            Some(a) => {
                drop(d);
                let a = self.check_axis(a)?;
                let (outer, len, inner) = self.axis_layout(a);
                let d = self.0.data.borrow();
                let mut out = Vec::with_capacity(outer * inner);
                let mut arg = Vec::with_capacity(outer * inner);
                for o in 0..outer {
                    for i in 0..inner {
                        let mut best = f64::NEG_INFINITY;
                        let mut bi = 0;
                        for l in 0..len {
                            let idx = (o * len + l) * inner + i;
                            if d[idx] > best {
                                best = d[idx];
                                bi = idx;
                            }
                        }
                        out.push(best);
                        arg.push(bi);
                    }
                }
                drop(d);
                let mut shape = self.shape().to_vec();
                shape.remove(a);
                Ok(self.result(out, shape, Op::Max(self.clone(), Rc::new(arg)), &[self]))
            }
        }
    }

    /// Indices of the maxima along `axis` (or of the flat maximum).
    pub fn argmax(&self, axis: Option<i64>) -> TResult<Tensor> {
        let d = self.0.data.borrow();
        if d.is_empty() {
            return Err("argmax of an empty tensor".into());
        }
        let Some(axis) = axis else {
            let best = d
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |acc, (i, &x)| {
                    if x > acc.1 { (i, x) } else { acc }
                });
            return Ok(Tensor::scalar(best.0 as f64));
        };
        drop(d);
        let a = self.check_axis(axis)?;
        let (outer, len, inner) = self.axis_layout(a);
        let d = self.0.data.borrow();
        let mut out = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let mut best = f64::NEG_INFINITY;
                let mut best_l = 0;
                for l in 0..len {
                    let v = d[(o * len + l) * inner + i];
                    if v > best {
                        best = v;
                        best_l = l;
                    }
                }
                out.push(best_l as f64);
            }
        }
        let mut shape = self.shape().to_vec();
        shape.remove(a);
        Tensor::new(out, shape)
    }

    /// Softmax along `axis` (default: last).
    pub fn softmax(&self, axis: Option<i64>) -> TResult<Tensor> {
        let axis = axis.unwrap_or(-1);
        let a = self.check_axis(axis)?;
        let m = no_grad(|| self.max(Some(a as i64)))?;
        let m = m.unsqueeze(a);
        let e = self.sub(&m)?.unary(Unary::Exp);
        let s = e.sum(Some(a as i64))?.unsqueeze(a);
        e.div(&s)
    }

    /// log(softmax(x)) computed stably.
    pub fn log_softmax(&self, axis: Option<i64>) -> TResult<Tensor> {
        let axis = axis.unwrap_or(-1);
        let a = self.check_axis(axis)?;
        let m = no_grad(|| self.max(Some(a as i64)))?.unsqueeze(a);
        let shifted = self.sub(&m)?;
        let lse = shifted
            .unary(Unary::Exp)
            .sum(Some(a as i64))?
            .unary(Unary::Log)
            .unsqueeze(a);
        shifted.sub(&lse)
    }

    /// Inserts a length-1 axis at `axis` (keeps autograd history).
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let mut shape = self.shape().to_vec();
        shape.insert(axis.min(shape.len()), 1);
        self.reshape_unchecked(shape)
    }

    // ----- indexing -------------------------------------------------------------

    /// Gathers the given flat indices into a tensor of `shape`.
    fn gather(&self, indices: Vec<usize>, shape: Vec<usize>) -> Tensor {
        let d = self.0.data.borrow();
        let out = indices.iter().map(|&i| d[i]).collect();
        drop(d);
        self.result(
            out,
            shape,
            Op::Gather(self.clone(), Rc::new(indices)),
            &[self],
        )
    }

    /// NumPy-style indexing: one spec per leading axis.
    pub fn slice(&self, specs: &[SliceSpec]) -> TResult<Tensor> {
        let shape = self.shape().to_vec();
        if specs.len() > shape.len() {
            return Err(format!(
                "too many indices ({}) for a {}-D tensor",
                specs.len(),
                shape.len()
            ));
        }
        let st = strides(&shape);
        // Per-axis list of source positions, and whether the axis is kept.
        let mut axes: Vec<(Vec<usize>, bool)> = Vec::with_capacity(shape.len());
        for (ax, &dim) in shape.iter().enumerate() {
            let spec = specs
                .get(ax)
                .copied()
                .unwrap_or(SliceSpec::Range(None, None, 1));
            match spec {
                SliceSpec::Index(i) => {
                    let i = if i < 0 { i + dim as i64 } else { i };
                    if i < 0 || i >= dim as i64 {
                        return Err(format!(
                            "index {i} out of range for axis {ax} with size {dim}"
                        ));
                    }
                    axes.push((vec![i as usize], false));
                }
                SliceSpec::Range(start, stop, step) => {
                    if step == 0 {
                        return Err("slice step cannot be zero".into());
                    }
                    let positions = slice_positions(dim, start, stop, step);
                    axes.push((positions, true));
                }
            }
        }
        let out_shape: Vec<usize> = axes
            .iter()
            .filter(|(_, keep)| *keep)
            .map(|(p, _)| p.len())
            .collect();
        let total: usize = axes.iter().map(|(p, _)| p.len()).product();
        let mut indices = Vec::with_capacity(total);
        let mut counter = vec![0usize; axes.len()];
        for _ in 0..total {
            let flat: usize = counter
                .iter()
                .enumerate()
                .map(|(ax, &c)| axes[ax].0[c] * st[ax])
                .sum();
            indices.push(flat);
            for ax in (0..axes.len()).rev() {
                counter[ax] += 1;
                if counter[ax] < axes[ax].0.len() {
                    break;
                }
                counter[ax] = 0;
            }
        }
        Ok(self.gather(indices, out_shape))
    }

    /// Concatenates tensors along the first axis.
    pub fn concat(parts: &[Tensor]) -> TResult<Tensor> {
        let Some(first) = parts.first() else {
            return Err("concat needs at least one tensor".into());
        };
        let tail = &first.shape()[1.min(first.ndim())..];
        let mut rows = 0;
        let mut data = Vec::new();
        for p in parts {
            if p.ndim() != first.ndim() || &p.shape()[1.min(p.ndim())..] != tail {
                return Err("concat: shapes must match except along the first axis".into());
            }
            rows += p.shape().first().copied().unwrap_or(1);
            data.extend(p.data());
        }
        let mut shape = vec![rows];
        shape.extend_from_slice(tail);
        let track = grad_enabled() && parts.iter().any(Tensor::requires_grad);
        Ok(Tensor::make(
            data,
            shape,
            track.then(|| Op::Concat(parts.to_vec())),
            track,
        ))
    }

    /// Stacks equally shaped tensors along a new first axis.
    pub fn stack(parts: &[Tensor]) -> TResult<Tensor> {
        let lifted: Vec<Tensor> = parts.iter().map(|p| p.unsqueeze(0)).collect();
        Tensor::concat(&lifted)
    }

    // ----- autograd -----------------------------------------------------------

    /// Back-propagates from this tensor (which should be a scalar loss).
    pub fn backward(&self) {
        // Iterative post-order DFS for a topological order.
        let mut order: Vec<Tensor> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(Tensor, bool)> = vec![(self.clone(), false)];
        while let Some((t, expanded)) = stack.pop() {
            if expanded {
                order.push(t);
                continue;
            }
            if !seen.insert(t.ptr()) {
                continue;
            }
            stack.push((t.clone(), true));
            for input in t.inputs() {
                if input.requires_grad() && !seen.contains(&input.ptr()) {
                    stack.push((input, false));
                }
            }
        }
        {
            let mut g = self.0.grad.borrow_mut();
            let seed = vec![1.0; self.numel()];
            match g.as_mut() {
                Some(existing) => existing.iter_mut().zip(&seed).for_each(|(a, b)| *a += b),
                None => *g = Some(seed),
            }
        }
        for t in order.iter().rev() {
            let Some(op) = &t.0.op else { continue };
            let Some(grad) = t.0.grad.borrow().clone() else {
                continue;
            };
            t.backprop(op, &grad);
        }
    }

    fn inputs(&self) -> Vec<Tensor> {
        match &self.0.op {
            None => Vec::new(),
            Some(op) => match op {
                Op::Add(a, b)
                | Op::Sub(a, b)
                | Op::Mul(a, b)
                | Op::Div(a, b)
                | Op::Pow(a, b)
                | Op::MatMul(a, b) => {
                    vec![a.clone(), b.clone()]
                }
                Op::Neg(a)
                | Op::Unary(a, _)
                | Op::Sum(a, _)
                | Op::Mean(a, _)
                | Op::Max(a, _)
                | Op::Transpose(a)
                | Op::Reshape(a)
                | Op::Gather(a, _) => vec![a.clone()],
                Op::Concat(parts) => parts.clone(),
            },
        }
    }

    fn accumulate(&self, g: Vec<f64>) {
        if !self.requires_grad() {
            return;
        }
        let mut slot = self.0.grad.borrow_mut();
        match slot.as_mut() {
            Some(existing) => existing.iter_mut().zip(&g).for_each(|(a, b)| *a += b),
            None => *slot = Some(g),
        }
    }

    fn backprop(&self, op: &Op, g: &[f64]) {
        let out_shape = self.shape();
        match op {
            Op::Add(a, b) => {
                a.accumulate(reduce_to(g, out_shape, a.shape()));
                b.accumulate(reduce_to(g, out_shape, b.shape()));
            }
            Op::Sub(a, b) => {
                a.accumulate(reduce_to(g, out_shape, a.shape()));
                let neg: Vec<f64> = g.iter().map(|x| -x).collect();
                b.accumulate(reduce_to(&neg, out_shape, b.shape()));
            }
            Op::Mul(a, b) => {
                let (av, bv) = (expand(a, out_shape), expand(b, out_shape));
                let ga: Vec<f64> = g.iter().zip(&bv).map(|(g, b)| g * b).collect();
                let gb: Vec<f64> = g.iter().zip(&av).map(|(g, a)| g * a).collect();
                a.accumulate(reduce_to(&ga, out_shape, a.shape()));
                b.accumulate(reduce_to(&gb, out_shape, b.shape()));
            }
            Op::Div(a, b) => {
                let (av, bv) = (expand(a, out_shape), expand(b, out_shape));
                let ga: Vec<f64> = g.iter().zip(&bv).map(|(g, b)| g / b).collect();
                let gb: Vec<f64> = g
                    .iter()
                    .zip(av.iter().zip(&bv))
                    .map(|(g, (a, b))| -g * a / (b * b))
                    .collect();
                a.accumulate(reduce_to(&ga, out_shape, a.shape()));
                b.accumulate(reduce_to(&gb, out_shape, b.shape()));
            }
            Op::Pow(a, b) => {
                let (av, bv) = (expand(a, out_shape), expand(b, out_shape));
                let out = self.0.data.borrow();
                let ga: Vec<f64> = (0..g.len())
                    .map(|i| {
                        if bv[i] == 0.0 {
                            0.0
                        } else {
                            g[i] * bv[i] * av[i].powf(bv[i] - 1.0)
                        }
                    })
                    .collect();
                let gb: Vec<f64> = (0..g.len())
                    .map(|i| {
                        if av[i] > 0.0 {
                            g[i] * out[i] * av[i].ln()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                drop(out);
                a.accumulate(reduce_to(&ga, out_shape, a.shape()));
                b.accumulate(reduce_to(&gb, out_shape, b.shape()));
            }
            Op::Neg(a) => a.accumulate(g.iter().map(|x| -x).collect()),
            Op::Unary(a, u) => {
                let x = a.0.data.borrow();
                let y = self.0.data.borrow();
                let ga = (0..g.len())
                    .map(|i| g[i] * u.derivative(x[i], y[i]))
                    .collect();
                drop(x);
                drop(y);
                a.accumulate(ga);
            }
            Op::MatMul(a, b) => {
                let (a_shape, b_shape) = (a.shape().to_vec(), b.shape().to_vec());
                let (m, k) = if a_shape.len() == 2 {
                    (a_shape[0], a_shape[1])
                } else {
                    (1, a_shape[0])
                };
                let n = if b_shape.len() == 2 { b_shape[1] } else { 1 };
                let av = a.data();
                let bv = b.data();
                // dA = G (m×n) · Bᵀ (n×k); dB = Aᵀ (k×m) · G (m×n)
                let bt = transpose_raw(&bv, k, n);
                let at = transpose_raw(&av, m, k);
                a.accumulate(matmul_raw(g, &bt, m, n, k));
                b.accumulate(matmul_raw(&at, g, k, m, n));
            }
            Op::Sum(a, axis) => match axis {
                None => a.accumulate(vec![g[0]; a.numel()]),
                Some(ax) => {
                    let (outer, len, inner) = a.axis_layout(*ax);
                    let mut ga = vec![0.0; a.numel()];
                    for o in 0..outer {
                        for l in 0..len {
                            for i in 0..inner {
                                ga[(o * len + l) * inner + i] = g[o * inner + i];
                            }
                        }
                    }
                    a.accumulate(ga);
                }
            },
            Op::Mean(a, axis) => match axis {
                None => {
                    let n = a.numel().max(1) as f64;
                    a.accumulate(vec![g[0] / n; a.numel()]);
                }
                Some(ax) => {
                    let (outer, len, inner) = a.axis_layout(*ax);
                    let mut ga = vec![0.0; a.numel()];
                    for o in 0..outer {
                        for l in 0..len {
                            for i in 0..inner {
                                ga[(o * len + l) * inner + i] = g[o * inner + i] / len as f64;
                            }
                        }
                    }
                    a.accumulate(ga);
                }
            },
            Op::Max(a, idx) | Op::Gather(a, idx) => {
                let mut ga = vec![0.0; a.numel()];
                for (gi, &i) in g.iter().zip(idx.iter()) {
                    ga[i] += gi;
                }
                a.accumulate(ga);
            }
            Op::Transpose(a) => {
                let s = a.shape();
                if s.len() < 2 {
                    a.accumulate(g.to_vec());
                } else {
                    let (r, c) = (s[0], s[1..].iter().product::<usize>());
                    a.accumulate(transpose_raw(g, c, r));
                }
            }
            Op::Reshape(a) => a.accumulate(g.to_vec()),
            Op::Concat(parts) => {
                let mut offset = 0;
                for p in parts {
                    let n = p.numel();
                    p.accumulate(g[offset..offset + n].to_vec());
                    offset += n;
                }
            }
        }
    }

    // ----- display ----------------------------------------------------------

    /// Nested-bracket rendering like NumPy.
    pub fn display(&self) -> String {
        let d = self.0.data.borrow();
        let shape = self.shape();
        if shape.is_empty() {
            return format!("tensor({})", fmt_num(d.first().copied().unwrap_or(0.0)));
        }
        let mut out = String::from("tensor(");
        render(&d, shape, 0, &mut 0, &mut out, 7);
        if self.requires_grad() {
            out.push_str(", grad=true");
        }
        out.push(')');
        out
    }
}

fn render(
    d: &[f64],
    shape: &[usize],
    depth: usize,
    pos: &mut usize,
    out: &mut String,
    indent: usize,
) {
    out.push('[');
    let n = shape[depth];
    for i in 0..n {
        if i > 0 {
            out.push(',');
            if depth + 1 < shape.len() {
                out.push('\n');
                out.push_str(&" ".repeat(indent + depth + 1));
            } else {
                out.push(' ');
            }
        }
        if depth + 1 == shape.len() {
            out.push_str(&fmt_num(d[*pos]));
            *pos += 1;
        } else {
            render(d, shape, depth + 1, pos, out, indent);
        }
    }
    out.push(']');
}

/// Formats a tensor element compactly.
pub fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{x:.0}.")
    } else if x.abs() >= 1e-4 && x.abs() < 1e6 {
        let s = format!("{x:.4}");
        let s = s.trim_end_matches('0');
        s.to_owned()
    } else {
        format!("{x:.4e}")
    }
}

impl Tensor {
    /// The data broadcast to `shape` (which must be broadcast-compatible).
    pub fn broadcast_data(&self, shape: &[usize]) -> Vec<f64> {
        expand(self, shape)
    }
}

/// Data of `t` broadcast to `shape`.
fn expand(t: &Tensor, shape: &[usize]) -> Vec<f64> {
    let d = t.0.data.borrow();
    if t.shape() == shape {
        return d.clone();
    }
    broadcast_index_map(t.shape(), shape)
        .into_iter()
        .map(|i| d[i])
        .collect()
}

fn slice_positions(dim: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<usize> {
    let n = dim as i64;
    let norm = |v: i64| if v < 0 { v + n } else { v };
    let mut out = Vec::new();
    if step > 0 {
        let s = start.map_or(0, norm).clamp(0, n);
        let e = stop.map_or(n, norm).clamp(0, n);
        let mut i = s;
        while i < e {
            out.push(i as usize);
            i += step;
        }
    } else {
        let s = start.map_or(n - 1, norm).clamp(-1, n - 1);
        let e = stop.map_or(-1, norm).clamp(-1, n - 1);
        let mut i = s;
        while i > e {
            out.push(i as usize);
            i += step;
        }
    }
    out
}

/// Row-major (m×k)·(k×n) with an i-k-j loop for cache friendliness.
fn matmul_raw(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            let row = &b[p * n..p * n + n];
            let dst = &mut out[i * n..i * n + n];
            for (d, &bv) in dst.iter_mut().zip(row) {
                *d += av * bv;
            }
        }
    }
    out
}

fn transpose_raw(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    fn param(data: &[f64], shape: &[usize]) -> Tensor {
        let x = t(data, shape);
        x.set_requires_grad(true);
        x
    }

    #[test]
    fn broadcasting_add() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        let col = t(&[100.0, 200.0], &[2, 1]);
        assert_eq!(
            a.add(&col).unwrap().data(),
            vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
        );
        assert!(a.add(&t(&[1.0, 2.0], &[2])).is_err());
    }

    #[test]
    fn matmul_shapes() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        assert_eq!(a.matmul(&b).unwrap().data(), vec![19.0, 22.0, 43.0, 50.0]);
        let v = t(&[1.0, 1.0], &[2]);
        assert_eq!(a.matmul(&v).unwrap().shape(), &[2]);
        assert_eq!(v.matmul(&v).unwrap().item(), 2.0);
        assert!(a.matmul(&t(&[1.0, 2.0, 3.0], &[3])).is_err());
    }

    #[test]
    fn gradients_of_polynomial() {
        // f(x) = sum(x^2 * 3 + x) → df/dx = 6x + 1
        let x = param(&[1.0, 2.0, -3.0], &[3]);
        let y = x
            .pow(&Tensor::scalar(2.0))
            .unwrap()
            .mul(&Tensor::scalar(3.0))
            .unwrap()
            .add(&x)
            .unwrap();
        y.sum(None).unwrap().backward();
        assert_eq!(x.grad_data().unwrap(), vec![7.0, 13.0, -17.0]);
    }

    #[test]
    fn matmul_gradients_match_finite_differences() {
        let a = param(&[0.5, -1.0, 2.0, 0.3, 0.7, -0.2], &[2, 3]);
        let b = param(&[1.0, 2.0, -1.0, 0.5, 0.25, 3.0], &[3, 2]);
        let f = |a: &Tensor, b: &Tensor| a.matmul(b).unwrap().unary(Unary::Tanh).sum(None).unwrap();
        f(&a, &b).backward();
        let ga = a.grad_data().unwrap();
        let eps = 1e-6;
        for i in 0..6 {
            let mut d = a.data();
            d[i] += eps;
            let plus = no_grad(|| f(&t(&d, &[2, 3]), &b.detach()).item());
            d[i] -= 2.0 * eps;
            let minus = no_grad(|| f(&t(&d, &[2, 3]), &b.detach()).item());
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (numeric - ga[i]).abs() < 1e-5,
                "grad {i}: {numeric} vs {}",
                ga[i]
            );
        }
    }

    #[test]
    fn broadcast_gradients_are_reduced() {
        let w = param(&[1.0, 2.0], &[2]);
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        x.mul(&w).unwrap().sum(None).unwrap().backward();
        assert_eq!(w.grad_data().unwrap(), vec![9.0, 12.0]);
    }

    #[test]
    fn reductions_and_softmax() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(x.sum(Some(0)).unwrap().data(), vec![5.0, 7.0, 9.0]);
        assert_eq!(x.sum(Some(1)).unwrap().data(), vec![6.0, 15.0]);
        assert_eq!(x.mean(None).unwrap().item(), 3.5);
        assert_eq!(x.max(Some(1)).unwrap().data(), vec![3.0, 6.0]);
        assert_eq!(x.argmax(Some(1)).unwrap().data(), vec![2.0, 2.0]);
        let s = x.softmax(None).unwrap();
        let rows = s.sum(Some(1)).unwrap().data();
        assert!(rows.iter().all(|r| (r - 1.0).abs() < 1e-12));
        let ls = x.log_softmax(None).unwrap().data();
        let sm = s.data();
        for (a, b) in ls.iter().zip(&sm) {
            assert!((a - b.ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn slicing_and_its_gradient() {
        let x = param(&(0..12).map(f64::from).collect::<Vec<_>>(), &[3, 4]);
        let row = x.slice(&[SliceSpec::Index(1)]).unwrap();
        assert_eq!(row.data(), vec![4.0, 5.0, 6.0, 7.0]);
        let col = x
            .slice(&[SliceSpec::Range(None, None, 1), SliceSpec::Index(-1)])
            .unwrap();
        assert_eq!(col.data(), vec![3.0, 7.0, 11.0]);
        let rev = x
            .slice(&[
                SliceSpec::Range(None, None, -1),
                SliceSpec::Range(Some(1), Some(3), 1),
            ])
            .unwrap();
        assert_eq!(rev.shape(), &[3, 2]);
        assert_eq!(rev.data(), vec![9.0, 10.0, 5.0, 6.0, 1.0, 2.0]);
        col.sum(None).unwrap().backward();
        let g = x.grad_data().unwrap();
        assert_eq!(g.iter().sum::<f64>(), 3.0);
        assert_eq!(g[3], 1.0);
    }

    #[test]
    fn reshape_transpose_concat() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(x.reshape(&[3, -1]).unwrap().shape(), &[3, 2]);
        assert!(x.reshape(&[4, -1]).is_err());
        let tt = x.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let c = Tensor::concat(&[x.clone(), x.clone()]).unwrap();
        assert_eq!(c.shape(), &[4, 3]);
        let s = Tensor::stack(&[t(&[1.0, 2.0], &[2]), t(&[3.0, 4.0], &[2])]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
    }

    #[test]
    fn no_grad_disables_history() {
        let x = param(&[1.0], &[1]);
        let y = no_grad(|| x.mul(&x).unwrap());
        assert!(!y.requires_grad());
        let z = x.mul(&x).unwrap();
        assert!(z.requires_grad());
    }

    #[test]
    fn deep_graph_backward_is_iterative() {
        let x = param(&[1.0], &[1]);
        let mut y = x.clone();
        for _ in 0..20_000 {
            y = y.add(&Tensor::scalar(0.0)).unwrap();
        }
        y.sum(None).unwrap().backward();
        assert_eq!(x.grad_data().unwrap(), vec![1.0]);
    }

    #[test]
    fn display_formats() {
        assert_eq!(Tensor::scalar(3.0).display(), "tensor(3.)");
        assert_eq!(t(&[1.0, 2.5], &[2]).display(), "tensor([1., 2.5])");
        assert_eq!(
            t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).display(),
            "tensor([[1., 2.],\n        [3., 4.]])"
        );
    }
}
