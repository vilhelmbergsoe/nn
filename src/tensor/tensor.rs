use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor_ {
    storage: Storage,
    grad_fn: Option<BackpropOp>,
    is_leaf: bool,
    requires_grad: bool,
    grad: Option<Tensor>,
    dtype: DType,
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<Tensor_>>);

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

#[derive(Clone)]
pub struct Shape(Vec<usize>);

trait NdArray {
    // TODO: Option / Result?
    fn shape(&self) -> Option<Shape>;
}

impl<T: WithDType> NdArray for T {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(()))
    }
}

impl<T: WithDType, const N: usize> NdArray for &[T; N] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(self.len()))
    }
}

impl<T: WithDType> NdArray for &[T] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(self.len()))
    }
}

impl<T: WithDType, const N: usize, const M: usize> NdArray for &[[T; N]; M] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from((M, N)))
    }
}

impl<T: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[T; N3]; N2]; N1]
{
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from((N1, N2, N3)))
    }
}

enum DType {
    F32,
    F64,
}

trait WithDType {
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }
        }
    };
}

with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);

fn from_storage(
    storage: Storage,
    shape: Shape,
    op: BackpropOp,
    requires_grad: bool,
    is_leaf: bool,
) -> Tensor {
    let dtype = storage.dtype();
    let tensor_ = Tensor_ {
        storage,
        grad_fn: op,
        grad: None,
        dtype,
        is_leaf,
        requires_grad,
    };
    Tensor(Rc::new(RefCell::new(tensor_)))
}

impl Tensor {
    pub fn new<A: NdArray>(
        arr: A,
        device: &Device,
        requires_grad: bool,
        is_leaf: bool,
    ) -> Option<Self> {
        let shape = arr.shape()?;
        Self::new_impl(arr, shape, device, requires_grad, is_leaf);
    }

    pub fn new_impl<A: NdArray>(
        arr: A,
        shape: Shape,
        device: &Device,
        requires_grad: bool,
        is_leaf: bool,
    ) -> Option<Tensor> {
        let n: usize = shape.elem_count();
        let buf_size: usize = arr.shape()?.elem_count();
        if buf_size != n {
            // TODO: implement err
            return None;
        }
        let storage = device.storage(arr)?;
        let op: Option<BackpropOp> = None;
        Ok(from_storage(
            storage,
            shape.clone(),
            op,
            requires_grad,
            is_leaf,
        ))
    }
}
