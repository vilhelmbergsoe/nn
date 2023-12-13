use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{RwLock, Arc};
use crate::dtype::DType;
use crate::shape::Shape;
use crate::storage::Storage;
use crate::device::{Device, NdArray};

// Here each field that needs mutability wrapped in an Arc<RwLock<>> if cloning
// might happen or a RwLock if it just needs mutability. This might be wrong but
// that's what i'm going with for now
#[derive(Clone)]
pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>,
    grad_fn: Option<BackpropOp>,
    is_leaf: bool,
    requires_grad: bool,

    // TODO: RwLock might not be needed. Understand how candle mutates the
    // is_variable field
    grad: Arc<RwLock<Option<Tensor>>>,
    dtype: DType,
}

#[derive(Clone)]
pub struct Tensor(Rc<Tensor_>);

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

fn from_storage(
    storage: Storage,
    shape: Shape,
    op: BackpropOp,
    requires_grad: bool,
    is_leaf: bool,
) -> Tensor {
    let dtype = storage.dtype();
    let tensor_ = Tensor_ {
        storage: Arc::new(RwLock::new(storage)),
        grad_fn: op,
        grad: Arc::new(RwLock::new(None)),
        dtype,
        is_leaf,
        requires_grad,
    };
    Tensor(Rc::new(tensor_))
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
