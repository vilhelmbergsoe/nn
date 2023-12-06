use crate::tensor::backward::{BinaryBackward, UnaryBackward};
use crate::tensor::node::Node;
use ndarray::{arr0, arr1, arr2, Array2, ArrayD, NdFloat};
use std::cell::{Ref, RefCell, RefMut};
use std::fmt;
use std::rc::Rc;

use super::backward::{BinaryBackwardFn, UnaryBackwardFn};

#[derive(Debug, Clone)]
pub struct TensorRef<T: NdFloat> {
    pub _ref: Rc<RefCell<Tensor<T>>>,
}

impl<T: NdFloat + fmt::Debug> TensorRef<T> {
    pub fn new(tensor: Tensor<T>) -> TensorRef<T> {
        Self {
            _ref: Rc::new(RefCell::new(tensor)),
        }
    }

    pub fn borrow(&self) -> Ref<Tensor<T>> {
        self._ref.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<Tensor<T>> {
        self._ref.borrow_mut()
    }

    pub fn backward(&mut self) {
        self.borrow_mut().backward();
    }

    pub fn grad(&self) -> Option<ArrayD<T>> {
        self.borrow().grad.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T: NdFloat> {
    pub data: ArrayD<T>,
    pub grad_fn: Option<Box<Node<T>>>,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<T>>,
    pub is_leaf: bool,
}

impl<T: NdFloat + fmt::Debug> Tensor<T> {
    pub fn new(data: ArrayD<T>) -> Self {
        Self {
            data,
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    /// Initialize with the requires_grad flag set
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Do backward pass and compute gradients for tree, initializing the
    /// gradient to 1
    pub fn backward(&mut self) {
        if self.data.shape().is_empty() {
            self.backward_grad(&arr0(T::one()).into_dyn());
        } else {
            panic!("Error: Gradient computation only supports scalar outputs. Make sure you are calling backward on a scalar tensor.");
        }
    }

    pub fn backward_grad(&mut self, grad: &ArrayD<T>) {
        if let Some(ref mut grad_fn) = self.grad_fn {
            match **grad_fn {
                Node::Binary {
                    ref mut tensors,
                    ref mut backward_fn,
                } => match backward_fn {
                    BinaryBackwardFn::Add(add) => {
                        add.backward(tensors.clone(), &grad);
                    }
                    BinaryBackwardFn::Mul(mul) => {
                        mul.backward(tensors.clone(), &grad);
                    }
                    BinaryBackwardFn::Sub(sub) => {
                        sub.backward(tensors.clone(), &grad);
                    }
                },
                Node::Unary {
                    ref mut tensor,
                    ref mut backward_fn,
                } => match backward_fn {
                    UnaryBackwardFn::Pow(pow) => {
                        pow.backward(tensor.clone(), &grad);
                    }
                    UnaryBackwardFn::Relu(relu) => {
                        relu.backward(tensor.clone(), &grad);
                    }
                    UnaryBackwardFn::Mean(mean) => {
                        mean.backward(tensor.clone(), &grad);
                    }
                },
            };
        }
    }

    pub fn as_ref(&self) -> TensorRef<T> {
        TensorRef::new(self.clone())
    }
}

// Implement from T for Tensor
impl<T: NdFloat + fmt::Debug> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::new(arr0(value).into_dyn())
    }
}

// Implement from &[T; N] for Tensor
impl<T: NdFloat + fmt::Debug, const N: usize> From<&[T; N]> for Tensor<T> {
    fn from(slice: &[T; N]) -> Self {
        Tensor::new(arr1(slice).into_dyn())
    }
}

// Implement From<&[[T; N]; N]> for Tensor
impl<T: NdFloat + fmt::Debug, const N: usize, const M: usize> From<&[[T; N]; M]> for Tensor<T> {
    fn from(array: &[[T; N]; M]) -> Self {
        let data = Array2::from_shape_fn((M, N), |(i, j)| array[i][j]);

        Tensor::new(data.into_dyn())
    }
}

impl<T: NdFloat + fmt::Debug> fmt::Display for TensorRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.borrow().data)
    }
}

#[macro_export]
macro_rules! tensor {
    ($data:expr) => {
        Tensor::from($data).as_ref()
    };
    ($data:expr, requires_grad) => {
        Tensor::from($data).with_grad().as_ref()
    };
}
