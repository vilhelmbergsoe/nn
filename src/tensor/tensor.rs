use crate::tensor::backward::{Backward, BackwardFn};
use crate::tensor::node::Node;
use ndarray::{arr0, arr1, arr2, ArrayD, Array2};
use num_traits::Float;
use std::borrow::BorrowMut;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Tensor<T: Float> {
    pub data: ArrayD<T>,
    pub grad_fn: Option<Box<Node<T>>>,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<T>>,
    pub is_leaf: bool,
}

impl<T: Float + fmt::Debug> Tensor<T> {
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

    /// Do backward pass and compute gradients for tree
    pub fn backward(&mut self) {
        let grad = ArrayD::from_elem(self.data.shape(), T::from(1.0).unwrap());
        self.backward_grad(&grad);
    }

    pub fn backward_grad(&mut self, grad: &ArrayD<T>) {
        if let Some(ref mut grad_fn) = self.grad_fn {
            match &grad_fn.backward_fn {
                BackwardFn::Mul(mul_backward) => {
                    let mut tensors = grad_fn.saved_tensors.borrow_mut();
                    mul_backward.backward(&mut tensors, &grad);
                }
                BackwardFn::Add(add_backward) => {
                    let mut tensors = grad_fn.saved_tensors.borrow_mut();
                    add_backward.backward(&mut tensors, &grad);
                }
                BackwardFn::Relu(relu_backward) => {
                    relu_backward.backward(&mut grad_fn.saved_tensors.borrow_mut(), &grad)
                }
                _ => todo!(),
            };
        }
    }

}

// Implement from T for Tensor
impl<T: Float + fmt::Debug> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::new(arr0(value).into_dyn())
    }
}

// Implement from &[T; N] for Tensor
impl<T: Float + fmt::Debug, const N: usize> From<&[T; N]> for Tensor<T> {
    fn from(slice: &[T; N]) -> Self {
        Tensor::new(arr1(slice).into_dyn())
    }
}

// Implement From<&[[T; N]; N]> for Tensor
impl<T: Float + fmt::Debug, const N: usize, const M: usize> From<&[[T; N]; M]> for Tensor<T> {
    fn from(array: &[[T; N]; M]) -> Self {
        let data = Array2::from_shape_fn((M, N), |(i, j)| array[i][j]);

        Tensor::new(data.into_dyn())
    }
}

impl<T: Float + fmt::Debug> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
