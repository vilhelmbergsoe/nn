use crate::tensor::backward::{
    AddBackward,
    BinaryBackwardFn,
    MulBackward,
    PowBackward, // , MulBackward, PowBackward
    UnaryBackwardFn,
};
use crate::tensor::node::Node;
use crate::tensor::tensor::TensorRef;
use crate::tensor::Tensor;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use ndarray::NdFloat;

// TODO: fix reshaping of the output.
impl<T: NdFloat + fmt::Debug> Mul for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn mul(self, other: &TensorRef<T>) -> TensorRef<T> {
        // Reshape data in order to avoid broadcasting issues.
        // TODO: implement dot on TensorRef or fix underlying multiplication issue
        let binding = self.borrow();
        let reshaped_self = binding.data.view().into_shape(self.borrow().data.len()).unwrap();
        let binding = other.borrow();
        let reshaped_other = binding.data.view().into_shape((self.borrow().data.len(), other.borrow().data.shape()[1])).unwrap();
        let result_data = reshaped_self.dot(&reshaped_other).into_dyn();
        let mut result = Tensor::new(result_data);

        result.requires_grad = self.borrow().requires_grad || other.borrow().requires_grad;
        result.grad = None;

        if result.requires_grad {
            result.grad_fn = Some(Box::new(Node::Binary {
                tensors: (self.clone(), other.clone()),
                backward_fn: BinaryBackwardFn::Mul(MulBackward),
            }));
            result.is_leaf = false;
        }

        result.as_ref()
    }
}

impl<T: NdFloat + fmt::Debug> Add for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: &TensorRef<T>) -> TensorRef<T> {
        let result_data = &self.borrow().data + &other.borrow().data;
        let mut result = Tensor::new(result_data);

        result.requires_grad = self.borrow().requires_grad || other.borrow().requires_grad;
        result.grad = None;

        if result.requires_grad {
            result.grad_fn = Some(Box::new(Node::Binary {
                tensors: (self.clone(), other.clone()),
                backward_fn: BinaryBackwardFn::Add(AddBackward),
            }));
            result.is_leaf = false;
        }

        result.as_ref()
    }
}

impl<T: NdFloat + fmt::Debug> TensorRef<T> {
    /// Element-wise power operation
    pub fn pow(&self, exponent: T) -> TensorRef<T> {
        let result_data = self.borrow().data.mapv(|val| val.powf(exponent));
        let mut result = Tensor::new(result_data);

        // Set requires_grad and grad_fn for the result tensor
        result.requires_grad = self.borrow().requires_grad;
        result.grad_fn = Some(Box::new(Node::Unary {
            tensor: self.clone(),
            backward_fn: UnaryBackwardFn::Pow(PowBackward(exponent)),
        }));
        result.is_leaf = false;

        result.as_ref()
    }

    // /// Calculate the mean of all Tensor elements
    // pub fn mean(&self) -> Option<Tensor<T>> {
    //     if let Some(mean_data) = self.data.mean() {
    //         let mut result = Tensor::from(mean_data);

    //         result.requires_grad = self.requires_grad;

    //         if result.requires_grad {
    //             // Set grad_fn for result tensor
    //             result.grad_fn = Some(Box::new(Node {
    //                 saved_tensors: vec![Box::new(self)],
    //                 backward_fn: BackwardFn::Mean(mean),
    //             }));
    //             result.is_leaf = false;
    //         }

    //         Some(result)
    //     } else {
    //         None
    //     }
    // }

    // /// Calculate the sum of all Tensor elements
    // pub fn sum(&self, axis: usize) -> Tensor<T> {
    //     let sum_data = self.data.sum();
    //     let mut result = Tensor::from(sum_data);

    //     result.requires_grad = self.requires_grad;

    //     if result.requires_grad {
    //         // Set grad_fn for result tensor
    //         result.grad_fn = Some(Box::new(Node {
    //             saved_tensors: vec![Box::new(self)],
    //             backward_fn: BackwardFn::Sum(sum),
    //         }));
    //         result.is_leaf = false;
    //     }

    //     result
    // }
}
