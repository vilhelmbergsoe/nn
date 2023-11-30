use crate::tensor::backward::{BackwardFn, AddBackward, MulBackward};
use crate::tensor::node::Node;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

impl<T: Float + fmt::Debug> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        let result_data = &self.data * &other.data;
        let mut result = Tensor::new(result_data);

        // Propagate requires_grad if any of the input tensors requires_grad
        result.requires_grad = self.requires_grad || other.requires_grad;

        // Initialize grad with None
        result.grad = None;

        if result.requires_grad {
            // Set grad_fn for result tensor
            result.grad_fn = Some(Box::new(Node {
                saved_tensors: vec![Box::new(self), Box::new(other)],
                backward_fn: BackwardFn::Mul(MulBackward),
                // op: Some(Operation::Mul),
            }));
            result.is_leaf = false;
        }

        result
    }
}

impl<T: Float + fmt::Debug> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        let result_data = &self.data + &other.data;
        let mut result = Tensor::new(result_data);

        // Propagate requires_grad if any of the input tensors requires_grad
        result.requires_grad = self.requires_grad || other.requires_grad;

        // Initialize grad with None
        result.grad = None;

        if result.requires_grad {
            // Set grad_fn for result tensor
            result.grad_fn = Some(Box::new(Node {
                saved_tensors: vec![Box::new(self), Box::new(other)],
                backward_fn: BackwardFn::Add(AddBackward),
            }));
        }

        result
    }
}
