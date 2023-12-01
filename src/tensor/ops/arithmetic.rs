use crate::tensor::backward::{AddBackward, BackwardFn, MulBackward, PowBackward};
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
            result.is_leaf = false;
        }

        result
    }
}

impl<T: Float + fmt::Debug> Tensor<T> {
    /// Element-wise power operation
    pub fn pow(&self, exponent: T) -> Tensor<T> {
        let result_data = self.data.mapv(|val| val.powf(exponent));
        let mut result = Tensor::new(result_data);

        // Set requires_grad and grad_fn for the result tensor
        result.requires_grad = self.requires_grad;
        result.grad_fn = Some(Box::new(Node {
            saved_tensors: vec![Box::new(self.clone())],
            backward_fn: BackwardFn::Pow(PowBackward(exponent)),
        }));
        result.is_leaf = false;

        result
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
