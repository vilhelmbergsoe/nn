use crate::tensor::backward::{
    AddBackward,
    BinaryBackwardFn,
    DivBackward,
    MeanBackward,
    MulBackward,
    PowBackward, // , MulBackward, PowBackward
    SubBackward,
    UnaryBackwardFn,
};
use crate::tensor::node::Node;
use crate::tensor::tensor::TensorRef;
use crate::tensor::Tensor;
use ndarray::NdFloat;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use ndarray::linalg::Dot;

// TODO: fix reshaping of the output.
impl<T: NdFloat + fmt::Debug> Mul for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn mul(self, other: &TensorRef<T>) -> TensorRef<T> {
        // Reshape data in order to avoid broadcasting issues.
        // TODO: implement dot on TensorRef or fix underlying multiplication issue
        let binding = self.borrow();
        let reshaped_self = binding
            .data
            .view()
            .into_shape(self.borrow().data.len())
            .unwrap();
        let binding = other.borrow();
        let reshaped_other = binding
            .data
            .view()
            .into_shape((self.borrow().data.len(), other.borrow().data.shape()[1]))
            .unwrap();
        let result_data = reshaped_self.dot(&reshaped_other).into_dyn();
        // let result_data = self.borrow().data.dot(&other.borrow().data);
        let mut result = Tensor::new(result_data.into_dyn());

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

impl<T: NdFloat + fmt::Debug> Sub for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn sub(self, other: &TensorRef<T>) -> TensorRef<T> {
        let result_data = &self.borrow().data - &other.borrow().data;
        let mut result = Tensor::new(result_data);

        result.requires_grad = self.borrow().requires_grad || other.borrow().requires_grad;
        result.grad = None;

        if result.requires_grad {
            result.grad_fn = Some(Box::new(Node::Binary {
                tensors: (self.clone(), other.clone()),
                backward_fn: BinaryBackwardFn::Sub(SubBackward),
            }));
            result.is_leaf = false;
        }

        result.as_ref()
    }
}

impl<T: NdFloat + fmt::Debug> Div for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn div(self, other: &TensorRef<T>) -> TensorRef<T> {
        let result_data = &self.borrow().data / &other.borrow().data;
        let mut result = Tensor::new(result_data);

        result.requires_grad = self.borrow().requires_grad || other.borrow().requires_grad;
        result.grad = None;

        if result.requires_grad {
            result.grad_fn = Some(Box::new(Node::Binary {
                tensors: (self.clone(), other.clone()),
                backward_fn: BinaryBackwardFn::Div(DivBackward),
            }));
            result.is_leaf = false;
        }

        result.as_ref()
    }
}

impl<T: NdFloat + fmt::Debug + num_traits::cast::FromPrimitive> TensorRef<T> {
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

    /// Calculate the mean of all Tensor elements
    pub fn mean(&self) -> Option<TensorRef<T>> {
        if let Some(mean_data) = self.borrow().data.mean() {
            let mut result = Tensor::from(mean_data);

            result.requires_grad = self.borrow().requires_grad;

            if result.requires_grad {
                // Set grad_fn for result tensor
                result.grad_fn = Some(Box::new(Node::Unary {
                    tensor: self.clone(),
                    backward_fn: UnaryBackwardFn::Mean(MeanBackward),
                }));
                result.is_leaf = false;
            }

            Some(result.as_ref())
        } else {
            None
        }
    }

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
