use crate::tensor::backward::{BinaryBackwardFn, UnaryBackwardFn};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use ndarray::NdFloat;

use super::tensor::TensorRef;

pub enum Node<T: NdFloat> {
    Unary {
        tensor: TensorRef<T>,
        backward_fn: UnaryBackwardFn<T>,
    },
    Binary {
        tensors: (TensorRef<T>, TensorRef<T>),
        backward_fn: BinaryBackwardFn,
    },
}

impl<T: NdFloat> Clone for Node<T> {
    fn clone(&self) -> Self {
        match &self {
            Node::Unary {
                tensor,
                backward_fn,
            } => Node::Unary {
                tensor: tensor.clone(),
                backward_fn: backward_fn.clone(),
            },
            Node::Binary {
                tensors,
                backward_fn,
            } => Node::Binary {
                tensors: tensors.clone(),
                backward_fn: backward_fn.clone(),
            },
        }
    }
}

impl<T: NdFloat + fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Unary {
                tensor,
                backward_fn,
            } => {
                write!(
                    f,
                    "Node::Unary {{
    tensor: {:#?},
    backward_fn: {:#?}
}}",
                    tensor, backward_fn
                )
            }
            Node::Binary {
                tensors,
                backward_fn,
            } => {
                write!(
                    f,
                    "Node::Binary {{
    tensors: {:#?},
    backward_fn: {:#?}
}}",
                    tensors, backward_fn
                )
            }
        }
    }
}
