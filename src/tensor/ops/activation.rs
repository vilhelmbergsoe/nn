use core::fmt;

use crate::tensor::backward::{UnaryBackwardFn, ReluBackward};
use crate::tensor::node::Node;
use crate::tensor::Tensor;
use crate::tensor::tensor::TensorRef;
use num_traits::Float;

// Relu'(0.0) = 0.0 Not sure if this is right
pub fn relu<T: Float + fmt::Debug>(tensor: &TensorRef<T>) -> TensorRef<T> {
    let x = tensor.borrow();
    let data = x.data.mapv(|val| {
        if val < T::from(0.0).unwrap() {
            T::from(0.0).unwrap()
        } else {
            val
        }
    });

    TensorRef::new(
        Tensor {
        data,
        grad_fn: Some(Box::new(Node::Unary {
            tensor: tensor.clone(),
            backward_fn: UnaryBackwardFn::Relu(ReluBackward),
        })),
        requires_grad: x.requires_grad,
        grad: None,
        is_leaf: false,
    })
}
