use crate::tensor::backward::{BackwardFn, ReluBackward};
use crate::tensor::node::Node;
use crate::tensor::Tensor;
use num_traits::Float;

// Relu'(0.0) = 0.0 Not sure if this is right
pub fn relu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
    let data = x.data.mapv(|val| {
        if val < T::from(0.0).unwrap() {
            T::from(0.0).unwrap()
        } else {
            val
        }
    });
    Tensor {
        data,
        grad_fn: Some(Box::new(Node {
            saved_tensors: vec![Box::new(x.clone())],
            backward_fn: BackwardFn::Relu(ReluBackward),
        })),
        requires_grad: x.requires_grad,
        grad: None,
        is_leaf: false,
    }
}
