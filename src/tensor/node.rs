use crate::tensor::backward::BackwardFn;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt;

pub struct Node<T: Float> {
    pub saved_tensors: Vec<Box<Tensor<T>>>,
    pub backward_fn: BackwardFn,
}

impl<T: Float> Clone for Node<T> {
    fn clone(&self) -> Self {
        // Implement Clone for Node
        // You need to clone fields like self.saved_tensors and self.backward here
        Node {
            saved_tensors: self.saved_tensors.clone(),
            backward_fn: self.backward_fn.clone(),
        }
    }
}

impl<T: Float + fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Implement Debug for Node
        // You can access fields like self.saved_tensors and self.backward here
        write!(
            f,
            "Node {{
    saved_tensors: {:#?},
    backward_fn: {:#?}
}}",
            self.saved_tensors, self.backward_fn
        )
    }
}
