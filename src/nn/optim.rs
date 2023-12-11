use ndarray::{ArrayD, NdFloat};

use crate::tensor::{TensorRef, Tensor};

pub trait Optimizer<T: NdFloat> {
    fn new(params: Vec<TensorRef<T>>, learning_rate: f32) -> Self;
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD<T: NdFloat> {
    learning_rate: f32,
    params: Vec<TensorRef<T>>
}

impl<T: NdFloat> Optimizer<T> for SGD<T> {
    fn new(params: Vec<TensorRef<T>>, learning_rate: f32) -> Self {
        Self {
            params,
            learning_rate,
        }
    }

    fn step(&mut self) {
        for param in self.params.iter_mut() {
            let mut param = param.borrow_mut();
            param.data = param.data.clone() - param.grad.clone().unwrap() * T::from(self.learning_rate).unwrap();
        }
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter_mut() {
            let mut param = param.borrow_mut();

            param.grad = None;
        }
    }
}
