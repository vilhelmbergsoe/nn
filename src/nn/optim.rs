use ndarray::{ArrayD, NdFloat};

use crate::tensor::TensorRef;

pub trait Optimizer<T: NdFloat> {
    fn new(params: Vec<TensorRef<T>>, learning_rate: f32) -> Self;
    fn step(&mut self, parameters: &mut Vec<TensorRef<T>>, gradients: Vec<ArrayD<T>>);
}

pub struct SGD<T: NdFloat> {
    learning_rate: f32,
    params: Vec<TensorRef<T>>
}

impl<T: NdFloat> Optimizer<T> for SGD<T> {
    fn new(params: Vec<TensorRef<T>>, learning_rate: f32) {
        Self {
            learning_rate,
            params
        }
    }

    fn step(&mut self, gradients: Vec<TensorRef<T>>) {
        for (param, grad) in parameters.iter_mut().zip(gradients) {
            *param.borrow_mut() = param - &(grad * Tensor::from(self.learning_rate));
        }
    }
}
