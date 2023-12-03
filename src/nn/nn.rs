use ndarray::{Array2, Array1, NdFloat};
use rand::{random, distributions::{Distribution, Standard}};

use crate::tensor::{Tensor, TensorRef};

#[macro_use]
use crate::tensor;

pub struct Linear<T: NdFloat> {
    w: TensorRef<T>,
    b: TensorRef<T>,
}

impl<T: NdFloat> Linear<T> where Standard: Distribution<T> {
    pub fn new(p: usize, c: usize) -> Self {
        Self {
            w: Tensor::<T>::new(Array2::from_shape_simple_fn((p, c), random).into_dyn()).with_grad().as_ref(),
            b: Tensor::<T>::new(Array1::from_shape_simple_fn(c, random).into_dyn()).with_grad().as_ref(),
        }
    }

    pub fn forward(&self, input: TensorRef<T>) -> TensorRef<T> {
        let input = input.borrow_mut();
        let weights = self.w.borrow_mut();

        // let reshaped_input = input.data.view_mut().into_shape(input.data.len()).unwrap();
        // let reshaped_weights = weights.data.view_mut().into_shape((input.data.len(), weights.data.shape()[1])).unwrap();

        &(&input * &self.w) + &self.b
    }
}

pub trait NN<T: NdFloat> {
    fn new() -> Self;

    fn forward(&self, input: TensorRef<T>) -> TensorRef<T>;
}
