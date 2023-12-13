use ndarray::{Array1, Array2, NdFloat};
use num_traits::FromPrimitive;
use rand::{
    distributions::{Distribution, Standard},
    random,
};

use crate::tensor::{Tensor, TensorRef};

#[macro_use]
use crate::tensor;

pub struct Linear<T: NdFloat> {
    pub w: TensorRef<T>,
    pub b: TensorRef<T>,
}

impl<T: NdFloat> Linear<T>
where
    Standard: Distribution<T>,
{
    pub fn new(p: usize, c: usize) -> Self {
        Self {
            w: Tensor::<T>::new(Array2::from_shape_simple_fn((p, c), random).into_dyn())
                .with_grad()
                .as_ref(),
            b: Tensor::<T>::new(Array1::from_shape_simple_fn(c, random).into_dyn())
                .with_grad()
                .as_ref(),
        }
    }
}

impl<T: ndarray::NdFloat> Module<T> for Linear<T> {
    fn forward(&self, input: &TensorRef<T>) -> TensorRef<T> {
        // TODO: gradient computation only supporting scalar outputs when this
        // returns a 1d array of only one element
        &input.dot(&self.w) + &self.b
    }

    fn params(&self) -> Vec<TensorRef<T>> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub trait Module<T: NdFloat> {
    fn forward(&self, input: &TensorRef<T>) -> TensorRef<T>;

    fn params(&self) -> Vec<TensorRef<T>>;
}

pub fn mse_loss<T: NdFloat>(output: &TensorRef<T>, target: &TensorRef<T>) -> TensorRef<T>
where
    T: FromPrimitive,
{
    // THIS IS THE PROBLEM NOW
    let diff = output - target;

    let squared_diff = &diff * &diff;
    squared_diff.mean().unwrap()
}
