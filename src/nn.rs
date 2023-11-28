use ndarray::{Array1, Array2, ArrayD};
use rand::random;
use core::fmt;

#[derive(Debug)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Self {
            data,
            requires_grad,
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

pub struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    pub fn new(p: usize, c: usize) -> Self {
        Self {
            w: Tensor::new(Array2::from_shape_simple_fn((p, c), random).into_dyn(), false),
            b: Tensor::new(Array1::from_shape_simple_fn(c, random).into_dyn(), false),
        }
    }
    pub fn calc(&self, input: &Tensor) -> Tensor {
        // input.data.dot(&self.w.data) + &self.b.data

        // Ensure that the dimensions are compatible
        // assert_eq!(self.w.data.shape()[1], input.data.shape()[0]);

        let reshaped_input = input.data.view().into_shape(input.data.len()).unwrap();
        // let reshaped_weights = self.w.data.clone().into_shape((self.w.data.len(), 1)).unwrap();
        let reshaped_weights = self.w.data.view().into_shape((input.data.len(), self.w.data.shape()[1])).unwrap();

        Tensor::new(reshaped_input.dot(&reshaped_weights) + &self.b.data, input.requires_grad)

        // Tensor::new(input.data.dot(&self.w.data) + &self.b.data, input.requires_grad)
    }
}

pub trait NN {
    fn new() -> Self;

    fn forward(&self, x: Tensor) -> Tensor;

    // fn backward(&self, target: &Array1<f32>) -> Array1<f32> {
    // // Calculate the error for each output
    // let error = target - self.forward(target);

    // // Calculate the gradients of the loss function with respect to each parameter
    // let mut gradients = Array1::zeros(self.hl1.w.shape());
    // let mut last_gradient = error;

    // for axis in Axis::new(0) {
    //     let current_gradient = last_gradient.clone();
    //     last_gradient = concatenate(axis, &[current_gradient.view(), current_gradient.view()]).unwrap();
    //     gradients[axis] = last_gradient;
    // }

    // // Update the parameters using the calculated gradients and an optimization algorithm (e.g., gradient descent)
    // // ...
    // }
}

// trait Optimizer {
//     fn optimize(&mut self, net: &mut dyn Net, target: &Array1<f32>);
// }

// struct SGD {
//     learning_rate: f32,
// }

// impl SGD {
//     pub fn new(learning_rate: f32) -> Self {
//         Self { learning_rate }
//     }
// }

// impl Optimizer for SGD {
//     fn optimize(&mut self, net: &mut NN, target: &Array1<f32>) {
//         net.backward(target, self.learning_rate);
//     }
// }

pub fn mean_squared_error(y: &Array1<f32>, y_pred: &Array1<f32>) -> f32 {
    (y - y_pred).mapv(|x| x.powi(2)).sum() / y.len() as f32
}

pub fn relu(arr: Tensor) -> Tensor {
    Tensor::new(arr.data.mapv(|x| if x > 0.0 { x } else { 0.0 }), arr.requires_grad)
}

pub fn sigmoid(arr: Tensor) -> Tensor {
    Tensor::new(arr.data.mapv(|x| 1.0 / (1.0 + (-x).exp())), arr.requires_grad)
}

pub fn tanh(arr: Tensor) -> Tensor {
    Tensor::new(arr.data.mapv(|x| x.tanh()), arr.requires_grad)
}
