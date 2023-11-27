use ndarray::{Array1, Array2};
use rand::random;

pub struct Linear {
    w: Array2<f32>,
    b: Array1<f32>,
}

impl Linear {
    pub fn new(p: usize, c: usize) -> Self {
        Self {
            w: Array2::from_shape_simple_fn((p, c), random),
            b: Array1::from_shape_simple_fn(c, random),
        }
    }
    pub fn calc(&self, inputs: &Array1<f32>) -> Array1<f32> {
        inputs.dot(&self.w) + &self.b
    }
}

pub trait NN {
    fn new() -> Self;

    fn forward(&self, x: Array1<f32>) -> Array1<f32>;

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

pub fn relu(arr: Array1<f32>) -> Array1<f32> {
    arr.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

pub fn sigmoid(arr: Array1<f32>) -> Array1<f32> {
    arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

pub fn tanh(arr: Array1<f32>) -> Array1<f32> {
    arr.mapv(|x| x.tanh())
}
