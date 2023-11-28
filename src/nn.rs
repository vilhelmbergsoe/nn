use core::fmt;
use ndarray::{Array, Array1, Array2, ArrayD, ShapeBuilder};
use rand::random;
use std::ops::{Add, Mul};

pub fn randn(shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_simple_fn(shape.into_shape(), random),
        requires_grad,
    )
}

#[derive(Debug)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub requires_grad: bool,
    pub gradient: Option<ArrayD<f32>>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Self {
            data: data.clone(),
            requires_grad,
            gradient: if requires_grad {
                Some(Array::zeros(data.view().raw_dim()))
            } else {
                None
            },
        }
    }

    pub fn backward(&mut self) {
        if let Some(ref grad) = self.gradient {
            // Do something with the gradients, e.g., update parameters
            // Example: self.data -= learning_rate * grad;
        }
    }

    // pub fn matmul(&self, other: &Self) -> Self {
    //     Tensor::new(self.data.view().dot(&other.data.view()).into_dyn(), self.requires_grad || other.requires_grad)
    // }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        let data = &self.data + &other.data;
        let result = Tensor::new(data, requires_grad);

        // if requires_grad {
        //     if let Some(ref mut grad) = result.gradient {
        //         grad.assign(&1.0);
        //     }
        // }

        result
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        let data = &self.data * &other.data;
        let result = Tensor::new(data, requires_grad);

        // if requires_grad {
        //     if let Some(ref mut grad) = result.gradient {
        //         grad.assign(&1.0);
        //     }
        // }

        result
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
            w: Tensor::new(
                Array2::from_shape_simple_fn((p, c), random).into_dyn(),
                false,
            ),
            b: Tensor::new(Array1::from_shape_simple_fn(c, random).into_dyn(), false),
        }
    }
    pub fn calc(&self, input: &Tensor) -> Tensor {
        // let input_shape = input.data.shape();
        // let desired_shape = (input.data.shape()[0], self.w.data.shape()[1]);

        // let reshaped_input = input
        //     .data
        //     .view()
        //     .into_shape(input.data.shape()[0])
        //     .unwrap_or_else(|_| {
        //         panic!(
        //             "Error reshaping input tensor from shape {:?} to 1D array (expected shape: ({}))",
        //             input_shape,
        //             input_shape[0],
        //         )
        //     });

        // let reshaped_weights = self
        //     .w
        //     .data
        //     .view()
        //     .into_shape((input.data.shape()[0], self.w.data.shape()[1]))
        //     .unwrap_or_else(|_| {
        //         panic!(
        //             "Error reshaping weight tensor from shape {:?} to 2D array (expected shape: {:?})",
        //             self.w.data.shape(),
        //             desired_shape,
        //         )
        //     });
        // let reshaped_input = Array1::from_shape_vec((input.data.shape()[0]), input.data.clone().into_raw_vec()).unwrap();
        // let reshaped_weights = self.w.data.clone().into_shape((self.w.data.len(), 1)).unwrap();
        // let reshaped_weights = Array2::from_shape_vec((self.w.data.view().shape()[0], self.w.data.view().shape()[1]), self.w.data.into_raw_vec()).unwrap();

        // return input.matmul(&self.w);

        // println!("{:?}", self.w.data.shape());

        // println!("input: {}", input);
        // the index doesn't take the right axis all the time so-
        // it needs to somehow shape it to the axis with the most inputs but idk
        // how you're supposed to do that if you want to be able to have a input
        // of [1]
        //                                  the problem with this is the idx here
        // let reshaped_input = input.data.view().into_shape(input.data.shape()[0]).unwrap();

        // println!("reshaped input: {}", reshaped_input);
        // println!();
        Tensor::new(
            &input.data * &self.w.data + &self.b.data,
            input.requires_grad,
        )

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
    Tensor::new(
        arr.data.mapv(|x| if x > 0.0 { x } else { 0.0 }),
        arr.requires_grad,
    )
}

pub fn sigmoid(arr: Tensor) -> Tensor {
    Tensor::new(
        arr.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        arr.requires_grad,
    )
}

pub fn tanh(arr: Tensor) -> Tensor {
    Tensor::new(arr.data.mapv(|x| x.tanh()), arr.requires_grad)
}
