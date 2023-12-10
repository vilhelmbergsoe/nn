use ndarray::{arr0, arr1, arr2, ArrayD};
use ndarray::{concatenate, Axis, NdFloat};

mod tensor;
use rand::distributions::{Distribution, Standard};
use tensor::relu;
use tensor::{Tensor, TensorRef};

mod nn;
use nn::nn::{Linear, Module};
use nn::optim::{SGD, Optimizer};

struct XORNet<T: NdFloat> {
    fl1: Linear<T>,
    fl2: Linear<T>,
}

impl<T: NdFloat> XORNet<T>
where
    Standard: Distribution<T>,
{
    fn new() -> Self {
        Self {
            fl1: Linear::new(2, 2),
            fl2: Linear::new(2, 1),
        }
    }

    fn forward(&self, input: &TensorRef<T>) -> TensorRef<T> {
        let x = relu(&self.fl1.forward(&input));
        relu(&self.fl2.forward(&x))
    }

    fn params(&self) -> Vec<TensorRef<T>> {
        // return all parameters
    }
}

fn main() {
    let inputs: TensorRef<f32> = tensor!(&[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    let targets: TensorRef<f32> = tensor!(&[0., 1., 1., 0.]);

    let batch_size: usize = inputs.borrow().data.len();

    let nn = XORNet::<f32>::new();
    let mut sgd = SGD::new(nn.params(), 0.1);
    for e in 0..100_000 {
        let mut outputs: Vec<TensorRef<f32>> = Vec::new();
        for i in 0..batch_size {
            let output = nn.forward(&inputs);
            outputs.push(output);
        }

        let outputs = Tensor::from(outputs).with_grad().as_ref();

        sgd.zero_grad();
        let mut loss = nn::nn::mse_loss(&outputs, &targets);
        loss.backward();

        sgd.step(&loss);
    }
}
