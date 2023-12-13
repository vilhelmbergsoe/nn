use ndarray::{arr0, arr1, arr2, Array, ArrayD, IxDyn};
use ndarray::{concatenate, Axis, NdFloat};

mod tensor;
use rand::distributions::{Distribution, Standard};
use tensor::relu;
use tensor::{Tensor, TensorRef};

mod nn;
use nn::nn::{Linear, Module};
use nn::optim::{Optimizer, SGD};

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
            fl2: Linear::new(2, 2),
        }
    }

    fn forward(&self, input: &TensorRef<T>) -> TensorRef<T> {
        let x = relu(&self.fl1.forward(&input));
        relu(&self.fl2.forward(&x))
    }
}

fn main() {
    // let a = Array::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // let b = Array::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // let result = &a.dot(&b);

    // println!("{}", result);
    let inputs: TensorRef<f32> = tensor!(&[[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    let targets: TensorRef<f32> = tensor!(&[[0.], [1.], [1.], [0.]]);

    let batch_size: usize = inputs.borrow().data.len();

    let nn = XORNet::<f32>::new();
    let mut sgd = SGD::new(vec![nn.fl1.w.clone(), nn.fl2.w.clone()], 0.1);
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

        sgd.step();
    }
}
