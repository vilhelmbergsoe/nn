use ndarray::{arr0, arr1, arr2, ArrayD};
use ndarray::{concatenate, Axis, NdFloat};

mod tensor;
use rand::distributions::{Distribution, Standard};
use tensor::relu;
use tensor::{Tensor, TensorRef};

mod nn;
use nn::nn::{Linear, NN};

struct XORNet<T: NdFloat> {
    fl1: Linear<T>,
    fl2: Linear<T>,
    // bl1: Linear<T>,
    // bl2: Linear<T>,

    // hl1: Linear<T>,
}

impl<T: NdFloat> NN<T> for XORNet<T>
where
    Standard: Distribution<T>,
{
    fn new() -> Self {
        Self {
            fl1: Linear::new(2, 2),
            fl2: Linear::new(2, 1),
            // bl1: Linear::new(2, 1),
            // bl2: Linear::new(1, 2),

            // hl1: Linear::new(4, 2),
        }
    }

    fn forward(&self, inputs: &TensorRef<T>) -> TensorRef<T> {
        let mut outputs: Vec<TensorRef<T>> = Vec::new();
        for input in inputs.borrow().data.iter() {
            let flx = relu(&self.fl1.forward(&tensor!(input.clone())));
            let flx = relu(&self.fl2.forward(&flx));
            outputs.push(flx);
        }

        // let blx = self.bl1.forward(&input.clone());

        // let flx = self.fl2.forward(&flx);
        // let blx = self.bl2.forward(&blx);

        // TODO: add concatenate operation on tensorref right now this doesn't
        // create a backward_fn
        // let merged_arr = Tensor::new(
        //     concatenate(
        //         Axis(0),
        //         &[flx.borrow().data.view(), blx.borrow().data.view()],
        //     )
        //     .unwrap(),
        // )
        // .as_ref();

        // let x = self.hl1.forward(&merged_arr);

        // x
        tensor!(&outputs).as_ref()
    }
}

fn main() {
    // let x = tensor!(2.0, requires_grad);
    // let y = tensor!(3.0, requires_grad);
    // let z = &x * &y;
    // let mut g = relu(&z);

    // let f1 = Linear::<f32>::new(1, 2);
    // let f2 = Linear::<f32>::new(2, 1);

    // let x = f1.forward(tensor!(&[1.0, 2.0]));

    // println!("{}", x);
    let inputs = &[[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let targets = &[0., 1., 1., 0.];

    let nn = XORNet::<f32>::new();
    for i in 0..100_000 {
        let outputs: Vec<TensorRef<f32>> = Vec::new();
        for input in inputs {
            let output = nn.forward(&tensor!(input));
            outputs.push(output);
        }
        let mut loss = nn::nn::mse_loss(output, target);
    }

    // z.backward();

    // println!("{:?}", nn.fl2.w._ref.borrow().grad);

    // let g = tensor!(&[[1.], [2.]]);

    // println!("{}", &g*&x);

    // y.backward();

    // println!("{}", y);

    // g.backward();

    // // println!("{:?}", g);
    // println!("{:?}", x.grad());
    // println!("{:?}", y.grad());
}
