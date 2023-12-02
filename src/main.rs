use ndarray::{arr0, arr1, arr2, ArrayD};

mod tensor;
use tensor::Tensor;
use tensor::relu;

fn main() {
    let x = tensor!(2.0, requires_grad);
    let y = tensor!(3.0, requires_grad);
    let z = &x * &y;
    let mut g = relu(&z);

    g.backward();

    println!("{:#?}", g);
    println!("{:#?}", x.grad());
    println!("{:#?}", y.grad());
}
