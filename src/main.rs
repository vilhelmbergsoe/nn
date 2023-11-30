#![allow(warnings)]

use ndarray::{arr0, arr1, arr2, ArrayD};

mod tensor;
use tensor::Tensor;

fn main() {
    let x = Tensor::from_scalar(2.0).with_grad();

    // let y = Tensor::from_scalar(10.0).with_grad();
    // let y = Tensor::new(arr1(&[2.0, 0.5]).into_dyn());
    // let z = x*y;

    println!("{:#?}", tensor::relu(x));

    // let z = x * y.clone();

    // println!("{z}");

    // let g = z + y;

    // println!("{g}");
    // println!("{:#?}", g);
}
