#![allow(warnings)]

use ndarray::{arr0, arr1, arr2, ArrayD};

mod tensor;
use tensor::Tensor;

fn main() {
    // for i in 0..100000 {
    // let x = Tensor::from(&[[2.0, 1.0], [3.0, 1.0]]).with_grad();
    // let y = Tensor::from(&[[-2.0, 1.5], [3.5, 1.5]]).with_grad();
    // let g = Tensor::from(&[[1.0], [2.0]]);
    // let y = Tensor::from_scalar(10.0);
    // let y = Tensor::new(arr1(&[2.0, 0.5]).into_dyn());
    // let z = x + y;

    // let mut g = z * Tensor::from(2.0).with_grad();

    // let mut g =  z * Tensor::from_scalar(5.0).with_grad();

    let mut g = Tensor::from(2.0).with_grad().pow(3.0);

    println!("{:#?}", g.backward());

    // println!("{:#?}", g);
    // }

    // println!("{:?}", z)

    // println!("{:?}", g);

    // println!("{:?}", z);

    // let z = x * y.clone();

    // println!("{z}");

    // let g = z + y;

    // println!("{g}");
    // println!("{:#?}", g);
}
