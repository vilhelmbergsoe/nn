#![allow(warnings)]

use ndarray::{arr0, arr1, arr2, ArrayD};

mod tensor;
use tensor::Tensor;

fn main() {
    for i in 0..100000 {
        let x = Tensor::from_scalar(10.0).with_grad();
        // let y = Tensor::from_scalar(10.0);
        // let y = Tensor::new(arr1(&[2.0, 0.5]).into_dyn());
        let mut z = tensor::relu(&x);

        // let mut g =  z * Tensor::from_scalar(5.0).with_grad();

        z.backward();
    }

    // println!("{:?}", z)

    // println!("{:?}", g);

    // println!("{:?}", z);

    // let z = x * y.clone();

    // println!("{z}");

    // let g = z + y;

    // println!("{g}");
    // println!("{:#?}", g);
}
