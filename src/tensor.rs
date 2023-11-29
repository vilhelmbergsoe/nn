use ndarray::{arr0, arr1, arr2, ArrayD};
use num_traits::Float;
use std::ops::{Add, Mul, Sub};
use std::fmt;

#[derive(Debug, Clone, Copy)]
enum Operator {
    Add,
    Mul,
    Sub,
}

#[derive(Debug, Clone)]
struct Node<T: Float> {
    input_tensors: Vec<Tensor<T>>,
    operator: Option<Operator>,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Float> {
    pub data: ArrayD<T>,
    grad_fn: Option<Box<Node<T>>>,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<T>>,
    pub is_leaf: bool,
}

impl<T: Float> Tensor<T> {
    pub fn new(data: ArrayD<T>) -> Self {
        Self {
            data,
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    // Create a scalar tensor
    pub fn from_scalar(value: T) -> Self {
        Self {
            data: arr0(value).into_dyn(),
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    // Create a 1D tensor from a slice
    pub fn from_slice(values: &[T]) -> Self {
        Self {
            data: arr1(values).into_dyn(),
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    // // Backward pass for the current tensor
    // fn backward(&self) {
    //     if self.requires_grad {
    //         if let Some(ref grad_fn) = self.grad_fn {
    //             match grad_fn.operator {
    //                 Some(Operator::Mul) => self.mul_backward(1.0),
    //                 // Handle other operators if needed
    //                 _ => unimplemented!("Backward pass not implemented for this operator"),
    //             }
    //         }
    //     }
    // }

    // // Backward pass for multiplication
    // fn mul_backward(&self, grad: T) {
    //     if self.requires_grad {
    //         // Only calculate gradient for tensors marked as leaf nodes
    //         if let Some(ref grad_fn) = self.grad_fn {
    //             match grad_fn.operator {
    //                 Some(Operator::Mul) => {
    //                     // Assuming only two input tensors for simplicity
    //                     let input1 = &grad_fn.input_tensors[0];
    //                     let input2 = &grad_fn.input_tensors[1];

    //                     // Compute the gradients
    //                     let grad_input1 = grad * &input2.data;
    //                     let grad_input2 = grad * &input1.data;

    //                     // Set the gradients for the input tensors
    //                     if let Some(ref mut grad) = input1.grad {
    //                         *grad += grad_input1;
    //                     } else {
    //                         input1.grad = Some(grad_input1);
    //                     }

    //                     if let Some(ref mut grad) = input2.grad {
    //                         *grad += grad_input2;
    //                     } else {
    //                         input2.grad = Some(grad_input2);
    //                     }

    //                     // Continue the backward pass for input tensors
    //                     input1.backward(&grad_input1);
    //                     input2.backward(&grad_input2);
    //                 }
    //                 _ => {
    //                     // Handle other operators if needed
    //                     unimplemented!("Backward pass not implemented for this operator");
    //                 }
    //             }
    //         }
    //     }
    // }

    // pub fn from_2d_array(&self, values: &[Vec<T>]) -> Self {
    //     Self {
    //         data: arr2(values).into_dyn(),
    //         grad_fn: None,
    //     }
    // }
}

impl<T: Float + std::fmt::Debug> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl<T: Float> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        let result_data = &self.data * &other.data;
        let mut result = Tensor::new(result_data);

        // Propagate requires_grad if any of the input tensors requires_grad
        result.requires_grad = self.requires_grad || other.requires_grad;

        // Initialize grad with None
        result.grad = None;

        if result.requires_grad {
            // Set grad_fn for result tensor
            result.grad_fn = Some(Box::new(Node {
                input_tensors: vec![self.clone(), other.clone()],
                operator: Some(Operator::Mul),
            }));
            result.is_leaf = false;
        }

        result
    }
}

impl<T: Float> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        let result_data = &self.data + &other.data;
        let mut result = Tensor::new(result_data);

        // Propagate requires_grad if any of the input tensors requires_grad
        result.requires_grad = self.requires_grad || other.requires_grad;

        // Initialize grad with None
        result.grad = None;

        if result.requires_grad {
            // Set grad_fn for result tensor
            result.grad_fn = Some(Box::new(Node {
                input_tensors: vec![self.clone(), other.clone()],
                operator: Some(Operator::Add),
            }));
        }

        result
    }
}

// // Implement Mul and Add for Tensor and &Tensor
// macro_rules! impl_mul_add {
//     ($type:ty) => {
//         impl Mul<$type> for $type {
//             type Output = Tensor;

//             fn mul(self, other: $type) -> Tensor {
//                 let mut result = Tensor::new(self.data * other.data);

//                 // Set grad_fn for result tensor
//                 result.grad_fn = Some(Box::new(Node {
//                     input_tensors: vec![Box::new(self.clone()), Box::new(other.clone())],
//                     operator: Some(Operator::Mul),
//                 }));

//                 result
//             }
//         }

//         impl Add<$type> for $type {
//             type Output = Tensor;

//             fn add(self, other: $type) -> Tensor {
//                 let result_data = self.data.clone() + other.data.clone();
//                 let mut result = Tensor::new(result_data);

//                 // Set grad_fn for result tensor
//                 result.grad_fn = Some(Box::new(Node {
//                     input_tensors: vec![self, other],
//                     operator: Some(Operator::Add),
//                 }));

//                 result
//             }
//         }
//     };
// }

// impl_mul_add!(&Tensor);
// impl_mul_add!(Tensor);
