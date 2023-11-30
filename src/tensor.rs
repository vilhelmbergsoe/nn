use ndarray::{arr0, arr1, arr2, ArrayD};
use num_traits::Float;
use std::ops::{Add, Mul, Sub};
use std::fmt;

// #[derive(Debug, Clone, Copy)]
// enum Operator {
//     Add,
//     Mul,
//     Sub,
// }

// #[derive(Debug, Clone)]
// enum Operation {
//     Add,
//     Mul,
//     Sub,
//     Div,
//     Relu,
//     Sigmoid,
//     Tanh,
//     // TODO add custom function functionality
//     // Custom(Box<dyn Fn(T) -> T>),
// }

trait Backward<T: Float>: fmt::Debug + Clone {
    fn backward(&self, saved_tensors: &[Tensor<T>], grad: &ArrayD<T>);
}

#[derive(Debug, Clone)]
enum BackwardFn {
    Mul(MulBackward),
    Relu(ReluBackward),
    // Add(AddBackward),
    // Add other operations as needed
}

struct Node<T: Float> {
    saved_tensors: Vec<Tensor<T>>,
    backward_fn: BackwardFn,
}

impl<T: Float> Clone for Node<T> {
    fn clone(&self) -> Self {
        // Implement Clone for Node
        // You need to clone fields like self.saved_tensors and self.backward here
        Node {
            saved_tensors: self.saved_tensors.clone(),
            backward_fn: self.backward_fn.clone(),
        }
    }
}

impl<T: Float + fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Implement Debug for Node
        // You can access fields like self.saved_tensors and self.backward here
        write!(f, "Node {{ saved_tensors: {:?}, backward: {:?} }}", self.saved_tensors, self.backward_fn)
    }
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

    /// Create a scalar tensor from a value
    pub fn from_scalar(value: T) -> Self {
        Self {
            data: arr0(value).into_dyn(),
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    /// Create a 1D tensor from a slice
    pub fn from_slice(values: &[T]) -> Self {
        Self {
            data: arr1(values).into_dyn(),
            grad_fn: None,
            requires_grad: false,
            grad: None,
            is_leaf: true,
        }
    }

    /// Initialize with the requires_grad flag set
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Do backward pass and compute gradients for tree
    fn backward(&mut self) {
        if let Some(ref grad_fn) = self.grad_fn {
            // if let Some(ref backward) = grad_fn.backward_fn {
            //     let grad = ArrayD::from_elem(self.data.shape(), T::from(1.0).unwrap());
            //     let _ = backward.backward(&grad_fn.saved_tensors, &grad);
            // }
        }
    }

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

#[derive(Debug, Clone)]
struct MulBackward;

impl<T: Float + std::fmt::Debug> Backward<T> for MulBackward {
    fn backward(&self, tensors: &[Tensor<T>], grad: &ArrayD<T>) {
        // Implementation for MulBackward
        // Mutate the grad values in tensors
        // ...
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
                saved_tensors: vec![self.clone(), other.clone()],
                backward_fn: BackwardFn::Mul(MulBackward),
                // op: Some(Operation::Mul),
            }));
            result.is_leaf = false;
        }

        result
    }
}

// impl<T: Float> Add for Tensor<T> {
//     type Output = Tensor<T>;

//     fn add(self, other: Tensor<T>) -> Tensor<T> {
//         let result_data = &self.data + &other.data;
//         let mut result = Tensor::new(result_data);

//         // Propagate requires_grad if any of the input tensors requires_grad
//         result.requires_grad = self.requires_grad || other.requires_grad;

//         // Initialize grad with None
//         result.grad = None;

//         if result.requires_grad {
//             // Set grad_fn for result tensor
//             result.grad_fn = Some(Box::new(Node {
//                 saved_tensors: vec![self.clone(), other.clone()],
//                 op: Some(Operation::Add),
//             }));
//         }

//         result
//     }
// }

// // Implement Mul and Add for Tensor and &Tensor
// macro_rules! impl_mul_add {
//     ($type:ty) => {
//         impl Mul<$type> for $type {
//             type Output = Tensor;

//             fn mul(self, other: $type) -> Tensor {
//                 let mut result = Tensor::new(self.data * other.data);

//                 // Set grad_fn for result tensor
//                 result.grad_fn = Some(Box::new(Node {
//                     saved_tensors: vec![Box::new(self.clone()), Box::new(other.clone())],
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
//                     saved_tensors: vec![self, other],
//                     operator: Some(Operator::Add),
//                 }));

//                 result
//             }
//         }
//     };
// }

// impl_mul_add!(&Tensor);
// impl_mul_add!(Tensor);

#[derive(Debug, Clone)]
struct ReluBackward;

impl<T: Float + std::fmt::Debug> Backward<T> for ReluBackward {
    fn backward(&self, tensors: &[Tensor<T>], grad: &ArrayD<T>) {
    }
}

pub fn relu<T: Float>(x: Tensor<T>) -> Tensor<T> {
    let data = x.data.mapv(|val| if val < T::from(0.0).unwrap() { T::from(0.0).unwrap() } else { val });
    Tensor {
        data,
        grad_fn: Some(Box::new(Node {
            saved_tensors: vec![x.clone()],
            backward_fn: BackwardFn::Relu(ReluBackward),
        })),
        requires_grad: x.requires_grad,
        grad: None,
        is_leaf: false,
    }
}

// pub fn relu<T: Float>(x: Tensor<T>) -> Tensor<T> {
//     if x.data < 0.0 {
//         Tensor {
//             data: x.data.mapv(|val| if val < 0.0 { T::from(0.0) } else { Some(val) }),
//             grad_fn: Some(Box::new(Node {
//                 saved_tensors: vec![x],
//                 op: Some(Operation::Relu),
//             })),
//             requires_grad: x.requires_grad,
//             grad: None,
//             is_leaf: false,
//         }
//     } else {
//         x
//     }
// }
