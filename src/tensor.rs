use ndarray::{arr0, arr1, arr2, ArrayD};
use num_traits::Float;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul, Sub};

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
    fn backward(&self, saved_tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>);
}

#[derive(Debug, Clone)]
enum BackwardFn {
    Mul(MulBackward),
    Add(AddBackward),
    Relu(ReluBackward),
    // Add other operations as needed
}

struct Node<T: Float> {
    saved_tensors: Vec<Box<Tensor<T>>>,
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
        write!(
            f,
            "Node {{
    saved_tensors: {:#?},
    backward_fn: {:#?}
}}",
            self.saved_tensors, self.backward_fn
        )
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

impl<T: Float + fmt::Debug> Tensor<T> {
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
    pub fn backward(&mut self) {
        if let Some(ref mut grad_fn) = self.grad_fn {
            let grad = ArrayD::from_elem(self.data.shape(), T::from(1.0).unwrap());
            match &grad_fn.backward_fn {
                BackwardFn::Mul(mul_backward) => {
                    let mut tensors = grad_fn.saved_tensors.borrow_mut();
                    mul_backward.backward(&mut tensors, &grad);
                }
                BackwardFn::Relu(relu_backward) => {
                    relu_backward.backward(&mut grad_fn.saved_tensors.borrow_mut(), &grad)
                }
                _ => todo!(),
            };
            // let _ = *grad_fn.backward_fn.backward(&grad_fn.saved_tensors, &grad);
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
    fn backward(&self, tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>) {
        // Set the gradients for the input tensors
        if tensors[0].requires_grad {
            let grad_input0 = grad * &tensors[1].data;

            if tensors[0].is_leaf {
                if let Some(ref mut input0_grad) = tensors[0].grad {
                    *input0_grad = grad_input0;
                } else {
                    tensors[0].grad = Some(grad_input0);
                }
            }
        }

        if tensors[1].requires_grad {
            let grad_input1 = grad * &tensors[0].data;

            if tensors[1].is_leaf {
                if let Some(ref mut input1_grad) = tensors[1].grad {
                    *input1_grad = grad_input1
                } else {
                    tensors[1].grad = Some(grad_input1);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AddBackward;

impl<T: Float + std::fmt::Debug> Backward<T> for AddBackward {
    fn backward(&self, tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>) {

        // Set the gradients for the input tensors
        if tensors[0].requires_grad {
            let grad_input0 = grad * &tensors[1].data;

            if tensors[0].is_leaf {
                if let Some(ref mut input0_grad) = tensors[0].grad {
                    *input0_grad = grad_input0;
                } else {
                    tensors[0].grad = Some(grad_input0);
                }
            // Propagate back and call the next backward function
            } else {

            }
        }

        if tensors[1].requires_grad {
            let grad_input1 = grad * &tensors[0].data;

            if tensors[1].is_leaf {
                if let Some(ref mut input1_grad) = tensors[1].grad {
                    *input1_grad = grad_input1
                } else {
                    tensors[1].grad = Some(grad_input1);
                }
            } else {

            }
        }
    }
}

impl<T: Float + fmt::Debug> Mul for Tensor<T> {
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
                saved_tensors: vec![Box::new(self), Box::new(other)],
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

#[derive(Debug, Clone)]
struct ReluBackward;

impl<T: Float + std::fmt::Debug> Backward<T> for ReluBackward {
    fn backward(&self, tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>) {
        if tensors[0].requires_grad {
            // let mut input_grad = tensors[0].grad.borrow_mut();
            let grad_mask = tensors[0].data.mapv(|val| {
                if val <= T::from(0.0).unwrap() {
                    T::from(0.0).unwrap()
                } else {
                    T::from(1.0).unwrap()
                }
            });

            if tensors[0].is_leaf {
                if let Some(ref mut input0_grad) = tensors[0].grad {
                    *input0_grad = grad * &grad_mask;
                } else {
                    tensors[0].grad = Some(grad * &grad_mask);
                }
            }
        }
    }
}

// Relu'(0.0) = 0.0 Not sure if this is right
pub fn relu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
    let data = x.data.mapv(|val| {
        if val < T::from(0.0).unwrap() {
            T::from(0.0).unwrap()
        } else {
            val
        }
    });
    Tensor {
        data,
        grad_fn: Some(Box::new(Node {
            saved_tensors: vec![Box::new(x.clone())],
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
