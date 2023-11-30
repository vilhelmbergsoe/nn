use crate::tensor::Tensor;
use ndarray::ArrayD;
use num_traits::Float;
use std::fmt;

pub trait Backward<T: Float>: fmt::Debug + Clone {
    fn backward(&self, saved_tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>);
}

#[derive(Debug, Clone)]
pub enum BackwardFn {
    Mul(MulBackward),
    Add(AddBackward),
    Relu(ReluBackward),
    // Add other operations as needed
}

#[derive(Debug, Clone)]
pub struct MulBackward;

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
            // if tensor isn't a leaf node propagate through to the next gradient compute
            } else {
                tensors[0].backward_grad(&grad);
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
                tensors[1].backward_grad(&grad);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AddBackward;

impl<T: Float + std::fmt::Debug> Backward<T> for AddBackward {
    fn backward(&self, tensors: &mut Vec<Box<Tensor<T>>>, grad: &ArrayD<T>) {
        // Set the gradients for the input tensors
        if tensors[0].requires_grad {
            // TODO: FIX THE CALCULATION OF THIS GRADIENT
            let grad_input0 = ArrayD::ones(tensors[0].data.shape());

            if tensors[0].is_leaf {
                if let Some(ref mut input0_grad) = tensors[0].grad {
                    *input0_grad = grad_input0;
                } else {
                    tensors[0].grad = Some(grad_input0);
                }
            // Propagate back and call the next backward function
            } else {
                tensors[0].backward_grad(&grad);
            }
        }

        if tensors[1].requires_grad {
            // TODO: FIX THE CALCULATION OF THIS GRADIENT
            let grad_input1 = ArrayD::ones(tensors[1].data.shape());

            if tensors[1].is_leaf {
                if let Some(ref mut input1_grad) = tensors[1].grad {
                    *input1_grad = grad_input1
                } else {
                    tensors[1].grad = Some(grad_input1);
                }
            } else {
                tensors[1].backward_grad(&grad);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReluBackward;

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
