use ndarray::{ArrayD, NdFloat};
use std::fmt;

use super::tensor::TensorRef;

pub trait BinaryBackward<T: NdFloat>: fmt::Debug + Clone {
    fn backward(&self, tensors: (TensorRef<T>, TensorRef<T>), grad: &ArrayD<T>);
}

pub trait UnaryBackward<T: NdFloat>: fmt::Debug + Clone {
    fn backward(&self, tensor: TensorRef<T>, grad: &ArrayD<T>);
}

#[derive(Debug, Clone)]
pub enum BinaryBackwardFn {
    Mul(MulBackward),
    Add(AddBackward),
    Sub(SubBackward),
}

#[derive(Debug, Clone)]
pub enum UnaryBackwardFn<T: NdFloat> {
    Pow(PowBackward<T>),
    Relu(ReluBackward),
    Mean(MeanBackward),
}

// TODO: make the backward functions more standard and nicer

#[derive(Debug, Clone)]
pub struct AddBackward;

impl<T: NdFloat + std::fmt::Debug> BinaryBackward<T> for AddBackward {
    fn backward(&self, tensors: (TensorRef<T>, TensorRef<T>), grad: &ArrayD<T>) {
        for mut tensor in [tensors.0.clone(), tensors.1.clone()] {
            let mut tensor = tensor.borrow_mut();

            if tensor.requires_grad {
                let grad_input = grad.clone();

                if tensor.is_leaf {
                    if let Some(ref mut input_grad) = tensor.grad {
                        *input_grad = grad_input;
                    } else {
                        tensor.grad = Some(grad_input);
                    }
                } else {
                    tensor.backward_grad(&grad_input);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubBackward;

impl<T: NdFloat + std::fmt::Debug> BinaryBackward<T> for SubBackward {
    fn backward(&self, mut tensors: (TensorRef<T>, TensorRef<T>), grad: &ArrayD<T>) {
        let mut tensor0 = tensors.0.borrow_mut();
        let mut tensor1 = tensors.1.borrow_mut();

        if tensor0.requires_grad {
            let grad_input = grad.clone();

            if tensor0.is_leaf {
                if let Some(ref mut input_grad) = tensor0.grad {
                    *input_grad = grad_input;
                } else {
                    tensor0.grad = Some(grad_input);
                }
            } else {
                tensor0.backward_grad(&grad_input);
            }
        }

        if tensor1.requires_grad {
            let grad_input = -grad.clone();

            if tensor1.is_leaf {
                if let Some(ref mut input_grad) = tensor1.grad {
                    *input_grad = grad_input;
                } else {
                    tensor1.grad = Some(grad_input);
                }
            } else {
                tensor1.backward_grad(&grad_input);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MulBackward;

impl<T: NdFloat + std::fmt::Debug> BinaryBackward<T> for MulBackward {
    fn backward(&self, mut tensors: (TensorRef<T>, TensorRef<T>), grad: &ArrayD<T>) {
        let mut tensor0 = tensors.0.borrow_mut();
        let mut tensor1 = tensors.1.borrow_mut();

        if tensor0.requires_grad {
            let grad_input0 = grad * &tensor1.data;

            if tensor0.is_leaf {
                if let Some(ref mut input0_grad) = tensor0.grad {
                    *input0_grad = grad_input0;
                } else {
                    tensor0.grad = Some(grad_input0);
                }
            } else {
                tensor0.backward_grad(&grad_input0);
            }
        }

        if tensor1.requires_grad {
            let grad_input1 = grad * &tensor0.data;

            if tensor1.is_leaf {
                if let Some(ref mut input1_grad) = tensor1.grad {
                    *input1_grad = grad_input1;
                } else {
                    tensor1.grad = Some(grad_input1);
                }
            } else {
                tensor1.backward_grad(&grad_input1);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PowBackward<T: NdFloat>(pub T);

impl<T: NdFloat + std::fmt::Debug> UnaryBackward<T> for PowBackward<T> {
    fn backward(&self, mut tensor: TensorRef<T>, grad: &ArrayD<T>) {
        let mut tensor = tensor.borrow_mut();
        let exponent = self.0;

        // Set the gradients for the input tensors
        if tensor.requires_grad {
            let computed_grad = grad
                * &(tensor
                    .data
                    .mapv(|val| exponent * val.powf(exponent - T::one())));

            // If tensor is a leaf node set gradient field
            if tensor.is_leaf {
                if let Some(ref mut input_grad) = tensor.grad {
                    // Compute gradients for element-wise power operation
                    *input_grad = computed_grad;
                } else {
                    tensor.grad = Some(computed_grad);
                }
            // If tensor isn't a leaf node, propagate gradient to next grad_fn
            } else {
                tensor.backward_grad(&computed_grad);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeanBackward;

impl<T: NdFloat + std::fmt::Debug> UnaryBackward<T> for MeanBackward {
    fn backward(&self, mut tensor: TensorRef<T>, grad: &ArrayD<T>) {
        let mut tensor = tensor.borrow_mut();

        if tensor.requires_grad {
            let computed_grad = grad * T::from(1.0).unwrap() / T::from(tensor.data.len()).unwrap();

            if tensor.is_leaf {
                if let Some(ref mut input0_grad) = tensor.grad {
                    *input0_grad = computed_grad;
                } else {
                    tensor.grad = Some(computed_grad);
                }
            } else {
                tensor.backward_grad(&computed_grad);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReluBackward;

impl<T: NdFloat + std::fmt::Debug> UnaryBackward<T> for ReluBackward {
    fn backward(&self, mut tensor: TensorRef<T>, grad: &ArrayD<T>) {
        let mut tensor = tensor.borrow_mut();

        if tensor.requires_grad {
            let grad_mask = tensor.data.mapv(|val| {
                if val <= T::from(0.0).unwrap() {
                    T::from(0.0).unwrap()
                } else {
                    T::from(1.0).unwrap()
                }
            });

            if tensor.is_leaf {
                if let Some(ref mut input0_grad) = tensor.grad {
                    *input0_grad = grad * &grad_mask;
                } else {
                    tensor.grad = Some(grad * &grad_mask);
                }
            } else {
                tensor.backward_grad(&(grad * &grad_mask));
            }
        }
    }
}
