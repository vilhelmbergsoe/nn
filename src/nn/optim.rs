pub trait Optimizer {
    fn step(&mut self, parameters: &mut [TensorRef<T>], gradients: &[ArrayD<T>]);
}

pub struct SGD {
    learning_rate: f32,
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [TensorRef<T>], gradients: &[TensorRef<T>]) {
        for (param, grad) in parameters.iter_mut().zip(gradients) {
            *param.borrow_mut() -= &(*grad * self.learning_rate);
        }
    }
}