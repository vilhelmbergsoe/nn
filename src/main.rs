use crate::tensor::Tensor;
use crate::device::Device;

fn main() {
    let dev = Device::Cpu;

    // Macro probably still possible with envisioned API / codebase
    let a = Tensor::from(&[1.0, 2.0], &dev, true);
    let b = Tensor::from(&[2.0, 1.0], &dev, true);

    let c = a.matmul(&b);

    c.backward();

    println!("{}", a.grad);
    println!("{}", b.grad);
}
