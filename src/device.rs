use crate::shape::Shape;
use crate::dtype::WithDType;

#[derive(Debug, Clone)]
pub enum DeviceLocation {
    Cpu,
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
}

pub trait NdArray {
    // TODO: Option / Result?
    fn shape(&self) -> Option<Shape>;
}

impl<T: WithDType> NdArray for T {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(()))
    }
}

impl<T: WithDType, const N: usize> NdArray for &[T; N] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(self.len()))
    }
}

impl<T: WithDType> NdArray for &[T] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from(self.len()))
    }
}

impl<T: WithDType, const N: usize, const M: usize> NdArray for &[[T; N]; M] {
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from((M, N)))
    }
}

impl<T: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[T; N3]; N2]; N1]
{
    fn shape(&self) -> Option<Shape> {
        Some(Shape::from((N1, N2, N3)))
    }
}
