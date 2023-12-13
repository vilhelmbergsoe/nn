use crate::shape::Shape;

// TODO: read up on strides and start_offset
#[derive(Debug, Clone)]
pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
    start_offset: usize,
}
