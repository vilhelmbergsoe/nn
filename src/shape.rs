#[derive(Clone)]
pub struct Shape(Vec<usize>);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<(usize,)> for Shape {
    fn from(d1: (usize,)) -> Self {
        Self(vec![d1.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d12: (usize, usize)) -> Self {
        Self(vec![d12.0, d12.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d123: (usize, usize, usize)) -> Self {
        Self(vec![d123.0, d123.1, d123.2])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}
