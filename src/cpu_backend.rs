#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct CpuDevice;
