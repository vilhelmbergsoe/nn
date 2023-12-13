use crate::dtype::DType;
use crate::device::Device;
use crate::cpu_backend::CpuStorage;

pub enum Storage {
    Cpu(CpuStorage),
    // TODO: WebGPU / Cuda - HIP - ROCM
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
        }
    }
}
