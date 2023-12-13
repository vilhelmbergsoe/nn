// TODO: add f16 and bf16 support from half
#[derive(Debug, Clone)]
pub enum DType {
    F32,
    F64,
}

pub trait WithDType {
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }
        }
    };
}

with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
