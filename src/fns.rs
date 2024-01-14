use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}
