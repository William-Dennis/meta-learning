
use std::f64::consts::PI;

/// 2D Rastrigin function
#[inline]
pub fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let scale = 1.5;
    let x_scaled = x / scale;
    let y_scaled = y / scale;
    20.0 + x_scaled.powi(2) - 10.0 * (2.0 * PI * x_scaled).cos()
        + y_scaled.powi(2) - 10.0 * (2.0 * PI * y_scaled).cos()
}

/// Simple 2D quadratic function: z = x^2 + y^2
#[inline]
pub fn quadratic_2d(x: f64, y: f64) -> f64 {
    x.powi(2) + y.powi(2)
}