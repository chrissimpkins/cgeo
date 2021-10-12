use std::fmt;

use approx::relative_eq;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct I2DCoordinate {
    x: i64,
    y: i64,
}

impl Default for I2DCoordinate {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl fmt::Display for I2DCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// The PartialEq trait for the I2DCoordinate struct uses the relative epsilon float equality testing implementation
/// as defined by the approx library. i64 coordinate values are cast to f64 for comparisons. The test uses a
/// relative comparison if the values are far apart. This implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
impl PartialEq<F2DCoordinate> for I2DCoordinate {
    fn eq(&self, other: &F2DCoordinate) -> bool {
        relative_eq!(self.x as f64, other.x) && relative_eq!(self.y as f64, other.y)
    }
}

impl I2DCoordinate {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    pub fn into_f2dcoord(self) -> F2DCoordinate {
        self.into()
    }
}

impl From<F2DCoordinate> for I2DCoordinate {
    fn from(item: F2DCoordinate) -> Self {
        I2DCoordinate { x: item.x.round() as i64, y: item.y.round() as i64 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct F2DCoordinate {
    x: f64,
    y: f64,
}

impl Default for F2DCoordinate {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl fmt::Display for F2DCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// The PartialEq trait for the F2DCoordinate struct uses the relative epsilon float equality testing implementation
/// as defined by the approx library. The test uses a relative comparison if the values are far apart.
/// This implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
impl PartialEq for F2DCoordinate {
    fn eq(&self, other: &F2DCoordinate) -> bool {
        relative_eq!(self.x, other.x) && relative_eq!(self.y, other.y)
    }
}

impl PartialEq<I2DCoordinate> for F2DCoordinate {
    fn eq(&self, other: &I2DCoordinate) -> bool {
        relative_eq!(self.x, other.x as f64) && relative_eq!(self.y, other.y as f64)
    }
}

impl F2DCoordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn approx_eq(&self, f: &F2DCoordinate) -> bool {
        relative_eq!(self.x, f.x) && relative_eq!(self.y, f.y)
    }

    pub fn into_i2dcoord(self) -> I2DCoordinate {
        self.into()
    }
}

impl From<I2DCoordinate> for F2DCoordinate {
    fn from(item: I2DCoordinate) -> Self {
        F2DCoordinate { x: item.x as f64, y: item.y as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    const ABOVE_F64_EPSILON: f64 = 0.0000000000000002220446049250314;

    // ~~~~~~~~~~~~
    // I2DCoordinate
    // ~~~~~~~~~~~~
    #[test]
    fn i2dcoordinate_default() {
        let coord = I2DCoordinate::default();
        assert_eq!(coord.x, 0);
        assert_eq!(coord.y, 0);
    }

    #[test]
    fn i2dcoordinate_new() {
        let coord = I2DCoordinate::new(2, 3);
        assert_eq!(coord.x, 2);
        assert_eq!(coord.y, 3);
    }

    #[test]
    fn i2dcoordinate_instance_equality() {
        let coord1 = I2DCoordinate::new(3, 4);
        let coord2 = I2DCoordinate::new(3, 4);
        let coord3 = I2DCoordinate::new(4, 5);
        let coord4 = I2DCoordinate::new(3, 5);
        let coord5 = I2DCoordinate::new(4, 4);
        assert_eq!(coord1, coord2);
        assert_ne!(coord1, coord3);
        assert_ne!(coord1, coord4); // y differ
        assert_ne!(coord1, coord5); // x differ
    }

    #[test]
    fn i2dcoordinate_f2dcoordinate_equality() {
        let coord1 = I2DCoordinate::new(3, 4);
        let coord2 = F2DCoordinate::new(3.0, 4.0);
        let coord3 = F2DCoordinate::new(3.0001, 4.0001);
        let coord4 = I2DCoordinate::new(0, 0);
        let coord5 = F2DCoordinate::new(0.0, 0.0);
        let coord6 = I2DCoordinate::new(-2, -3);
        let coord7 = F2DCoordinate::new(-2.0, -3.0);
        let x_exceed_epsilon = 2.0 + ABOVE_F64_EPSILON; // just beyond f64::EPSILON
        let y_exceed_epsilon = 3.0 + ABOVE_F64_EPSILON; // just beyond f64::EPSILON
        let coord8 = F2DCoordinate::new(x_exceed_epsilon, y_exceed_epsilon);
        let coord9 = I2DCoordinate::new(2, 3);
        let x_zero_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON;
        let y_zero_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON;
        let coord10 = F2DCoordinate::new(x_zero_exceed_epsilon, y_zero_exceed_epsilon);
        assert!(coord1 == coord2);
        assert!(coord1 != coord3);
        assert!(coord4 == coord5);
        assert!(coord6 == coord7);
        assert!(x_exceed_epsilon != 2.0);
        assert!(y_exceed_epsilon != 3.0);
        // relative equality resolves to true for values that are
        // significantly above or below "zero".
        assert!(coord9 == coord8);
        assert!(coord8 == coord9);
        // but not when coordinate values approach "zero"
        // where a coordinate is considered "different"
        assert!(x_zero_exceed_epsilon != 0.0);
        assert!(y_zero_exceed_epsilon != 0.0);
        assert!(coord4 != coord10);
        assert!(coord10 != coord4);
    }

    #[test]
    fn i2dcoordinate_display_trait() {
        let coord = I2DCoordinate::new(1, 2);
        assert_eq!(format!("{}", coord), "(1, 2)");
    }

    #[test]
    fn i2dcoordinate_from_into_trait_cast_f2dcoordinate() {
        let coord1 = I2DCoordinate::new(1, 2);
        let coord2 = F2DCoordinate::new(1.0, 2.0);
        assert_eq!(F2DCoordinate::from(coord1), coord2);
        assert_eq!(coord1.into_f2dcoord(), coord2);
    }

    // F2DCoordinate
    #[test]
    fn f2dcoordinate_default() {
        let coord = F2DCoordinate::default();
        assert_eq!(coord.x, 0.0);
        assert_eq!(coord.y, 0.0);
    }

    #[test]
    fn f2dcoordinate_new() {
        let coord = F2DCoordinate::new(1.01, 2.03);
        assert_eq!(coord.x, 1.01);
        assert_eq!(coord.y, 2.03);
    }

    #[test]
    fn f2dcoordinate_instance_equality() {
        let coord1 = F2DCoordinate::new(1.01, 2.02);
        let coord2 = F2DCoordinate::new(1.01, 2.02);
        let coord3 = F2DCoordinate::new(1.001, 2.002);
        assert_eq!(coord1, coord2);
        assert_ne!(coord1, coord3);
    }

    #[test]
    fn f2dcoordinate_instance_equality_zeroes() {
        let coord1 = F2DCoordinate::new(0.0, 0.0);
        let coord2 = F2DCoordinate::new(0.0, 0.0);
        let coord3 = F2DCoordinate::new(0.00000001, 0.00000002);
        let x_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let y_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let coord4 = F2DCoordinate::new(x_exceed_epsilon, y_exceed_epsilon);
        let coord5 = I2DCoordinate::new(0, 0);
        assert_ne!(x_exceed_epsilon, 0.0);
        assert_ne!(y_exceed_epsilon, 0.0);
        assert!(coord1 == coord2);
        assert!(coord1 != coord3);
        assert!(coord3 != coord1);
        // coords with differences above epsilon are "different" at values near zero
        // where the relative difference is magnified
        assert!(coord1 != coord4);
        assert!(coord4 != coord1);
        // and on comparisons with integer casts that occur with
        // I2DCoordinate to F2DCoordinate tests
        assert!(coord4 != coord5);
        assert!(coord5 != coord4);
    }

    #[test]
    fn f2dcoordinate_i2dcoordinate_equality() {
        let coord1 = F2DCoordinate::new(3.0, 4.0);
        let coord2 = I2DCoordinate::new(3, 4);
        let coord3 = F2DCoordinate::new(3.0001, 4.0001);
        let coord4 = I2DCoordinate::new(0, 0);
        let coord5 = F2DCoordinate::new(0.0, 0.0);
        let coord6 = I2DCoordinate::new(-2, -3);
        let coord7 = F2DCoordinate::new(-2.0, -3.0);
        let x_exceed_epsilon = 1.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let y_exceed_epsilon = 1.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let coord8 = F2DCoordinate::new(x_exceed_epsilon, y_exceed_epsilon);
        let coord9 = I2DCoordinate::new(1, 1);
        assert_ne!(x_exceed_epsilon, 0.01);
        assert_ne!(y_exceed_epsilon, 0.01);
        assert!(coord1 == coord2);
        assert!(coord3 != coord2);
        assert!(coord5 == coord4);
        assert!(coord7 == coord6);
        // coords are "the same" at locations significantly above / below
        // "zero" even though the absolute difference is above the f64 epsilon value
        assert!(coord9 == coord8);
        assert!(coord8 == coord9);
    }

    #[test]
    fn f2dcoordinate_approx_eq() {
        let coord1 = F2DCoordinate::new(-1.0, 1.0);
        let coord2 = F2DCoordinate::new(-1.0, 1.0);
        let coord3 = F2DCoordinate::new(-1.00000000000001, 1.0);
        let coord4 = F2DCoordinate::new(0.0, 0.0);
        let coord5 = F2DCoordinate::new(0.000, 0.000);
        assert!(coord1.approx_eq(&coord2));
        assert!(!coord1.approx_eq(&coord3));
        assert!(!coord1.approx_eq(&coord4));
        assert!(coord4.approx_eq(&coord5));
    }

    #[test]
    fn f2dcoordinate_display_trait() {
        let coord = F2DCoordinate::new(1.001, 2.02);
        assert_eq!(format!("{}", coord), "(1.001, 2.02)");
    }

    #[test]
    fn f2dcoordinate_from_into_trait_cast_i2dcoordinate() {
        let coord1 = I2DCoordinate::new(1, 2);
        let coord2 = F2DCoordinate::new(1.0, 2.0);
        let coord3 = F2DCoordinate::new(1.1, 1.9);
        assert_eq!(I2DCoordinate::from(coord2), coord1);
        assert_eq!(coord2.into_i2dcoord(), coord1);
        // the cast uses f64.round() as i64
        assert_eq!(I2DCoordinate::from(coord3), coord1);
        assert_eq!(coord3.into_i2dcoord(), coord1);
    }
}
