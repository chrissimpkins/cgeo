use std::fmt;

use approx::relative_eq;

// #[derive(Copy, Clone, Debug, PartialEq)]
// struct Point<T, U> {
//     x: T,
//     y: U,
// }

// impl<T, U> Point<T, U> {
//     pub fn new(x: T, y: U) -> Self {
//         Self { x, y }
//     }
// }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IPoint {
    x: i64,
    y: i64,
}

impl Default for IPoint {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl fmt::Display for IPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// The PartialEq trait for the IPoint struct uses the relative epsilon float equality testing implementation
/// as defined by the approx library. i64 coordinate values are cast to f64 for comparisons. The test uses a
/// relative comparison if the values are far apart. This implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
impl PartialEq<FPoint> for IPoint {
    fn eq(&self, other: &FPoint) -> bool {
        relative_eq!(self.x as f64, other.x) && relative_eq!(self.y as f64, other.y)
    }
}

impl IPoint {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FPoint {
    x: f64,
    y: f64,
}

impl Default for FPoint {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// The PartialEq trait for the FPoint struct uses the relative epsilon float equality testing implementation
/// as defined by the approx library. The test uses a relative comparison if the values are far apart.
/// This implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
impl PartialEq for FPoint {
    fn eq(&self, other: &FPoint) -> bool {
        relative_eq!(self.x, other.x) && relative_eq!(self.y, other.y)
    }
}

impl PartialEq<IPoint> for FPoint {
    fn eq(&self, other: &IPoint) -> bool {
        relative_eq!(self.x, other.x as f64) && relative_eq!(self.y, other.y as f64)
    }
}

impl FPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn approx_eq(&self, f: &FPoint) -> bool {
        relative_eq!(self.x, f.x) && relative_eq!(self.y, f.y)
    }
}

impl fmt::Display for FPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    const ABOVE_F64_EPSILON: f64 = 0.0000000000000002220446049250314;

    // ~~~~~~~~~~~~
    // IPoint
    // ~~~~~~~~~~~~
    #[test]
    fn ipoint_default() {
        let pt = IPoint::default();
        assert_eq!(pt.x, 0);
        assert_eq!(pt.y, 0);
    }

    #[test]
    fn ipoint_new() {
        let pt = IPoint::new(2, 3);
        assert_eq!(pt.x, 2);
        assert_eq!(pt.y, 3);
    }

    #[test]
    fn ipoint_instance_equality() {
        let pt1 = IPoint::new(3, 4);
        let pt2 = IPoint::new(3, 4);
        let pt3 = IPoint::new(4, 5);
        let pt4 = IPoint::new(3, 5);
        let pt5 = IPoint::new(4, 4);
        assert_eq!(pt1, pt2);
        assert_ne!(pt1, pt3);
        assert_ne!(pt1, pt4); // y differ
        assert_ne!(pt1, pt5); // x differ
    }

    #[test]
    fn ipoint_fpoint_equality() {
        let pt1 = IPoint::new(3, 4);
        let pt2 = FPoint::new(3.0, 4.0);
        let pt3 = FPoint::new(3.0001, 4.0001);
        let pt4 = IPoint::new(0, 0);
        let pt5 = FPoint::new(0.0, 0.0);
        let pt6 = IPoint::new(-2, -3);
        let pt7 = FPoint::new(-2.0, -3.0);
        let x_exceed_epsilon = 2.0 + ABOVE_F64_EPSILON; // just beyond f64::EPSILON
        let y_exceed_epsilon = 3.0 + ABOVE_F64_EPSILON; // just beyond f64::EPSILON
        let pt8 = FPoint::new(x_exceed_epsilon, y_exceed_epsilon);
        let pt9 = IPoint::new(2, 3);
        let x_zero_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON;
        let y_zero_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON;
        let pt10 = FPoint::new(x_zero_exceed_epsilon, y_zero_exceed_epsilon);
        assert!(pt1 == pt2);
        assert!(pt1 != pt3);
        assert!(pt4 == pt5);
        assert!(pt6 == pt7);
        assert!(x_exceed_epsilon != 2.0);
        assert!(y_exceed_epsilon != 3.0);
        // relative equality resolves to true for values that are
        // significantly above or below "zero".
        assert!(pt9 == pt8);
        assert!(pt8 == pt9);
        // but not when coordinate values approach "zero"
        // where a point is considered "different"
        assert!(x_zero_exceed_epsilon != 0.0);
        assert!(y_zero_exceed_epsilon != 0.0);
        assert!(pt4 != pt10);
        assert!(pt10 != pt4);
    }

    #[test]
    fn ipoint_display_trait() {
        let pt = IPoint::new(1, 2);
        assert_eq!(format!("{}", pt), "(1, 2)");
    }

    // FPoint
    #[test]
    fn fpoint_default() {
        let pt = FPoint::default();
        assert_eq!(pt.x, 0.0);
        assert_eq!(pt.y, 0.0);
    }

    #[test]
    fn fpoint_new() {
        let pt = FPoint::new(1.01, 2.03);
        assert_eq!(pt.x, 1.01);
        assert_eq!(pt.y, 2.03);
    }

    #[test]
    fn fpoint_instance_equality() {
        let pt1 = FPoint::new(1.01, 2.02);
        let pt2 = FPoint::new(1.01, 2.02);
        let pt3 = FPoint::new(1.001, 2.002);
        assert_eq!(pt1, pt2);
        assert_ne!(pt1, pt3);
    }

    #[test]
    fn fpoint_instance_equality_zeroes() {
        let pt1 = FPoint::new(0.0, 0.0);
        let pt2 = FPoint::new(0.0, 0.0);
        let pt3 = FPoint::new(0.00000001, 0.00000002);
        let x_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let y_exceed_epsilon = 0.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let pt4 = FPoint::new(x_exceed_epsilon, y_exceed_epsilon);
        let pt5 = IPoint::new(0, 0);
        assert_ne!(x_exceed_epsilon, 0.0);
        assert_ne!(y_exceed_epsilon, 0.0);
        assert!(pt1 == pt2);
        assert!(pt1 != pt3);
        assert!(pt3 != pt1);
        // points above epsilon are "different" at values near zero
        // where the relative difference is magnified
        assert!(pt1 != pt4);
        assert!(pt4 != pt1);
        // and on comparisons with integer casts that occur with
        // IPoint to FPoint tests
        assert!(pt4 != pt5);
        assert!(pt5 != pt4);
    }

    #[test]
    fn fpoint_ipoint_equality() {
        let pt1 = FPoint::new(3.0, 4.0);
        let pt2 = IPoint::new(3, 4);
        let pt3 = FPoint::new(3.0001, 4.0001);
        let pt4 = IPoint::new(0, 0);
        let pt5 = FPoint::new(0.0, 0.0);
        let pt6 = IPoint::new(-2, -3);
        let pt7 = FPoint::new(-2.0, -3.0);
        let x_exceed_epsilon = 1.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let y_exceed_epsilon = 1.0 + ABOVE_F64_EPSILON; // just above f64::EPSILON
        let pt8 = FPoint::new(x_exceed_epsilon, y_exceed_epsilon);
        let pt9 = IPoint::new(1, 1);
        assert_ne!(x_exceed_epsilon, 0.01);
        assert_ne!(y_exceed_epsilon, 0.01);
        assert!(pt1 == pt2);
        assert!(pt3 != pt2);
        assert!(pt5 == pt4);
        assert!(pt7 == pt6);
        // points are "the same" at locations significantly above / below
        // "zero" even though the absolute difference is above the f64 epsilon value
        assert!(pt9 == pt8);
        assert!(pt8 == pt9);
    }

    #[test]
    fn fpoint_approx_eq() {
        let pt1 = FPoint::new(-1.0, 1.0);
        let pt2 = FPoint::new(-1.0, 1.0);
        let pt3 = FPoint::new(-1.00000000000001, 1.0);
        let pt4 = FPoint::new(0.0, 0.0);
        let pt5 = FPoint::new(0.000, 0.000);
        assert!(pt1.approx_eq(&pt2));
        assert!(!pt1.approx_eq(&pt3));
        assert!(!pt1.approx_eq(&pt4));
        assert!(pt4.approx_eq(&pt5));
    }

    #[test]
    fn fpoint_display_trait() {
        let pt = FPoint::new(1.001, 2.02);
        assert_eq!(format!("{}", pt), "(1.001, 2.02)");
    }
}
