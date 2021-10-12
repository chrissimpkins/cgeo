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

impl IPoint {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FPoint {
    x: f64,
    y: f64,
}

impl Default for FPoint {
    fn default() -> Self {
        Self::new(0.0, 0.0)
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
        assert_eq!(pt1, pt2);
        assert_ne!(pt1, pt3);
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
        assert_eq!(pt1, pt2);
        assert_ne!(pt1, pt3);
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
