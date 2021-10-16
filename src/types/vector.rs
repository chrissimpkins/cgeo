use std::ops::{Add, Mul, Neg, Sub};

use super::coordinate::{F2DCoordinate, I2DCoordinate};

// trait Vectorable {
//     // marker trait for now
// }

/// A 2D vector struct with [`i64`] coordinates
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector2DInt {
    pub coord: I2DCoordinate,
}

impl Vector2DInt {
    pub fn new(begin: (i64, i64), end: (i64, i64)) -> Self {
        Self { coord: I2DCoordinate::new(end.0 - begin.0, end.1 - begin.1) }
    }

    pub fn new_bound(end: (i64, i64)) -> Self {
        Self { coord: I2DCoordinate::new(end.0, end.1) }
    }
}

/// [`Vector2DInt`] addition with the `+` operator
impl Add for Vector2DInt {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { coord: I2DCoordinate::new(self.coord.x + rhs.coord.x, self.coord.y + rhs.coord.y) }
    }
}

/// [`Vector2DInt`] subtraction with the `-` operator
impl Sub for Vector2DInt {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { coord: I2DCoordinate::new(self.coord.x - rhs.coord.x, self.coord.y - rhs.coord.y) }
    }
}

/// [`Vector2DInt`] unary negation with the `-` operator
impl Neg for Vector2DInt {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { coord: I2DCoordinate::new(-self.coord.x, -self.coord.y) }
    }
}

/// [`Vector2DInt`] scalar multiplication with i64 using the `*` operator
impl Mul<i64> for Vector2DInt {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self { coord: I2DCoordinate::new(rhs * self.coord.x, rhs * self.coord.y) }
    }
}

/// [`Vector2DFloat`] to [`Vector2DInt`] type casts
impl From<Vector2DFloat> for Vector2DInt {
    fn from(item: Vector2DFloat) -> Self {
        Self { coord: I2DCoordinate::from(item.coord) }
    }
}

/// A 2D vector struct with [`f64`] coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vector2DFloat {
    pub coord: F2DCoordinate,
}

impl Vector2DFloat {
    pub fn new(begin: (f64, f64), end: (f64, f64)) -> Self {
        Self { coord: F2DCoordinate::new(end.0 - begin.0, end.1 - begin.1) }
    }

    pub fn new_bound(end: (f64, f64)) -> Self {
        Self { coord: F2DCoordinate::new(end.0, end.1) }
    }
}

/// [`Vector2DFloat`] addition with the `+` operator
impl Add for Vector2DFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { coord: F2DCoordinate::new(self.coord.x + rhs.coord.x, self.coord.y + rhs.coord.y) }
    }
}

/// [`Vector2DFloat`] subtraction with the `-` operator
impl Sub for Vector2DFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { coord: F2DCoordinate::new(self.coord.x - rhs.coord.x, self.coord.y - rhs.coord.y) }
    }
}

/// [`Vector2DFloat`] unary negation with the `-` operator
impl Neg for Vector2DFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { coord: F2DCoordinate::new(-self.coord.x, -self.coord.y) }
    }
}

/// [`Vector2DFloat`] scalar multiplication with [`f64`] using the `*` operator
impl Mul<f64> for Vector2DFloat {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self { coord: F2DCoordinate::new(rhs * self.coord.x, rhs * self.coord.y) }
    }
}

/// [`Vector2DFloat`] scalar multiplication with [`i64`] using the `*` operator
impl Mul<i64> for Vector2DFloat {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self { coord: F2DCoordinate::new(rhs as f64 * self.coord.x, rhs as f64 * self.coord.y) }
    }
}

/// [`Vector2DInt`] to [`Vector2DFloat`] type casts
impl From<Vector2DInt> for Vector2DFloat {
    fn from(item: Vector2DInt) -> Self {
        Self { coord: F2DCoordinate::from(item.coord) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn vector2dint_instantiation() {
        let v = Vector2DInt::new((1, 2), (3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(2, 2));

        let v = Vector2DInt::new_bound((3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(3, 4));
    }

    #[test]
    fn vector2dint_add_operator() {
        let v1 = Vector2DInt::new_bound((5, 5));
        let v2 = Vector2DInt::new_bound((4, 4));
        let v3 = Vector2DInt::new_bound((1, 1));
        let v_zero = Vector2DInt::new_bound((0, 0));
        assert_eq!(v1 + v2, Vector2DInt::new_bound((9, 9)));
        assert_eq!(v2 + v1, Vector2DInt::new_bound((9, 9)));
        assert_eq!(v1 + v2 + v3, Vector2DInt::new_bound((10, 10)));
        assert_eq!((v1 + v2) + v3, Vector2DInt::new_bound((10, 10)));
        assert_eq!(v1 + (v2 + v3), Vector2DInt::new_bound((10, 10)));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(v1 + (-v1), v_zero);
    }

    #[test]
    fn vector2dint_sub_operator() {
        let v1 = Vector2DInt::new_bound((5, 5));
        let v2 = Vector2DInt::new_bound((4, 4));
        let v3 = Vector2DInt::new_bound((1, 1));
        let v_zero = Vector2DInt::new_bound((0, 0));
        assert_eq!(v1 - v2, Vector2DInt::new_bound((1, 1)));
        assert_eq!(v3 - v1, Vector2DInt::new_bound((-4, -4)));
        assert_eq!(v1 - v_zero, v1);
        assert_eq!(v1 - v2 - v3, v_zero);
        assert_eq!((v1 - v2) - v3, v_zero);
        assert_eq!(v1 - (v2 - v3), Vector2DInt::new_bound((2, 2)));
        assert_eq!(v1 - (v2 + v3), v_zero);
        assert_eq!(v2 - (-v3), v1);
    }

    #[test]
    fn vector2dint_mul_operator() {
        let v1 = Vector2DInt::new_bound((2, 3));
        let v2 = Vector2DInt::new_bound((-3, -2));
        let v_zero = Vector2DInt::new_bound((0, 0));
        assert_eq!(v1 * 2, Vector2DInt::new_bound((4, 6)));
        assert_eq!(v2 * 2, Vector2DInt::new_bound((-6, -4)));
        assert_eq!(v_zero * 10, Vector2DInt::new_bound((0, 0)));
        assert_eq!(v_zero * 10, v_zero);
        assert_eq!((v1 * 3) * 6, v1 * (6 * 3));
        assert_eq!(v1 * (6 + 3), (v1 * 6) + (v1 * 3));
        assert_eq!(v1 * (6 - 3), (v1 * 6) - (v1 * 3));
        assert_eq!((v1 + v2) * 6, (v1 * 6) + (v2 * 6));
        assert_eq!((v1 - v2) * 6, (v1 * 6) - (v2 * 6));
    }

    #[test]
    fn vector2dint_from_into_vector2dfloat() {
        let vi1 = Vector2DInt::new_bound((1, 2));
        let vi2 = Vector2DInt::new_bound((2, 3));
        let vi3 = Vector2DInt::new_bound((-1, -2));
        let vi4 = Vector2DInt::new_bound((-2, -3));
        let vf1 = Vector2DFloat::new_bound((1.4, 2.4));
        let vf2 = Vector2DFloat::new_bound((1.6, 2.6));
        let vf3 = Vector2DFloat::new_bound((-1.4, -2.4));
        let vf4 = Vector2DFloat::new_bound((-1.6, -2.6));
        assert_eq!(Vector2DInt::from(vf1), vi1);
        assert_ne!(Vector2DInt::from(vf2), vi1);
        assert_eq!(Vector2DInt::from(vf2), vi2);
        assert_eq!(Vector2DInt::from(vf3), vi3);
        assert_ne!(Vector2DInt::from(vf4), vi3);
        assert_eq!(Vector2DInt::from(vf4), vi4);
        let x: Vector2DInt = vf1.into();
        assert_eq!(x, vi1);
    }

    #[test]
    fn vector2dfloat_instantiation() {
        let v = Vector2DFloat::new((1.0, 2.0), (3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(2.123, 2.321));

        let v = Vector2DFloat::new_bound((3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(3.123, 4.321));
    }

    #[test]
    fn vector2dfloat_add_operator() {
        let v1 = Vector2DFloat::new_bound((5.1, 5.1));
        let v2 = Vector2DFloat::new_bound((4.2, 4.2));
        let v3 = Vector2DFloat::new_bound((1.3, 1.3));
        let v_zero = Vector2DFloat::new_bound((0.0, 0.0));
        assert_eq!(v1 + v2, Vector2DFloat::new_bound((5.1 + 4.2, 5.1 + 4.2)));
        assert_eq!(v2 + v1, Vector2DFloat::new_bound((5.1 + 4.2, 5.1 + 4.2)));
        assert_eq!(v1 + v2 + v3, Vector2DFloat::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!((v1 + v2) + v3, Vector2DFloat::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!(v1 + (v2 + v3), Vector2DFloat::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(v1 + (-v1), v_zero);
    }

    #[test]
    fn vector2dfloat_sub_operator() {
        let v1 = Vector2DFloat::new_bound((5.3, 5.3));
        let v2 = Vector2DFloat::new_bound((4.2, 4.2));
        let v3 = Vector2DFloat::new_bound((1.1, 1.1));
        let v_zero = Vector2DFloat::new_bound((0.0, 0.0));
        assert_eq!(v1 - v2, Vector2DFloat::new_bound((5.3 - 4.2, 5.3 - 4.2)));
        assert_eq!(v3 - v1, Vector2DFloat::new_bound((1.1 - 5.3, 1.1 - 5.3)));
        assert_eq!(v1 - v_zero, Vector2DFloat::new_bound((5.3 - 0.0, 5.3 - 0.0)));
        assert_eq!(v1 - v2 - v3, Vector2DFloat::new_bound((5.3 - 4.2 - 1.1, 5.3 - 4.2 - 1.1)));
        assert_eq!((v1 - v2) - v3, Vector2DFloat::new_bound((5.3 - 4.2 - 1.1, 5.3 - 4.2 - 1.1)));
        assert_eq!(
            v1 - (v2 - v3),
            Vector2DFloat::new_bound((5.3 - (4.2 - 1.1), 5.3 - (4.2 - 1.1)))
        );
        assert_eq!(
            v1 - (v2 + v3),
            Vector2DFloat::new_bound((5.3 - (4.2 + 1.1), 5.3 - (4.2 + 1.1)))
        );
        assert_eq!(v2 - (-v3), Vector2DFloat::new_bound((4.2 - (-1.1), 4.2 - (-1.1))));
    }

    #[test]
    fn vector2dfloat_mul_f64_operator() {
        let v1 = Vector2DFloat::new_bound((2.0, 3.0));
        let v2 = Vector2DFloat::new_bound((-3.0, -2.0));
        let v_zero = Vector2DFloat::new_bound((0.0, 0.0));
        assert_eq!(v1 * 2, Vector2DFloat::new_bound((2.0_f64 * 2.0_f64, 3.0_f64 * 2.0_f64)));
        assert_eq!(v2 * 2, Vector2DFloat::new_bound((-3.0_f64 * 2.0_f64, -2.0_f64 * 2.0_f64)));
        assert_eq!(v_zero * 10, Vector2DFloat::new_bound((0.0_f64 * 10.0_f64, 0.0_f64 * 10.0_f64)));
        assert_eq!((v1 * 3.0) * 6.0, v1 * (6.0 * 3.0));
        assert_eq!(v1 * (6.0 + 3.0), (v1 * 6.0) + (v1 * 3.0));
        assert_eq!(v1 * (6.0 - 3.0), (v1 * 6.0) - (v1 * 3.0));
        assert_eq!((v1 + v2) * 6.0, (v1 * 6.0) + (v2 * 6.0));
        assert_eq!((v1 - v2) * 6.0, (v1 * 6.0) - (v2 * 6.0));
    }

    #[test]
    fn vector2dfloat_mul_i64_operator() {
        let v1 = Vector2DFloat::new_bound((2.0, 3.0));
        let v2 = Vector2DFloat::new_bound((-3.0, -2.0));
        let v_zero = Vector2DFloat::new_bound((0.0, 0.0));
        assert_eq!(v1 * 2_i64, Vector2DFloat::new_bound((2.0_f64 * 2.0_f64, 3.0_f64 * 2.0_f64)));
        assert_eq!(v2 * 2_i64, Vector2DFloat::new_bound((-3.0_f64 * 2.0_f64, -2.0_f64 * 2.0_f64)));
        assert_eq!(
            v_zero * 10_i64,
            Vector2DFloat::new_bound((0.0_f64 * 10.0_f64, 0.0_f64 * 10.0_f64))
        );
        assert_eq!((v1 * 3_i64) * 6_i64, v1 * (6 as f64 * 3 as f64));
        assert_eq!(v1 * (6_i64 + 3_i64), (v1 * 6 as f64) + (v1 * 3 as f64));
        assert_eq!(v1 * (6_i64 - 3_i64), (v1 * 6 as f64) - (v1 * 3 as f64));
        assert_eq!((v1 + v2) * 6_i64, (v1 * 6 as f64) + (v2 * 6 as f64));
        assert_eq!((v1 - v2) * 6_i64, (v1 * 6 as f64) - (v2 * 6 as f64));
    }

    #[test]
    fn vector2float_from_into_vector2dint() {
        let vi1 = Vector2DInt::new_bound((1, 2));
        let vi2 = Vector2DInt::new_bound((2, 3));
        let vi3 = Vector2DInt::new_bound((-1, -2));
        let vi4 = Vector2DInt::new_bound((-2, -3));
        let vf1 = Vector2DFloat::new_bound((1 as f64, 2 as f64));
        let vf2 = Vector2DFloat::new_bound((2 as f64, 3 as f64));
        let vf3 = Vector2DFloat::new_bound((-1 as f64, -2 as f64));
        let vf4 = Vector2DFloat::new_bound((-2 as f64, -3 as f64));
        assert_eq!(Vector2DFloat::from(vi1), vf1);
        assert_ne!(Vector2DFloat::from(vi2), vf1);
        assert_eq!(Vector2DFloat::from(vi2), vf2);
        assert_eq!(Vector2DFloat::from(vi3), vf3);
        assert_ne!(Vector2DFloat::from(vi4), vf3);
        assert_eq!(Vector2DFloat::from(vi4), vf4);
        let x: Vector2DFloat = vi1.into();
        assert_eq!(x, vf1);
    }
}
