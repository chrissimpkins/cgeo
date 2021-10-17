use std::ops::{Add, Mul, Neg, Sub};

use super::coordinate::{F2DCoordinate, I2DCoordinate};

// trait Vectorable {
//     // marker trait for now
// }

/// A 2D vector struct with [`i64`] coordinates
#[derive(Copy, Clone, Debug)]
pub struct Vector2DInt {
    /// The vector start coordinates.
    pub begin: I2DCoordinate,
    /// The bound vector coordinates where a bound vector is defined as having
    /// a start (0, 0) origin coordinate.
    pub coord: I2DCoordinate,
}

impl Vector2DInt {
    pub fn new(begin: (i64, i64), end: (i64, i64)) -> Self {
        Self {
            begin: I2DCoordinate::new(begin.0, begin.1),
            coord: I2DCoordinate::new(end.0 - begin.0, end.1 - begin.1),
        }
    }

    pub fn new_bound(end: (i64, i64)) -> Self {
        Self { begin: I2DCoordinate::origin(), coord: I2DCoordinate::new(end.0, end.1) }
    }

    pub fn zero() -> Self {
        Self::new_bound((0, 0))
    }

    /// Euclidean vector magnitude.
    pub fn magnitude(&self) -> f64 {
        // uses the dot product approach for performance
        (self.dot_product(self) as f64).sqrt()
    }

    pub fn begin(&self) -> &I2DCoordinate {
        &self.begin
    }

    pub fn end(&self) -> I2DCoordinate {
        I2DCoordinate::new(self.coord.x + self.begin.x, self.coord.y + self.begin.y)
    }

    pub fn dot_product(&self, other: &Vector2DInt) -> i64 {
        (self.coord.x * other.coord.x) + (self.coord.y * other.coord.y)
    }
}

/// [`Vector2DInt`] addition with the `+` operator
impl Add for Vector2DInt {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            begin: self.begin,
            coord: I2DCoordinate::new(self.coord.x + rhs.coord.x, self.coord.y + rhs.coord.y),
        }
    }
}

/// [`Vector2DInt`] subtraction with the `-` operator
impl Sub for Vector2DInt {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            begin: self.begin,
            coord: I2DCoordinate::new(self.coord.x - rhs.coord.x, self.coord.y - rhs.coord.y),
        }
    }
}

/// [`Vector2DInt`] unary negation with the `-` operator
impl Neg for Vector2DInt {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new((self.end().x, self.end().y), (self.begin().x, self.begin().y))
    }
}

/// [`Vector2DInt`] scalar multiplication with i64 using the `*` operator
impl Mul<i64> for Vector2DInt {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            begin: self.begin,
            coord: I2DCoordinate::new(rhs * self.coord.x, rhs * self.coord.y),
        }
    }
}

/// Partial equivalence is defined based on vector magnitude and
/// direction comparisons only.  Vectors with different begin and
/// end coordinates are defined as equivalent when they have the same
/// magnitude and direction but different locations in coordinate space.
impl PartialEq for Vector2DInt {
    fn eq(&self, other: &Vector2DInt) -> bool {
        self.coord == other.coord
    }
}

/// [`Vector2DFloat`] to [`Vector2DInt`] type casts
impl From<Vector2DFloat> for Vector2DInt {
    fn from(item: Vector2DFloat) -> Self {
        Self { begin: I2DCoordinate::from(item.begin), coord: I2DCoordinate::from(item.coord) }
    }
}

/// A 2D vector struct with [`f64`] coordinates.
#[derive(Copy, Clone, Debug)]
pub struct Vector2DFloat {
    /// The vector start coordinates.
    pub begin: F2DCoordinate,
    /// The bound vector coordinates where a bound vector is defined as having
    /// a start (0, 0) origin coordinate.
    pub coord: F2DCoordinate,
}

impl Vector2DFloat {
    pub fn new(begin: (f64, f64), end: (f64, f64)) -> Self {
        Self {
            begin: F2DCoordinate::new(begin.0, begin.1),
            coord: F2DCoordinate::new(end.0 - begin.0, end.1 - begin.1),
        }
    }

    pub fn new_bound(end: (f64, f64)) -> Self {
        Self { begin: F2DCoordinate::origin(), coord: F2DCoordinate::new(end.0, end.1) }
    }

    pub fn zero() -> Self {
        Self::new_bound((0.0, 0.0))
    }

    /// Euclidean vector magnitude.
    pub fn magnitude(&self) -> f64 {
        // uses the dot product approach - consistent with i64 approach
        // but does not appear to confer the same performance benefits in
        // benchmark testing
        (self.dot_product(self)).sqrt()
    }

    pub fn begin(&self) -> &F2DCoordinate {
        &self.begin
    }

    pub fn end(&self) -> F2DCoordinate {
        F2DCoordinate::new(self.coord.x + self.begin.x, self.coord.y + self.begin.y)
    }

    pub fn dot_product(&self, other: &Vector2DFloat) -> f64 {
        (self.coord.x * other.coord.x) + (self.coord.y * other.coord.y)
    }
}

/// [`Vector2DFloat`] addition with the `+` operator
impl Add for Vector2DFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            begin: self.begin,
            coord: F2DCoordinate::new(self.coord.x + rhs.coord.x, self.coord.y + rhs.coord.y),
        }
    }
}

/// [`Vector2DFloat`] subtraction with the `-` operator
impl Sub for Vector2DFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            begin: self.begin,
            coord: F2DCoordinate::new(self.coord.x - rhs.coord.x, self.coord.y - rhs.coord.y),
        }
    }
}

/// [`Vector2DFloat`] unary negation with the `-` operator
impl Neg for Vector2DFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new((self.end().x, self.end().y), (self.begin().x, self.begin().y))
    }
}

/// [`Vector2DFloat`] scalar multiplication with [`f64`] using the `*` operator
impl Mul<f64> for Vector2DFloat {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            begin: self.begin,
            coord: F2DCoordinate::new(rhs * self.coord.x, rhs * self.coord.y),
        }
    }
}

/// [`Vector2DFloat`] scalar multiplication with [`i64`] using the `*` operator
impl Mul<i64> for Vector2DFloat {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            begin: self.begin,
            coord: F2DCoordinate::new(rhs as f64 * self.coord.x, rhs as f64 * self.coord.y),
        }
    }
}

/// Partial equivalence is defined based on vector magnitude and
/// direction comparisons only.  Vectors with different begin and
/// end coordinates are defined as equivalent when they have the same
/// magnitude and direction but different locations in coordinate space.
impl PartialEq for Vector2DFloat {
    fn eq(&self, other: &Vector2DFloat) -> bool {
        self.coord == other.coord
    }
}

/// [`Vector2DInt`] to [`Vector2DFloat`] type casts
impl From<Vector2DInt> for Vector2DFloat {
    fn from(item: Vector2DInt) -> Self {
        Self { begin: F2DCoordinate::from(item.begin), coord: F2DCoordinate::from(item.coord) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::{assert_relative_eq, assert_relative_ne};
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn vector2dint_instantiation() {
        let v = Vector2DInt::new((1, 2), (3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(2, 2));
        assert_eq!(v.begin, I2DCoordinate::new(1, 2));

        let v = Vector2DInt::new_bound((3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(3, 4));

        let v = Vector2DInt::zero();
        assert_eq!(v.coord, I2DCoordinate::new(0, 0));
        assert_eq!(v.begin, I2DCoordinate::new(0, 0));
    }

    #[test]
    fn vector2dint_magnitude() {
        let v1 = Vector2DInt::new((1, 2), (3, 4));
        let v2 = Vector2DInt::new((-1, -2), (-3, -4));
        assert_relative_eq!(v1.magnitude(), 2.8284271247461903);
        assert_relative_eq!(v2.magnitude(), 2.8284271247461903);
    }

    #[test]
    fn vector2dint_begin() {
        let v = Vector2DInt::new((1, 2), (3, 4));
        assert_eq!(v.begin().x, 1);
        assert_eq!(v.begin().y, 2);
    }

    #[test]
    fn vector2dint_end() {
        let v = Vector2DInt::new((1, 2), (3, 4));
        assert_eq!(v.end().x, 3);
        assert_eq!(v.end().y, 4);
    }

    #[test]
    fn vector2dint_dot_product() {
        let v1 = Vector2DInt::new_bound((1, 2));
        let v2 = Vector2DInt::new_bound((3, 4));
        let v3 = Vector2DInt::new_bound((5, 6));
        let v4 = Vector2DInt::new_bound((-3, -4));
        assert_eq!(v1.dot_product(&v2), 11);
        assert_eq!(v1.dot_product(&v4), -11);
        assert_eq!(-v1.dot_product(&-v2), 11);
        assert_eq!(v1.dot_product(&v2), v2.dot_product(&v1));
        let x1 = v1 * 3;
        let x2 = v2 * 6;
        assert_eq!(x1.dot_product(&x2), ((3 * 6) * v1.dot_product(&v2)));
        assert_eq!(v1.dot_product(&(v2 + v3)), v1.dot_product(&v2) + v1.dot_product(&v3));
    }

    #[test]
    fn vector2dint_partialeq() {
        let v1 = Vector2DInt::new((1, 2), (2, 3));
        let v2 = Vector2DInt::new((2, 3), (3, 4));
        let v3 = Vector2DInt::new_bound((10, 10));
        // equivalence defined on magnitude and direction
        // comparison only
        assert_eq!(v1.coord, v2.coord);
        assert_ne!(v1.begin, v2.begin);
        assert_ne!(v1.end(), v2.end());
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn vector2dint_unary_neg_operator() {
        let v1 = Vector2DInt::new((1, 2), (3, 4));
        let v2 = Vector2DInt::new((3, 4), (1, 2));
        let v3 = Vector2DInt::new((-1, -2), (-3, -4));
        let v4 = Vector2DInt::new((-3, -4), (-1, -2));
        assert_eq!(v1.coord.x, 2);
        assert_eq!(v1.coord.y, 2);
        assert_eq!(-v1.coord.x, -2);
        assert_eq!(-v1.coord.y, -2);
        assert_eq!(-v1, -v1);
        assert_eq!(-v1, v2);
        assert_eq!(-v3, -v3);
        assert_eq!(-v3, v4);
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
        let v1_v2 = v1 + v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in addition.
        let v4 = Vector2DInt::new((1, 1), (2, 2));
        let v5 = Vector2DInt::new((2, 2), (3, 3));
        let v4_v5 = v4 + v5;
        let v5_v4 = v5 + v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
        assert_eq!(v4_v5, v5_v4);
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
        let v1_v2 = v1 - v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in subtraction
        let v4 = Vector2DInt::new((1, 1), (2, 2));
        let v5 = Vector2DInt::new((2, 2), (3, 3));
        let v4_v5 = v4 - v5;
        let v5_v4 = v5 - v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
        assert_eq!(v4_v5, v5_v4);
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
        let v3 = Vector2DInt::new((1, 2), (3, 4));
        let v3_2 = v3 * 2;
        assert_eq!(v3_2.begin, v3.begin);
        assert_ne!(v3_2.end(), v3.end());
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
        let x1: Vector2DInt = vf1.into();
        assert_eq!(x1, vi1);
        let vi5 = Vector2DInt::new((1, 2), (3, 4));
        let vi6 = Vector2DInt::new((2, 3), (4, 5));
        let vf5 = Vector2DFloat::new((1.4, 2.4), (3.4, 4.4));
        let vf6 = Vector2DFloat::new((1.6, 2.6), (3.6, 4.6));
        let x2 = Vector2DInt::from(vf5);
        let x3 = Vector2DInt::from(vf6);
        assert_eq!(x2, vi5);
        assert_eq!(x3, vi6);
        // begin/end coordinates differ due to f64 coordinate rounding
        assert_ne!(x2.begin, x3.begin);
        assert_ne!(x2.end(), x3.end());
        // but vectors are considered equivalent with equal
        // magnitude and direction
        assert_eq!(x2, x3);
    }

    #[test]
    fn vector2dfloat_instantiation() {
        let v = Vector2DFloat::new((1.0, 2.0), (3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(2.123, 2.321));
        assert_eq!(v.begin, F2DCoordinate::new(1.0, 2.0));

        let v = Vector2DFloat::new_bound((3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(3.123, 4.321));

        let v = Vector2DInt::zero();
        assert_eq!(v.coord, F2DCoordinate::new(0.0, 0.0));
        assert_eq!(v.begin, F2DCoordinate::new(0.0, 0.0));
    }

    #[test]
    fn vector2dfloat_magnitude() {
        let v = Vector2DFloat::new((1.1, 2.1), (3.9, 4.7));
        assert_relative_eq!(v.magnitude(), 3.82099463490856);
    }

    #[test]
    fn vector2float_begin() {
        let v = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        assert_relative_eq!(v.begin().x, 1.0);
        assert_relative_eq!(v.begin().y, 2.0);
    }

    #[test]
    fn vector2dfloat_end() {
        let v = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        assert_relative_eq!(v.end().x, 3.0);
        assert_relative_eq!(v.end().y, 4.0);
    }

    #[test]
    fn vector2dfloat_dot_product() {
        let v1 = Vector2DFloat::new_bound((1.0, 2.0));
        let v2 = Vector2DFloat::new_bound((3.0, 4.0));
        let v3 = Vector2DFloat::new_bound((5.0, 6.0));
        assert_relative_eq!(v1.dot_product(&v2), 11.0);
        assert_relative_eq!(-v1.dot_product(&-v2), 11.0);
        assert_relative_eq!(v1.dot_product(&v2), v2.dot_product(&v1));
        let x1 = v1 * 3.1;
        let x2 = v2 * 6.1;
        assert_relative_eq!(x1.dot_product(&x2), ((3.1 * 6.1) * v1.dot_product(&v2)));
        assert_relative_eq!(v1.dot_product(&(v2 + v3)), v1.dot_product(&v2) + v1.dot_product(&v3));
    }

    #[test]
    fn vector2dfloat_partialeq() {
        let v1 = Vector2DFloat::new((1.0, 2.0), (2.0, 3.0));
        let v2 = Vector2DFloat::new((2.0, 3.0), (3.0, 4.0));
        let v3 = Vector2DFloat::new_bound((10.0, 10.0));
        // equivalence defined on magnitude and direction comparison only
        assert_eq!(v1.coord, v2.coord);
        assert_ne!(v1.begin, v2.begin);
        assert_ne!(v1.end(), v2.end());
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn vector2dfloat_unary_neg_operator() {
        let v1 = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        let v2 = Vector2DFloat::new((3.0, 4.0), (1.0, 2.0));
        let v3 = Vector2DFloat::new((-1.0, -2.0), (-3.0, -4.0));
        let v4 = Vector2DFloat::new((-3.0, -4.0), (-1.0, -2.0));
        assert_relative_eq!(v1.coord.x, 2.0);
        assert_relative_eq!(v1.coord.y, 2.0);
        assert_relative_eq!(-v1.coord.x, -2.0);
        assert_relative_eq!(-v1.coord.y, -2.0);
        assert_eq!(-v1, -v1);
        assert_eq!(-v1, v2);
        assert_eq!(-v3, -v3);
        assert_eq!(-v3, v4);
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
        let v1_v2 = v1 + v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in addition.
        let v4 = Vector2DFloat::new((1.3, 1.3), (2.3, 2.3));
        let v5 = Vector2DFloat::new((2.3, 2.3), (3.3, 3.3));
        let v4_v5 = v4 + v5;
        let v5_v4 = v5 + v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
        assert_eq!(v4_v5, v5_v4);
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
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in subtraction
        let v4 = Vector2DFloat::new((1.3, 1.3), (2.3, 2.3));
        let v5 = Vector2DFloat::new((2.3, 2.3), (3.3, 3.3));
        let v4_v5 = v4 - v5;
        let v5_v4 = v5 - v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
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
        let v3 = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        let v3_2 = v3 * 2.0_f64;
        assert_eq!(v3_2.begin, v3.begin);
        assert_ne!(v3_2.end(), v3.end());
        assert_ne!(v3_2, v3);
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
        let v3 = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        let v3_2 = v3 * 2_i64;
        assert_eq!(v3_2.begin, v3.begin);
        assert_ne!(v3_2.end(), v3.end());
        assert_ne!(v3_2, v3);
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
        let x1: Vector2DFloat = vi1.into();
        assert_eq!(x1, vf1);
        let vf5 = Vector2DFloat::new((1.0, 2.0), (3.0, 4.0));
        let vi5 = Vector2DInt::new((1, 2), (3, 4));
        let x2 = Vector2DFloat::from(vi5);
        assert_eq!(x2, vf5);
        assert_relative_eq!(x2.begin.x, 1.0);
        assert_relative_eq!(x2.begin.y, 2.0);
        assert_relative_eq!(x2.end().x, 3.0);
        assert_relative_eq!(x2.end().y, 4.0);
    }
}
