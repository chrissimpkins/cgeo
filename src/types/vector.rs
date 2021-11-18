//! Vector types.

use std::cmp::PartialOrd;
use std::ops::{Add, Mul, Neg, Sub};

use approx::{relative_eq, RelativeEq};
use num::{Float, Num};

use crate::error::VectorError;
use crate::types::coordinate::{F2DCoordinate, I2DCoordinate};

trait VectorType {
    // marker trait for now
}

/// A generic 2D vector type.
#[derive(Copy, Clone, Debug)]
pub struct Vector<N: Num + Copy> {
    pub begin: (N, N),
    pub coord: (N, N),
}

impl<N> Vector<N>
where
    N: Num + Copy,
{
    // public methods
    pub fn new(begin: (N, N), end: (N, N)) -> Self {
        Self { begin, coord: (end.0 - begin.0, end.1 - begin.1) }
    }

    pub fn new_bound(end: (N, N)) -> Self {
        Self { begin: (N::zero(), N::zero()), coord: (end.0, end.1) }
    }

    pub fn new_zero() -> Self {
        Self::new_bound((N::zero(), N::zero()))
    }

    pub fn x(&self) -> N {
        self.coord.0
    }

    pub fn y(&self) -> N {
        self.coord.1
    }

    pub fn coord(&self) -> (N, N) {
        self.coord
    }

    pub fn begin(&self) -> (N, N) {
        self.begin
    }

    pub fn end(&self) -> (N, N) {
        (self.coord.0 + self.begin.0, self.coord.1 + self.begin.1)
    }

    /// Euclidean vector magnitude.
    pub fn magnitude(&self) -> N
    where
        N: Float,
    {
        // uses the dot product approach for performance
        self.dot_product(self).sqrt()
    }

    pub fn normalize(&self) -> Vector<N>
    where
        N: Float,
    {
        let x = self.coord.0 / self.magnitude();
        let y = self.coord.1 / self.magnitude();
        Vector::new_bound((x, y))
    }

    pub fn dot_product(&self, other: &Vector<N>) -> N {
        (self.coord.0 * other.coord.0) + (self.coord.1 * other.coord.1)
    }

    pub fn exterior_product(&self, other: &Vector<N>) -> N {
        (self.coord.0 * other.coord.1) - (self.coord.1 * other.coord.0)
    }

    pub fn is_perpendicular(&self, other: &Vector<N>) -> Result<bool, VectorError> {
        if (self.coord.0.is_zero() && self.coord.1.is_zero())
            || (other.coord.0.is_zero() && other.coord.1.is_zero())
        {
            return Err(VectorError::ValueError(
                "Invalid use of a zero vector in the is_perpendicular calculation".into(),
            ));
        }

        Ok(self.dot_product(other).is_zero())
    }

    pub fn is_left_of(&self, other: &Vector<N>) -> bool
    where
        N: Num + Copy + PartialOrd,
    {
        self.exterior_product(other) < N::zero()
    }

    pub fn is_right_of(&self, other: &Vector<N>) -> bool
    where
        N: Num + Copy + PartialOrd,
    {
        self.exterior_product(other) > N::zero()
    }

    pub fn is_collinear(&self, other: &Vector<N>) -> bool {
        self.exterior_product(other).is_zero()
    }

    /// Calculate the angle between two [`Vector`] in radians.
    pub fn angle(&self, other: &Vector<N>) -> N
    where
        N: Float,
    {
        self.normalize().dot_product(&other.normalize()).acos()
    }

    // private methods
    fn partial_eq_int(&self, other: &Vector<N>) -> bool {
        (self.coord.0 == other.coord.0) && (self.coord.1 == other.coord.1)
    }

    fn partial_eq_float(&self, other: &Vector<N>) -> bool
    where
        N: Num + Copy + RelativeEq<N>,
    {
        relative_eq!(self.coord.0, other.coord.0) && relative_eq!(self.coord.1, other.coord.1)
    }
}

/// [`Vector`] unary negation with the `-` operator
impl<N> Neg for Vector<N>
where
    N: Num + Copy,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new((self.end().0, self.end().1), (self.begin().0, self.begin().1))
    }
}

/// [`Vector`] addition with the `+` operator
impl<N> Add for Vector<N>
where
    N: Num + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { begin: self.begin, coord: (self.coord.0 + rhs.coord.0, self.coord.1 + rhs.coord.1) }
    }
}

/// [`Vector`] subtraction with the `-` operator
impl<N> Sub for Vector<N>
where
    N: Num + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { begin: self.begin, coord: (self.coord.0 - rhs.coord.0, self.coord.1 - rhs.coord.1) }
    }
}

/// [`Vector`] scalar multiplication using the `*` operator
impl<N> Mul<N> for Vector<N>
where
    N: Num + Copy,
{
    type Output = Self;

    fn mul(self, rhs: N) -> Self::Output {
        Self { begin: self.begin, coord: (rhs * self.coord.0, rhs * self.coord.1) }
    }
}

/// Partial equivalence is defined based on vector magnitude and
/// direction comparisons only.  Vectors with different begin and
/// end coordinates are defined as equivalent when they have the same
/// magnitude and direction but different locations in coordinate space.
impl PartialEq<Vector<usize>> for Vector<usize> {
    fn eq(&self, other: &Vector<usize>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<u8>> for Vector<u8> {
    fn eq(&self, other: &Vector<u8>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<u16>> for Vector<u16> {
    fn eq(&self, other: &Vector<u16>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<u32>> for Vector<u32> {
    fn eq(&self, other: &Vector<u32>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<u64>> for Vector<u64> {
    fn eq(&self, other: &Vector<u64>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<u128>> for Vector<u128> {
    fn eq(&self, other: &Vector<u128>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<isize>> for Vector<isize> {
    fn eq(&self, other: &Vector<isize>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<i8>> for Vector<i8> {
    fn eq(&self, other: &Vector<i8>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<i16>> for Vector<i16> {
    fn eq(&self, other: &Vector<i16>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<i32>> for Vector<i32> {
    fn eq(&self, other: &Vector<i32>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<i64>> for Vector<i64> {
    fn eq(&self, other: &Vector<i64>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<i128>> for Vector<i128> {
    fn eq(&self, other: &Vector<i128>) -> bool {
        self.partial_eq_int(other)
    }
}

impl PartialEq<Vector<f32>> for Vector<f32> {
    fn eq(&self, other: &Vector<f32>) -> bool {
        self.partial_eq_float(other)
    }
}

impl PartialEq<Vector<f64>> for Vector<f64> {
    fn eq(&self, other: &Vector<f64>) -> bool {
        self.partial_eq_float(other)
    }
}

/// A 2D vector struct with [`i64`] coordinates
#[derive(Copy, Clone, Debug)]
pub struct Vector2DInt {
    /// The vector start coordinates.
    pub begin: I2DCoordinate,
    /// The bound vector coordinates where a bound vector is defined as having
    /// a start (0, 0) origin coordinate.
    pub coord: I2DCoordinate,
}

impl VectorType for Vector2DInt {}

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

    pub fn new_zero() -> Self {
        Self::new_bound((0, 0))
    }

    /// Normalize the [`Vector2DInt`]to a [`Vector2DFloat`] unit vector.  
    /// Note that normalization changes the type.
    pub fn normalize(&self) -> Vector2DFloat {
        let x = self.coord.x as f64 / self.magnitude();
        let y = self.coord.y as f64 / self.magnitude();
        Vector2DFloat::new_bound((x, y))
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

    pub fn exterior_product(&self, other: &Vector2DInt) -> i64 {
        (self.coord.x * other.coord.y) - (self.coord.y * other.coord.x)
    }

    pub fn is_perpendicular(&self, other: &Vector2DInt) -> Result<bool, VectorError> {
        if (self.coord.x == 0 && self.coord.y == 0) || (other.coord.x == 0 && other.coord.y == 0) {
            return Err(VectorError::ValueError(
                "Invalid use of a zero vector in the Vector2DInt::is_perpendicular calculation"
                    .into(),
            ));
        }

        Ok(self.dot_product(other) == 0)
    }

    pub fn is_left_of(&self, other: &Vector2DInt) -> bool {
        self.exterior_product(other) < 0
    }

    pub fn is_right_of(&self, other: &Vector2DInt) -> bool {
        self.exterior_product(other) > 0
    }

    pub fn is_collinear(&self, other: &Vector2DInt) -> bool {
        self.exterior_product(other) == 0
    }

    /// Calculate the angle between two [`Vector2DInt`] in radians.
    pub fn angle(&self, other: &Vector2DInt) -> f64 {
        self.normalize().dot_product(&other.normalize()).acos()
    }

    /// Returns the clockwise normal vector of a [`Vector2DInt`].  This has
    /// a magnitude that is equivalent to the original [`Vector2DInt`] and a coordinate
    /// offset that is 90 degrees in the clockwise direction.
    pub fn cw_normal(&self) -> Self {
        Self {
            begin: I2DCoordinate::new(self.begin.y, -self.begin.x),
            coord: I2DCoordinate::new(self.coord.y, -self.coord.x),
        }
    }

    /// Returns the counter-clockwise normal vector of a [`Vector2DInt`].  This has
    /// a magnitude that is equivalent to the original [`Vector2DInt`] and a coordinate
    /// offset that is 90 degrees in the counter-clockwise direction.
    pub fn ccw_normal(&self) -> Self {
        Self {
            begin: I2DCoordinate::new(-self.begin.y, self.begin.x),
            coord: I2DCoordinate::new(-self.coord.y, self.coord.x),
        }
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

impl VectorType for Vector2DFloat {}

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

    pub fn new_zero() -> Self {
        Self::new_bound((0.0, 0.0))
    }

    /// Normalize the [`Vector2DFloat`] to a unit vector.
    pub fn normalize(&self) -> Vector2DFloat {
        let x = self.coord.x as f64 / self.magnitude();
        let y = self.coord.y as f64 / self.magnitude();
        Vector2DFloat::new_bound((x, y))
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

    pub fn exterior_product(&self, other: &Vector2DFloat) -> f64 {
        (self.coord.x * other.coord.y) - (self.coord.y * other.coord.x)
    }

    pub fn is_perpendicular(&self, other: &Vector2DFloat) -> Result<bool, VectorError> {
        if (relative_eq!(self.coord.x, 0.0) && relative_eq!(self.coord.y, 0.0))
            || (relative_eq!(other.coord.x, 0.0) && relative_eq!(other.coord.y, 0.0))
        {
            return Err(VectorError::ValueError(
                "Invalid use of a zero vector in the Vector2DFloat::is_perpendicular calculation"
                    .into(),
            ));
        }

        Ok(relative_eq!(self.dot_product(other), 0.0))
    }

    pub fn is_left_of(&self, other: &Vector2DFloat) -> bool {
        self.exterior_product(other) < 0.0
    }

    pub fn is_right_of(&self, other: &Vector2DFloat) -> bool {
        self.exterior_product(other) > 0.0
    }

    pub fn is_collinear(&self, other: &Vector2DFloat) -> bool {
        relative_eq!(self.exterior_product(other), 0.0)
    }

    /// Calculate the angle between two [`Vector2DFloat`] in radians.
    pub fn angle(&self, other: &Vector2DFloat) -> f64 {
        self.normalize().dot_product(&other.normalize()).acos()
    }

    /// Returns the clockwise normal vector of a [`Vector2DFloat`].  This has
    /// a magnitude that is equivalent to the original [`Vector2DFloat`] and a coordinate
    /// offset that is 90 degrees in the clockwise direction.
    pub fn cw_normal(&self) -> Self {
        Self {
            begin: F2DCoordinate::new(self.begin.y, -self.begin.x),
            coord: F2DCoordinate::new(self.coord.y, -self.coord.x),
        }
    }

    /// Returns the counter-clockwise normal vector of a [`Vector2DFloat`].  This has
    /// a magnitude that is equivalent to the original [`Vector2DFloat`] and a coordinate
    /// offset that is 90 degrees in the counter-clockwise direction.
    pub fn ccw_normal(&self) -> Self {
        Self {
            begin: F2DCoordinate::new(-self.begin.y, self.begin.x),
            coord: F2DCoordinate::new(-self.coord.y, self.coord.x),
        }
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

    // =====================
    // Vector
    // =====================
    #[test]
    fn vector_instantiation() {
        let v1 = Vector::new((1, 2), (3, 4));
        let v2 = Vector::new((1 as i64, 2 as i64), (3 as i64, 4 as i64));
        let v3 = Vector::new((1.0, 2.0), (3.0, 4.0));
        let v4 = Vector::new((1.0 as f32, 2.0 as f32), (3.0 as f32, 4.0 as f32));
        assert_eq!(v1.coord, (2 as i32, 2 as i32));
        assert_eq!(v1.begin, (1 as i32, 2 as i32));
        assert_eq!(v2.coord, (2 as i64, 2 as i64));
        assert_eq!(v2.begin, (1 as i64, 2 as i64));
        assert_relative_eq!(v3.coord.0, 2.0 as f64);
        assert_relative_eq!(v3.coord.1, 2.0 as f64);
        assert_relative_eq!(v3.begin.0, 1.0 as f64);
        assert_relative_eq!(v3.begin.1, 2.0 as f64);
        assert_relative_eq!(v4.coord.0, 2.0 as f32);
        assert_relative_eq!(v4.coord.1, 2.0 as f32);
        assert_relative_eq!(v4.begin.0, 1.0 as f32);
        assert_relative_eq!(v4.begin.1, 2.0 as f32);
    }

    #[test]
    fn vector_new_bound() {
        let v1 = Vector::new_bound((1, 2));
        let v2 = Vector::new_bound((1.0, 2.0));
        assert_eq!(v1.begin, (0, 0));
        assert_eq!(v1.coord, (1, 2));
        assert_relative_eq!(v2.begin.0, 0.0);
        assert_relative_eq!(v2.begin.1, 0.0);
        assert_relative_eq!(v2.coord.0, 1.0);
        assert_relative_eq!(v2.coord.1, 2.0);
    }

    #[test]
    fn vector_x() {
        let v1 = Vector::new((1, 2), (3, 6));
        let v2 = Vector::new((1.0, 2.0), (3.0, 6.0));
        assert_eq!(v1.x(), 2);
        assert_relative_eq!(v2.x(), 2.0);
        assert_relative_eq!(v2.x(), 2.0);
    }

    #[test]
    fn vector_y() {
        let v1 = Vector::new((1, 2), (3, 6));
        let v2 = Vector::new((1.0, 2.0), (3.0, 6.0));
        assert_eq!(v1.y(), 4);
        assert_relative_eq!(v2.y(), 4.0);
        assert_relative_eq!(v2.y(), 4.0);
    }

    #[test]
    fn vector_coord() {
        let v1 = Vector::new((1, 2), (3, 4));
        let v2 = Vector::new((1.0, 2.0), (3.0, 4.0));
        assert_eq!(v1.coord, (2, 2));
        assert_relative_eq!(v2.coord.0, 2.0);
        assert_relative_eq!(v2.coord.1, 2.0);
    }

    #[test]
    fn vector_begin() {
        let v1 = Vector::new((1, 2), (3, 4));
        let v2 = Vector::new((1.0, 2.0), (3.0, 4.0));
        assert_eq!(v1.begin, (1, 2));
        assert_relative_eq!(v2.begin.0, 1.0);
        assert_relative_eq!(v2.begin.1, 2.0);
    }

    #[test]
    fn vector_end() {
        let v1 = Vector::new((1, 2), (3, 4));
        let v2 = Vector::new((1.0, 2.0), (3.0, 4.0));
        assert_eq!(v1.end(), (3, 4));
        assert_relative_eq!(v2.end().0, 3.0);
        assert_relative_eq!(v2.end().1, 4.0);
    }

    #[test]
    fn vector_unary_neg_operator() {
        let v1 = Vector::new((1, 2), (3, 4));
        let v2 = Vector::new((3, 4), (1, 2));
        let v3 = Vector::new((-1, -2), (-3, -4));
        let v4 = Vector::new((-3, -4), (-1, -2));
        assert_eq!(v1.coord.0, 2);
        assert_eq!(v1.coord.1, 2);
        assert_eq!(-v1.coord.0, -2);
        assert_eq!(-v1.coord.1, -2);
        assert_eq!(-v1, -v1);
        assert_eq!(-v1, v2);
        assert_eq!(-v3, -v3);
        assert_eq!(-v3, v4);

        let v5 = Vector::new((1.0, 2.0), (3.0, 4.0));
        let v6 = Vector::new((3.0, 4.0), (1.0, 2.0));
        let v7 = Vector::new((-1.0, -2.0), (-3.0, -4.0));
        let v8 = Vector::new((-3.0, -4.0), (-1.0, -2.0));
        assert_relative_eq!(v5.coord.0, 2.0);
        assert_relative_eq!(v5.coord.1, 2.0);
        assert_relative_eq!(-v5.coord.0, -2.0);
        assert_relative_eq!(-v5.coord.1, -2.0);
        assert_eq!(-v5, -v5);
        assert_eq!(-v5, v6);
        assert_eq!(-v7, -v7);
        assert_eq!(-v7, v8);
    }

    #[test]
    fn vector_add_operator() {
        let v1 = Vector::new_bound((5, 5));
        let v2 = Vector::new_bound((4, 4));
        let v3 = Vector::new_bound((1, 1));
        let v_zero = Vector::new_bound((0, 0));
        assert_eq!(v1 + v2, Vector::new_bound((9, 9)));
        assert_eq!(v2 + v1, Vector::new_bound((9, 9)));
        assert_eq!(v1 + v2 + v3, Vector::new_bound((10, 10)));
        assert_eq!((v1 + v2) + v3, Vector::new_bound((10, 10)));
        assert_eq!(v1 + (v2 + v3), Vector::new_bound((10, 10)));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(v1 + (-v1), v_zero);
        let v1_v2 = v1 + v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in addition.
        let v4 = Vector::new((1, 1), (2, 2));
        let v5 = Vector::new((2, 2), (3, 3));
        let v4_v5 = v4 + v5;
        let v5_v4 = v5 + v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
        assert_eq!(v4_v5, v5_v4);

        let v1 = Vector::new_bound((5.1, 5.1));
        let v2 = Vector::new_bound((4.2, 4.2));
        let v3 = Vector::new_bound((1.3, 1.3));
        let v_zero = Vector::new_bound((0.0, 0.0));
        assert_eq!(v1 + v2, Vector::new_bound((5.1 + 4.2, 5.1 + 4.2)));
        assert_eq!(v2 + v1, Vector::new_bound((5.1 + 4.2, 5.1 + 4.2)));
        assert_eq!(v1 + v2 + v3, Vector::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!((v1 + v2) + v3, Vector::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!(v1 + (v2 + v3), Vector::new_bound((5.1 + 4.2 + 1.3, 5.1 + 4.2 + 1.3)));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(v1 + (-v1), v_zero);
        let v1_v2 = v1 + v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in addition.
        let v4 = Vector::new((1.3, 1.3), (2.3, 2.3));
        let v5 = Vector::new((2.3, 2.3), (3.3, 3.3));
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
    fn vector_sub_operator() {
        let v1 = Vector::new_bound((5, 5));
        let v2 = Vector::new_bound((4, 4));
        let v3 = Vector::new_bound((1, 1));
        let v_zero = Vector::new_bound((0, 0));
        assert_eq!(v1 - v2, Vector::new_bound((1, 1)));
        assert_eq!(v3 - v1, Vector::new_bound((-4, -4)));
        assert_eq!(v1 - v_zero, v1);
        assert_eq!(v1 - v2 - v3, v_zero);
        assert_eq!((v1 - v2) - v3, v_zero);
        assert_eq!(v1 - (v2 - v3), Vector::new_bound((2, 2)));
        assert_eq!(v1 - (v2 + v3), v_zero);
        assert_eq!(v2 - (-v3), v1);
        let v1_v2 = v1 - v2;
        assert_eq!(v1.begin, v1_v2.begin);
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in subtraction
        let v4 = Vector::new((1, 1), (2, 2));
        let v5 = Vector::new((2, 2), (3, 3));
        let v4_v5 = v4 - v5;
        let v5_v4 = v5 - v4;
        assert_eq!(v4.begin, v4_v5.begin);
        assert_ne!(v5.begin, v4_v5.begin);
        assert_ne!(v4.end(), v4_v5.end());
        assert_eq!(v5.begin, v5_v4.begin);
        assert_ne!(v4.begin, v5_v4.begin);
        assert_ne!(v5.end(), v5_v4.end());
        assert_eq!(v4_v5, v5_v4);

        let v1 = Vector::new_bound((5.3, 5.3));
        let v2 = Vector::new_bound((4.2, 4.2));
        let v3 = Vector::new_bound((1.1, 1.1));
        let v_zero = Vector::new_bound((0.0, 0.0));
        assert_eq!(v1 - v2, Vector::new_bound((5.3 - 4.2, 5.3 - 4.2)));
        assert_eq!(v3 - v1, Vector::new_bound((1.1 - 5.3, 1.1 - 5.3)));
        assert_eq!(v1 - v_zero, Vector::new_bound((5.3 - 0.0, 5.3 - 0.0)));
        assert_eq!(v1 - v2 - v3, Vector::new_bound((5.3 - 4.2 - 1.1, 5.3 - 4.2 - 1.1)));
        assert_eq!((v1 - v2) - v3, Vector::new_bound((5.3 - 4.2 - 1.1, 5.3 - 4.2 - 1.1)));
        assert_eq!(v1 - (v2 - v3), Vector::new_bound((5.3 - (4.2 - 1.1), 5.3 - (4.2 - 1.1))));
        assert_eq!(v1 - (v2 + v3), Vector::new_bound((5.3 - (4.2 + 1.1), 5.3 - (4.2 + 1.1))));
        assert_eq!(v2 - (-v3), Vector::new_bound((4.2 - (-1.1), 4.2 - (-1.1))));
        // vectors have same distance and magnitude but different
        // begin / end coordinates. The lhs begin takes precedence
        // in subtraction
        let v4 = Vector::new((1.3, 1.3), (2.3, 2.3));
        let v5 = Vector::new((2.3, 2.3), (3.3, 3.3));
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
    fn vector_mul_operator() {
        let v1 = Vector::new_bound((2, 3));
        let v2 = Vector::new_bound((-3, -2));
        let v_zero = Vector::new_bound((0, 0));
        assert_eq!(v1 * 2, Vector::new_bound((4, 6)));
        assert_eq!(v2 * 2, Vector::new_bound((-6, -4)));
        assert_eq!(v_zero * 10, Vector::new_bound((0, 0)));
        assert_eq!(v_zero * 10, v_zero);
        assert_eq!((v1 * 3) * 6, v1 * (6 * 3));
        assert_eq!(v1 * (6 + 3), (v1 * 6) + (v1 * 3));
        assert_eq!(v1 * (6 - 3), (v1 * 6) - (v1 * 3));
        assert_eq!((v1 + v2) * 6, (v1 * 6) + (v2 * 6));
        assert_eq!((v1 - v2) * 6, (v1 * 6) - (v2 * 6));
        let v3 = Vector::new((1, 2), (3, 4));
        let v3_2 = v3 * 2;
        assert_eq!(v3_2.begin, v3.begin);
        assert_ne!(v3_2.end(), v3.end());

        let v1 = Vector::new_bound((2.0, 3.0));
        let v2 = Vector::new_bound((-3.0, -2.0));
        let v_zero = Vector::new_bound((0.0, 0.0));
        assert_eq!(v1 * 2.0_f64, Vector::new_bound((2.0_f64 * 2.0_f64, 3.0_f64 * 2.0_f64)));
        assert_eq!(v2 * 2.0_f64, Vector::new_bound((-3.0_f64 * 2.0_f64, -2.0_f64 * 2.0_f64)));
        assert_eq!(v_zero * 10.0_f64, Vector::new_bound((0.0_f64 * 10.0_f64, 0.0_f64 * 10.0_f64)));
        assert_eq!((v1 * 3.0) * 6.0, v1 * (6.0 * 3.0));
        assert_eq!(v1 * (6.0 + 3.0), (v1 * 6.0) + (v1 * 3.0));
        assert_eq!(v1 * (6.0 - 3.0), (v1 * 6.0) - (v1 * 3.0));
        assert_eq!((v1 + v2) * 6.0, (v1 * 6.0) + (v2 * 6.0));
        assert_eq!((v1 - v2) * 6.0, (v1 * 6.0) - (v2 * 6.0));
        let v3 = Vector::new((1.0, 2.0), (3.0, 4.0));
        let v3_2 = v3 * 2.0_f64;
        assert_eq!(v3_2.begin, v3.begin);
        assert_ne!(v3_2.end(), v3.end());
        assert_ne!(v3_2, v3);
    }

    #[test]
    fn vector_magnitude() {
        let v1 = Vector::new((1.0, 2.0), (3.0, 4.0));
        let v2 = Vector::new((-1.0, -2.0), (-3.0, -4.0));
        assert_relative_eq!(v1.magnitude(), 2.8284271247461903);
        assert_relative_eq!(v2.magnitude(), 2.8284271247461903);
    }

    #[test]
    fn vector_normalize() {
        let v1 = Vector::new_bound((25.123, 30.456));
        let v2 = Vector::new_bound((-25.123, -30.456));
        assert_relative_eq!(v1.normalize().magnitude(), 1.0);
        assert_relative_eq!(v2.normalize().magnitude(), 1.0);
        assert_relative_eq!((v1.normalize() * v1.magnitude()).magnitude(), v1.magnitude());
        assert_relative_eq!((v2.normalize() * v2.magnitude()).magnitude(), v2.magnitude());
    }

    #[test]
    fn vector_dot_product() {
        let v1 = Vector::new_bound((1, 2));
        let v2 = Vector::new_bound((3, 4));
        let v3 = Vector::new_bound((5, 6));
        let v4 = Vector::new_bound((-3, -4));
        assert_eq!(v1.dot_product(&v2), 11);
        assert_eq!(v1.dot_product(&v4), -11);
        assert_eq!(-v1.dot_product(&-v2), 11);
        assert_eq!(v1.dot_product(&v2), v2.dot_product(&v1));
        let x1 = v1 * 3;
        let x2 = v2 * 6;
        assert_eq!(x1.dot_product(&x2), ((3 * 6) * v1.dot_product(&v2)));
        assert_eq!(v1.dot_product(&(v2 + v3)), v1.dot_product(&v2) + v1.dot_product(&v3));

        let v1 = Vector::new_bound((1.0, 2.0));
        let v2 = Vector::new_bound((3.0, 4.0));
        let v3 = Vector::new_bound((5.0, 6.0));
        assert_relative_eq!(v1.dot_product(&v2), 11.0);
        assert_relative_eq!(-v1.dot_product(&-v2), 11.0);
        assert_relative_eq!(v1.dot_product(&v2), v2.dot_product(&v1));
        let x1 = v1 * 3.1;
        let x2 = v2 * 6.1;
        assert_relative_eq!(x1.dot_product(&x2), ((3.1 * 6.1) * v1.dot_product(&v2)));
        assert_relative_eq!(v1.dot_product(&(v2 + v3)), v1.dot_product(&v2) + v1.dot_product(&v3));
    }

    #[test]
    fn vector_exterior_product() {
        let v1 = Vector::new_bound((1, 2));
        let v2 = Vector::new_bound((3, 4));
        let v3 = Vector::new_bound((-3, -4));
        let v4 = Vector::new_bound((5, 6));
        assert_eq!(v1.exterior_product(&v2), -2);
        assert_eq!(v1.exterior_product(&v3), 2);
        // nilpotent
        assert_eq!(v1.exterior_product(&v1), 0);
        // scalar association
        assert_eq!((v1 * 4).exterior_product(&(v2 * 6)), (v1.exterior_product(&v2) * (4 * 6)));
        // antisymmetric
        assert_eq!(v1.exterior_product(&v2), -v2.exterior_product(&v1));
        // additive distribution
        let ep1 = v1.exterior_product(&v2);
        let ep2 = v1.exterior_product(&v4);
        let vec_sum = v2 + v4;
        assert_eq!(v1.exterior_product(&vec_sum), (ep1 + ep2));

        let v1 = Vector::new_bound((1.0, 2.0));
        let v2 = Vector::new_bound((3.0, 4.0));
        let v3 = Vector::new_bound((-3.0, -4.0));
        let v4 = Vector::new_bound((5.0, 6.0));
        assert_relative_eq!(v1.exterior_product(&v2), -2.0);
        assert_relative_eq!(v1.exterior_product(&v3), 2.0);
        // nilpotent
        assert_relative_eq!(v1.exterior_product(&v1), 0.0);
        // scalar association
        assert_relative_eq!(
            (v1 * 4.0).exterior_product(&(v2 * 6.0)),
            (v1.exterior_product(&v2) * (4.0 * 6.0))
        );
        // antisymmetric
        assert_relative_eq!(v1.exterior_product(&v2), -v2.exterior_product(&v1));
        // additive distribution
        let ep1 = v1.exterior_product(&v2);
        let ep2 = v1.exterior_product(&v4);
        let vec_sum = v2 + v4;
        assert_relative_eq!(v1.exterior_product(&vec_sum), (ep1 + ep2));
    }

    #[test]
    fn vector_is_perpendicular() {
        let v1 = Vector::new_bound((0, 1));
        let v2 = Vector::new_bound((1, 0));
        let v3 = Vector::new_bound((3, 4));
        let v4 = Vector::new_bound((0, 0));
        assert!(v1.is_perpendicular(&v2).is_ok());
        assert!(v1.is_perpendicular(&v2).unwrap());
        assert!(v1.is_perpendicular(&v3).is_ok());
        assert!(!v1.is_perpendicular(&v3).unwrap());
        // calculation does not support use of zero vectors
        assert!(v1.is_perpendicular(&v4).is_err());
        assert!(v4.is_perpendicular(&v1).is_err());
        assert!(matches!(v1.is_perpendicular(&v4), Err(VectorError::ValueError(_))));
        assert!(matches!(v4.is_perpendicular(&v1), Err(VectorError::ValueError(_))));

        let v1 = Vector::new_bound((0.0, 1.0));
        let v2 = Vector::new_bound((1.0, 0.0));
        let v3 = Vector::new_bound((3.0, 4.0));
        let v4 = Vector::new_bound((0.0, 0.0));
        assert!(v1.is_perpendicular(&v2).is_ok());
        assert!(v1.is_perpendicular(&v2).unwrap());
        assert!(v1.is_perpendicular(&v3).is_ok());
        assert!(!v1.is_perpendicular(&v3).unwrap());
        // calculation does not support use of zero vectors
        assert!(v1.is_perpendicular(&v4).is_err());
        assert!(v4.is_perpendicular(&v1).is_err());
        assert!(matches!(v1.is_perpendicular(&v4), Err(VectorError::ValueError(_))));
        assert!(matches!(v4.is_perpendicular(&v1), Err(VectorError::ValueError(_))));
    }

    #[test]
    fn vector_is_left_of() {
        let v1 = Vector::new_bound((2, 2));
        let v2 = Vector::new_bound((2, 4));
        let v3 = Vector::new_bound((3, -2));
        let v4 = Vector::new_bound((1, 1));
        assert!(!v1.is_left_of(&v2));
        assert!(v1.is_left_of(&v3));
        // collinear should not return true
        assert!(!v1.is_left_of(&v4));

        let v1 = Vector::new_bound((2.0, 2.0));
        let v2 = Vector::new_bound((2.0, 4.0));
        let v3 = Vector::new_bound((3.0, -2.0));
        let v4 = Vector::new_bound((1.0, 1.0));
        assert!(!v1.is_left_of(&v2));
        assert!(v1.is_left_of(&v3));
        // collinear should not return true
        assert!(!v1.is_left_of(&v4));
    }

    #[test]
    fn vector_is_right_of() {
        let v1 = Vector::new_bound((2, 2));
        let v2 = Vector::new_bound((2, 4));
        let v3 = Vector::new_bound((3, -2));
        let v4 = Vector::new_bound((1, 1));
        assert!(v1.is_right_of(&v2));
        assert!(!v1.is_right_of(&v3));
        // collinear should not return true
        assert!(!v1.is_right_of(&v4));

        let v1 = Vector::new_bound((2.0, 2.0));
        let v2 = Vector::new_bound((2.0, 4.0));
        let v3 = Vector::new_bound((3.0, -2.0));
        let v4 = Vector::new_bound((1.0, 1.0));
        assert!(v1.is_right_of(&v2));
        assert!(!v1.is_right_of(&v3));
        // collinear should not return true
        assert!(!v1.is_right_of(&v4));
    }

    #[test]
    fn vector_is_collinear() {
        let v1 = Vector::new_bound((2, 2));
        let v2 = Vector::new_bound((2, 4));
        let v3 = Vector::new_bound((3, -2));
        let v4 = Vector::new_bound((1, 1));
        assert!(!v1.is_collinear(&v2));
        assert!(!v1.is_collinear(&v3));
        // collinear
        assert!(v1.is_collinear(&v4));

        let v1 = Vector::new_bound((2.0, 2.0));
        let v2 = Vector::new_bound((2.0, 4.0));
        let v3 = Vector::new_bound((3.0, -2.0));
        let v4 = Vector::new_bound((1.0, 1.0));
        assert!(!v1.is_collinear(&v2));
        assert!(!v1.is_collinear(&v3));
        // collinear
        assert!(v1.is_collinear(&v4));
    }

    #[test]
    fn vector_angle() {
        let v1 = Vector::new_bound((0.0, 10.0));
        let v2 = Vector::new_bound((10.0, 0.0));
        let v3 = Vector::new_bound((-100.0, 0.0));
        let v4 = Vector::new_bound((0.0, -25.0));
        assert_relative_eq!(v1.angle(&v2).to_degrees(), 90.0);
        assert!(v1.angle(&v2).is_sign_positive());
        assert_relative_eq!(v2.angle(&v1).to_degrees(), 90.0);
        assert!(v2.angle(&v1).is_sign_positive());
        assert_relative_eq!(v2.angle(&v3).to_degrees(), 180.0);
        assert!(v2.angle(&v3).is_sign_positive());
        assert_relative_eq!(v2.angle(&v4).to_degrees(), 90.0);
        assert!(v2.angle(&v4).is_sign_positive());
    }

    // =====================
    // Vector2DInt
    // =====================
    #[test]
    fn vector2dint_instantiation() {
        let v = Vector2DInt::new((1, 2), (3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(2, 2));
        assert_eq!(v.begin, I2DCoordinate::new(1, 2));

        let v = Vector2DInt::new_bound((3, 4));
        assert_eq!(v.coord, I2DCoordinate::new(3, 4));

        let v = Vector2DInt::new_zero();
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
    fn vector2dint_normalize() {
        let v1 = Vector2DInt::new_bound((25, 30));
        let v2 = Vector2DInt::new_bound((-25, -30));
        assert_relative_eq!(v1.normalize().magnitude(), 1.0);
        assert_relative_eq!(v2.normalize().magnitude(), 1.0);
        assert_relative_eq!((v1.normalize() * v1.magnitude()).magnitude(), v1.magnitude());
        assert_relative_eq!((v2.normalize() * v2.magnitude()).magnitude(), v2.magnitude());
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
    fn vector2dint_exterior_product() {
        let v1 = Vector2DInt::new_bound((1, 2));
        let v2 = Vector2DInt::new_bound((3, 4));
        let v3 = Vector2DInt::new_bound((-3, -4));
        let v4 = Vector2DInt::new_bound((5, 6));
        assert_eq!(v1.exterior_product(&v2), -2);
        assert_eq!(v1.exterior_product(&v3), 2);
        // nilpotent
        assert_eq!(v1.exterior_product(&v1), 0);
        // scalar association
        assert_eq!((v1 * 4).exterior_product(&(v2 * 6)), (v1.exterior_product(&v2) * (4 * 6)));
        // antisymmetric
        assert_eq!(v1.exterior_product(&v2), -v2.exterior_product(&v1));
        // additive distribution
        let ep1 = v1.exterior_product(&v2);
        let ep2 = v1.exterior_product(&v4);
        let vec_sum = v2 + v4;
        assert_eq!(v1.exterior_product(&vec_sum), (ep1 + ep2));
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
    fn vector2dint_is_perpendicular() {
        let v1 = Vector2DInt::new_bound((0, 1));
        let v2 = Vector2DInt::new_bound((1, 0));
        let v3 = Vector2DInt::new_bound((3, 4));
        let v4 = Vector2DInt::new_bound((0, 0));
        assert!(v1.is_perpendicular(&v2).is_ok());
        assert!(v1.is_perpendicular(&v2).unwrap());
        assert!(v1.is_perpendicular(&v3).is_ok());
        assert!(!v1.is_perpendicular(&v3).unwrap());
        // calculation does not support use of zero vectors
        assert!(v1.is_perpendicular(&v4).is_err());
        assert!(v4.is_perpendicular(&v1).is_err());
        assert!(matches!(v1.is_perpendicular(&v4), Err(VectorError::ValueError(_))));
        assert!(matches!(v4.is_perpendicular(&v1), Err(VectorError::ValueError(_))));
    }

    #[test]
    fn vector2dint_is_left_of() {
        let v1 = Vector2DInt::new_bound((2, 2));
        let v2 = Vector2DInt::new_bound((2, 4));
        let v3 = Vector2DInt::new_bound((3, -2));
        let v4 = Vector2DInt::new_bound((1, 1));
        assert!(!v1.is_left_of(&v2));
        assert!(v1.is_left_of(&v3));
        // collinear should not return true
        assert!(!v1.is_left_of(&v4));
    }

    #[test]
    fn vector2dint_is_right_of() {
        let v1 = Vector2DInt::new_bound((2, 2));
        let v2 = Vector2DInt::new_bound((2, 4));
        let v3 = Vector2DInt::new_bound((3, -2));
        let v4 = Vector2DInt::new_bound((1, 1));
        assert!(v1.is_right_of(&v2));
        assert!(!v1.is_right_of(&v3));
        // collinear should not return true
        assert!(!v1.is_right_of(&v4));
    }

    #[test]
    fn vector2dint_is_collinear() {
        let v1 = Vector2DInt::new_bound((2, 2));
        let v2 = Vector2DInt::new_bound((2, 4));
        let v3 = Vector2DInt::new_bound((3, -2));
        let v4 = Vector2DInt::new_bound((1, 1));
        assert!(!v1.is_collinear(&v2));
        assert!(!v1.is_collinear(&v3));
        // collinear
        assert!(v1.is_collinear(&v4));
    }

    #[test]
    fn vector2dint_angle() {
        let v1 = Vector2DInt::new_bound((0, 10));
        let v2 = Vector2DInt::new_bound((10, 0));
        let v3 = Vector2DInt::new_bound((-100, 0));
        let v4 = Vector2DInt::new_bound((0, -25));
        assert_relative_eq!(v1.angle(&v2).to_degrees(), 90.0);
        assert!(v1.angle(&v2).is_sign_positive());
        assert_relative_eq!(v2.angle(&v1).to_degrees(), 90.0);
        assert!(v2.angle(&v1).is_sign_positive());
        assert_relative_eq!(v2.angle(&v3).to_degrees(), 180.0);
        assert!(v2.angle(&v3).is_sign_positive());
        assert_relative_eq!(v2.angle(&v4).to_degrees(), 90.0);
        assert!(v2.angle(&v4).is_sign_positive());
    }

    #[test]
    fn vector2dint_ccw_normal() {
        let v1 = Vector2DInt::new_bound((10, 0));
        let v2 = Vector2DInt::new_bound((0, 10));
        let v3 = Vector2DInt::new_bound((1, 25));
        let v4 = Vector2DInt::new_bound((-25, 1));
        let v5 = Vector2DInt::new((-1, -2), (4, 5));
        let v6 = Vector2DInt::new((2, -1), (-5, 4));
        assert_eq!(v1.ccw_normal(), v2);
        assert_relative_eq!(v1.ccw_normal().angle(&v1).to_degrees(), 90.0);
        // preserves magnitude
        assert_relative_eq!(v1.ccw_normal().magnitude(), v1.magnitude());
        // dot product is zero
        assert_eq!(v1.ccw_normal().dot_product(&v1), 0);
        // scalar association
        assert_eq!((v1 * 6).ccw_normal(), v1.ccw_normal() * 6);
        // linear
        assert_eq!((v1 * 6 + v1 * 8).ccw_normal(), (v1 * 6).ccw_normal() + (v1 * 8).ccw_normal());
        // anti-potent
        assert_eq!(v1.ccw_normal().ccw_normal(), -v1);

        assert_eq!(v3.ccw_normal(), v4);
        assert_relative_eq!(v3.ccw_normal().angle(&v3).to_degrees(), 90.0);
        assert_relative_eq!(v3.ccw_normal().magnitude(), v3.magnitude());
        assert_relative_eq!(v5.magnitude(), v6.magnitude());

        assert_eq!(v5.ccw_normal(), v6);
        assert_relative_eq!(v5.ccw_normal().angle(&v5).to_degrees(), 90.0);
        assert_relative_eq!(v5.ccw_normal().magnitude(), v5.magnitude());
        // tests of begin and end coordinate locations in the ccw normal
        assert_eq!(v5.ccw_normal().begin.x, 2);
        assert_eq!(v5.ccw_normal().begin.y, -1);
        assert_eq!(v5.ccw_normal().end().x, -5);
        assert_eq!(v5.ccw_normal().end().y, 4);
    }

    #[test]
    fn vector2dint_cw_normal() {
        let v1 = Vector2DInt::new_bound((0, 10));
        let v2 = Vector2DInt::new_bound((10, 0));
        let v3 = Vector2DInt::new_bound((-25, 1));
        let v4 = Vector2DInt::new_bound((1, 25));
        let v5 = Vector2DInt::new((2, -1), (-5, 4));
        let v6 = Vector2DInt::new((-1, -2), (4, 5));

        assert_eq!(v1.cw_normal(), v2);
        assert_relative_eq!(v1.cw_normal().angle(&v1).to_degrees(), 90.0);
        // preserves magnitude
        assert_relative_eq!(v1.cw_normal().magnitude(), v1.magnitude());
        // dot product is zero
        assert_eq!(v1.cw_normal().dot_product(&v1), 0);
        // scalar association
        assert_eq!((v1 * 6).cw_normal(), v1.cw_normal() * 6);
        // linear
        assert_eq!((v1 * 6 + v1 * 8).cw_normal(), (v1 * 6).cw_normal() + (v1 * 8).cw_normal());
        // anti-potent
        assert_eq!(v1.cw_normal().cw_normal(), -v1);

        assert_eq!(v3.cw_normal(), v4);
        assert_relative_eq!(v3.cw_normal().angle(&v3).to_degrees(), 90.0);
        assert_relative_eq!(v3.cw_normal().magnitude(), v3.magnitude());
        assert_relative_eq!(v5.magnitude(), v6.magnitude());

        assert_eq!(v5.cw_normal(), v6);
        assert_relative_eq!(v5.cw_normal().angle(&v5).to_degrees(), 90.0);
        assert_relative_eq!(v5.cw_normal().magnitude(), v5.magnitude());
        // tests of begin and end coordinate locations in the ccw normal
        assert_eq!(v5.cw_normal().begin.x, -1);
        assert_eq!(v5.cw_normal().begin.y, -2);
        assert_eq!(v5.cw_normal().end().x, 4);
        assert_eq!(v5.cw_normal().end().y, 5);
    }

    #[test]
    fn vector2dfloat_instantiation() {
        let v = Vector2DFloat::new((1.0, 2.0), (3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(2.123, 2.321));
        assert_eq!(v.begin, F2DCoordinate::new(1.0, 2.0));

        let v = Vector2DFloat::new_bound((3.123, 4.321));
        assert_eq!(v.coord, F2DCoordinate::new(3.123, 4.321));

        let v = Vector2DInt::new_zero();
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
    fn vector2dfloat_normalize() {
        let v1 = Vector2DFloat::new_bound((25.123, 30.456));
        let v2 = Vector2DFloat::new_bound((-25.123, -30.456));
        assert_relative_eq!(v1.normalize().magnitude(), 1.0);
        assert_relative_eq!(v2.normalize().magnitude(), 1.0);
        assert_relative_eq!((v1.normalize() * v1.magnitude()).magnitude(), v1.magnitude());
        assert_relative_eq!((v2.normalize() * v2.magnitude()).magnitude(), v2.magnitude());
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
    fn vector2dfloat_exterior_product() {
        let v1 = Vector2DFloat::new_bound((1.0, 2.0));
        let v2 = Vector2DFloat::new_bound((3.0, 4.0));
        let v3 = Vector2DFloat::new_bound((-3.0, -4.0));
        let v4 = Vector2DFloat::new_bound((5.0, 6.0));
        assert_eq!(v1.exterior_product(&v2), -2.0);
        assert_eq!(v1.exterior_product(&v3), 2.0);
        // nilpotent
        assert_eq!(v1.exterior_product(&v1), 0.0);
        // scalar association
        assert_eq!(
            (v1 * 4.0).exterior_product(&(v2 * 6.0)),
            (v1.exterior_product(&v2) * (4.0 * 6.0))
        );
        // antisymmetric
        assert_eq!(v1.exterior_product(&v2), -v2.exterior_product(&v1));
        // additive distribution
        let ep1 = v1.exterior_product(&v2);
        let ep2 = v1.exterior_product(&v4);
        let vec_sum = v2 + v4;
        assert_eq!(v1.exterior_product(&vec_sum), (ep1 + ep2));
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

    #[test]
    fn vector2dfloat_is_perpendicular() {
        let v1 = Vector2DFloat::new_bound((0.0, 1.0));
        let v2 = Vector2DFloat::new_bound((1.0, 0.0));
        let v3 = Vector2DFloat::new_bound((3.0, 4.0));
        let v4 = Vector2DFloat::new_bound((0.0, 0.0));
        assert!(v1.is_perpendicular(&v2).is_ok());
        assert!(v1.is_perpendicular(&v2).unwrap());
        assert!(v1.is_perpendicular(&v3).is_ok());
        assert!(!v1.is_perpendicular(&v3).unwrap());
        // calculation does not support use of zero vectors
        assert!(v1.is_perpendicular(&v4).is_err());
        assert!(v4.is_perpendicular(&v1).is_err());
        assert!(matches!(v1.is_perpendicular(&v4), Err(VectorError::ValueError(_))));
        assert!(matches!(v4.is_perpendicular(&v1), Err(VectorError::ValueError(_))));
    }

    #[test]
    fn vector2dfloat_is_left_of() {
        let v1 = Vector2DFloat::new_bound((2.0, 2.0));
        let v2 = Vector2DFloat::new_bound((2.0, 4.0));
        let v3 = Vector2DFloat::new_bound((3.0, -2.0));
        let v4 = Vector2DFloat::new_bound((1.0, 1.0));
        assert!(!v1.is_left_of(&v2));
        assert!(v1.is_left_of(&v3));
        // collinear should not return true
        assert!(!v1.is_left_of(&v4));
    }

    #[test]
    fn vector2dfloat_is_right_of() {
        let v1 = Vector2DFloat::new_bound((2.0, 2.0));
        let v2 = Vector2DFloat::new_bound((2.0, 4.0));
        let v3 = Vector2DFloat::new_bound((3.0, -2.0));
        let v4 = Vector2DFloat::new_bound((1.0, 1.0));
        assert!(v1.is_right_of(&v2));
        assert!(!v1.is_right_of(&v3));
        // collinear should not return true
        assert!(!v1.is_right_of(&v4));
    }

    #[test]
    fn vector2dfloat_is_collinear() {
        let v1 = Vector2DFloat::new_bound((2.0, 2.0));
        let v2 = Vector2DFloat::new_bound((2.0, 4.0));
        let v3 = Vector2DFloat::new_bound((3.0, -2.0));
        let v4 = Vector2DFloat::new_bound((1.0, 1.0));
        assert!(!v1.is_collinear(&v2));
        assert!(!v1.is_collinear(&v3));
        // collinear
        assert!(v1.is_collinear(&v4));
    }

    #[test]
    fn vector2dfloat_angle() {
        let v1 = Vector2DFloat::new_bound((0.0, 10.0));
        let v2 = Vector2DFloat::new_bound((10.0, 0.0));
        let v3 = Vector2DFloat::new_bound((-100.0, 0.0));
        let v4 = Vector2DFloat::new_bound((0.0, -25.0));
        assert_relative_eq!(v1.angle(&v2).to_degrees(), 90.0);
        assert!(v1.angle(&v2).is_sign_positive());
        assert_relative_eq!(v2.angle(&v1).to_degrees(), 90.0);
        assert!(v2.angle(&v1).is_sign_positive());
        assert_relative_eq!(v2.angle(&v3).to_degrees(), 180.0);
        assert!(v2.angle(&v3).is_sign_positive());
        assert_relative_eq!(v2.angle(&v4).to_degrees(), 90.0);
        assert!(v2.angle(&v4).is_sign_positive());
    }

    #[test]
    fn vector2dfloat_ccw_normal() {
        let v1 = Vector2DFloat::new_bound((10.0, 0.0));
        let v2 = Vector2DFloat::new_bound((0.0, 10.0));
        let v3 = Vector2DFloat::new_bound((1.0, 25.0));
        let v4 = Vector2DFloat::new_bound((-25.0, 1.0));
        let v5 = Vector2DFloat::new((-1.0, -2.0), (4.0, 5.0));
        let v6 = Vector2DFloat::new((2.0, -1.0), (-5.0, 4.0));
        assert_eq!(v1.ccw_normal(), v2);
        assert_relative_eq!(v1.ccw_normal().angle(&v1).to_degrees(), 90.0);
        // preserves magnitude
        assert_relative_eq!(v1.ccw_normal().magnitude(), v1.magnitude());
        // dot product is zero
        assert_relative_eq!(v1.ccw_normal().dot_product(&v1), 0.0);
        // scalar association
        assert_eq!((v1 * 6.0).ccw_normal(), v1.ccw_normal() * 6.0);
        // linear
        assert_eq!((v1 * 6 + v1 * 8).ccw_normal(), (v1 * 6).ccw_normal() + (v1 * 8).ccw_normal());
        // anti-potent
        assert_eq!(v1.ccw_normal().ccw_normal(), -v1);

        assert_eq!(v3.ccw_normal(), v4);
        assert_relative_eq!(v3.ccw_normal().angle(&v3).to_degrees(), 90.0);
        assert_relative_eq!(v3.ccw_normal().magnitude(), v3.magnitude());
        assert_relative_eq!(v5.magnitude(), v6.magnitude());

        assert_eq!(v5.ccw_normal(), v6);
        assert_relative_eq!(v5.ccw_normal().angle(&v5).to_degrees(), 90.0);
        assert_relative_eq!(v5.ccw_normal().magnitude(), v5.magnitude());
        assert_relative_eq!(v5.ccw_normal().begin.x, 2.0);
        assert_relative_eq!(v5.ccw_normal().begin.y, -1.0);
        assert_relative_eq!(v5.ccw_normal().end().x, -5.0);
        assert_relative_eq!(v5.ccw_normal().end().y, 4.0);
    }
}
