use super::coordinate::{
    CartesianCoordinateType, F2DCoordinate, I2DCoordinate, PolarCoordinate, PolarCoordinateType,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PointType {
    Cartesian,
    Polar,
    BezierOnCurve,
    BezierOffCurve,
    Other,
    NotSpecified,
}

#[derive(Copy, Clone, Debug)]
pub struct PointGen {}

impl PointGen {
    pub fn cartesian_i64(x: i64, y: i64) -> CartesianPoint<I2DCoordinate> {
        CartesianPoint { coord: I2DCoordinate::new(x, y), typ: PointType::Cartesian }
    }

    pub fn cartesian_f64(x: f64, y: f64) -> CartesianPoint<F2DCoordinate> {
        CartesianPoint { coord: F2DCoordinate::new(x, y), typ: PointType::Cartesian }
    }

    pub fn polar(r: f64, theta: f64) -> PolarPoint<PolarCoordinate> {
        PolarPoint { coord: PolarCoordinate::new(r, theta), typ: PointType::Polar }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CartesianPoint<T>
where
    T: CartesianCoordinateType,
{
    pub coord: T,
    pub typ: PointType,
}

impl<T> CartesianPoint<T>
where
    T: CartesianCoordinateType,
{
    pub fn new(coord: T) -> Self {
        Self { coord, typ: PointType::Cartesian }
    }

    pub fn type_of(&self) -> PointType {
        self.typ
    }

    pub fn x(&self) -> T::CoordNumberType {
        self.coord.get_x()
    }

    pub fn y(&self) -> T::CoordNumberType {
        self.coord.get_y()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PolarPoint<T>
where
    T: PolarCoordinateType,
{
    pub coord: T,
    pub typ: PointType,
}

impl<T> PolarPoint<T>
where
    T: PolarCoordinateType,
{
    pub fn new(coord: T) -> Self {
        Self { coord, typ: PointType::Polar }
    }

    pub fn type_of(&self) -> PointType {
        self.typ
    }

    pub fn degrees(&self) -> f64 {
        self.coord.get_degrees()
    }

    pub fn radians(&self) -> f64 {
        self.coord.get_radians()
    }

    pub fn radius(&self) -> f64 {
        self.coord.get_radius()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::coordinate::{
        CartesianCoordValType, F2DCoordinate, I2DCoordinate, PolarCoordValType, PolarCoordinate,
    };
    use approx::assert_relative_eq;
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn cartesianpoint_instantiation_new() {
        let pt1 = CartesianPoint::new(I2DCoordinate::new(1, 2));
        let pt2 = CartesianPoint::new(F2DCoordinate::new(1.0, 2.0));
        assert_eq!(pt1.coord.x, 1);
        assert_eq!(pt1.coord.y, 2);
        assert_eq!(pt1.coord.type_of(), CartesianCoordValType::I64);
        assert_eq!(pt1.type_of(), PointType::Cartesian);

        assert_relative_eq!(pt2.coord.x, 1.0);
        assert_relative_eq!(pt2.coord.y, 2.0);
        assert_eq!(pt2.coord.type_of(), CartesianCoordValType::F64);
        assert_eq!(pt2.type_of(), PointType::Cartesian);
    }

    #[test]
    fn cartesianpoint_instantiation_pointgen_builder() {
        let pt1 = PointGen::cartesian_i64(1, 2);
        let pt2 = PointGen::cartesian_f64(1.0, 2.0);
        assert_eq!(pt1.coord.x, 1);
        assert_eq!(pt1.coord.y, 2);
        assert_eq!(pt1.coord.type_of(), CartesianCoordValType::I64);
        assert_eq!(pt1.type_of(), PointType::Cartesian);

        assert_relative_eq!(pt2.coord.x, 1.0);
        assert_relative_eq!(pt2.coord.y, 2.0);
        assert_eq!(pt2.coord.type_of(), CartesianCoordValType::F64);
        assert_eq!(pt2.type_of(), PointType::Cartesian);
    }

    #[test]
    fn cartesianpoint_getters() {
        let pt1 = PointGen::cartesian_i64(1, 2);
        let pt2 = PointGen::cartesian_f64(1.0, 2.0);
        assert_eq!(pt1.x(), 1);
        assert_eq!(pt1.y(), 2);
        assert_relative_eq!(pt2.x(), 1.0);
        assert_relative_eq!(pt2.y(), 2.0);
    }

    #[test]
    fn polarpoint_instantiation_new() {
        let pt = PolarPoint::new(PolarCoordinate::new(14.142135623730951, 0.7853981633974483));
        assert_relative_eq!(pt.coord.r, 14.142135623730951);
        assert_relative_eq!(pt.coord.theta, 0.7853981633974483);
        assert_eq!(pt.coord.type_of(), PolarCoordValType::Radians);
        assert_eq!(pt.type_of(), PointType::Polar);
    }

    #[test]
    fn polarpoint_instantiation_pointgen_builder() {
        let pt = PointGen::polar(14.142135623730951, 0.7853981633974483);
        assert_relative_eq!(pt.coord.r, 14.142135623730951);
        assert_relative_eq!(pt.coord.theta, 0.7853981633974483);
        assert_eq!(pt.coord.type_of(), PolarCoordValType::Radians);
        assert_eq!(pt.type_of(), PointType::Polar);
    }

    #[test]
    fn polarpoint_getters() {
        let pt = PolarPoint::new(PolarCoordinate::new(14.142135623730951, 0.7853981633974483));
        assert_relative_eq!(pt.degrees(), 45.0);
        assert_relative_eq!(pt.radius(), 14.142135623730951);
        assert_relative_eq!(pt.radians(), 0.7853981633974483);
    }
}
