#[allow(unused_imports)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cgeo::types::vector::Vector2DFloat;

fn vector_magnitude(v: Vector2DFloat) -> f64 {
    v.magnitude()
}

fn criterion_benchmark(c: &mut Criterion) {
    let v = Vector2DFloat::new_bound((5.43678, 6.623423));
    c.bench_function("vec_magnitude_float", |b| b.iter(|| vector_magnitude(v)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
