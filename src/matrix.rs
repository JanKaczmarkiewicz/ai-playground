use std::ops::{Deref, DerefMut};

use crate::f::F;

type Vec2D = Vec<Vec<F>>;
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix(pub Vec2D);

impl Matrix {
    pub fn rows(&self) -> usize {
        self.0.len()
    }
    pub fn columns(&self) -> usize {
        self.0[0].len()
    }
    pub fn create<G: FnMut() -> F>(rows: usize, columns: usize, mut gen_value: G) -> Matrix {
        Matrix(Vec::from_iter(
            (0..rows).map(|_| Vec::from_iter((0..columns).map(|_| gen_value()))),
        ))
    }
}

impl Deref for Matrix {
    type Target = Vec2D;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec2D> for Matrix {
    fn from(vec2d: Vec2D) -> Matrix {
        Matrix(vec2d)
    }
}

pub fn matrix_multiply(m1: &Matrix, m2: &Matrix, out: &mut Matrix) {
    assert_eq!(m1.columns(), m2.rows());

    assert_eq!(m1.rows(), out.rows());
    assert_eq!(m2.columns(), m2.columns());

    for i in 0..out.rows() {
        for j in 0..out.columns() {
            out[i][j] = m1[i]
                .iter()
                .enumerate()
                .map(|(index, m1_cell)| m1_cell * m2[index][j])
                .sum()
        }
    }
}

pub fn matrix_add(m1: &mut Matrix, m2: &Matrix) {
    assert_eq!(m1.rows(), m2.rows());
    assert_eq!(m1.columns(), m2.columns());

    for i in 0..m1.columns() {
        for j in 0..m2.columns() {
            m1[i][j] += m2[i][j];
        }
    }
}

pub fn matrix_map(m: &mut Matrix, func: fn(F) -> F) {
    for row in m.iter_mut() {
        for cell in row {
            *cell = func(*cell)
        }
    }
}

pub fn matrix_subtract(m1: &Matrix, m2: &Matrix) -> Matrix {
    // TODO: optimize
    assert_eq!(m1.rows(), m2.rows());
    assert_eq!(m1.columns(), m2.columns());

    let mut out = Matrix::create(m1.rows(), m2.columns(), || 0.0);

    for i in 0..out.rows() {
        for j in 0..out.columns() {
            out[i][j] = m1[i][j] - m2[i][j];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::{matrix_add, matrix_multiply};

    #[test]
    fn matrix_multiply_simple() {
        let mut out = vec![vec![0.0], vec![0.0]].into();
        matrix_multiply(
            &vec![vec![1.0, 2.0], vec![3.0, 4.0]].into(),
            &vec![vec![1.0], vec![2.0]].into(),
            &mut out,
        );

        assert_eq!(out, vec![vec![5.0], vec![11.0]].into());
    }

    #[test]
    fn matrix_addition_simple() {
        let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0]].into();

        matrix_add(&mut a, &vec![vec![1.0, 2.0], vec![2.0, 4.0]].into());
        assert_eq!(a, vec![vec![2.0, 4.0], vec![5.0, 8.0]].into());
    }
}
