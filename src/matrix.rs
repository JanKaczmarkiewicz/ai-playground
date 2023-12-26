use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct Matrix {
    pub vec: Vec<f64>,
    pub rows: usize,
    pub columns: usize,
}

pub struct ColumnVec(Vec<f64>);

impl Deref for ColumnVec {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ColumnVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ColumnVec {
    pub fn new(capacity: usize, v: f64) -> Self {
        ColumnVec((0..capacity).map(|_| v).collect())
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        ColumnVec(slice.to_vec())
    }

    pub fn mul(&self, other: &Matrix, out: &mut Self) {
        let m1 = &self.0;
        let m2 = other;
        let out = &mut out.0;

        assert_eq!(m1.len(), m2.rows);
        assert_eq!(out.len(), m2.columns);

        for (i, out) in out.iter_mut().enumerate() {
            *out = m1
                .iter()
                .zip(m2.column_iter(i))
                .map(|(l, r)| *l * r)
                .sum::<f64>()
        }
    }
}

impl Matrix {
    pub fn from_generator<G: FnMut() -> f64>(rows: usize, columns: usize, mut gen: G) -> Self {
        Self {
            vec: (0..rows * columns).map(|_| gen()).collect(),
            rows,
            columns,
        }
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        Self {
            vec: slice.to_vec(),
            rows: 1,
            columns: slice.len(),
        }
    }

    pub fn from_iter(slice: impl Iterator<Item = f64>) -> Self {
        let vec = slice.collect::<Vec<_>>();
        let columns = vec.len();
        Self {
            vec,
            rows: 1,
            columns,
        }
    }

    pub fn get_cell_mut(&mut self, row: usize, column: usize) -> &mut f64 {
        &mut self.vec[row * self.columns + column]
    }

    pub fn get_cell(&self, row: usize, column: usize) -> f64 {
        self.vec[row * self.columns + column]
    }

    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.columns).map(move |column| self.get_cell(row, column))
    }

    pub fn column_iter(&self, column: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.rows).map(move |row| self.get_cell(row, column))
    }

    pub fn map(&mut self, apply: fn(f64) -> f64) {
        self.vec.iter_mut().for_each(|x| *x = apply(*x))
    }

    pub fn add(&mut self, other: &Self) {
        let m1 = self;
        let m2 = other;

        assert_eq!(m1.columns, m2.columns);
        assert_eq!(m1.rows, m2.rows);

        m1.vec
            .iter_mut()
            .zip(m2.vec.iter())
            .for_each(|(m1_cell, m2_cell)| *m1_cell += m2_cell)
    }
}
