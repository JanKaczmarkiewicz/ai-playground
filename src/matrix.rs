use crate::fns::sigmoid;

#[derive(Debug)]
pub struct Matrix {
    pub vec: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl Matrix {
    pub fn from_generator<G: FnMut() -> f64>(rows: usize, columns: usize, mut gen: G) -> Self {
        Self {
            vec: (0..rows * columns).map(|_| gen()).collect(),
            rows,
            columns,
        }
    }

    fn get_cell_mut(&mut self, row: usize, column: usize) -> &mut f64 {
        &mut self.vec[row * self.columns + column]
    }

    fn get_cell(&self, row: usize, column: usize) -> f64 {
        self.vec[row * self.columns + column]
    }

    fn row_iter(&self, row: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.columns).map(move |column| self.get_cell(row, column))
    }

    fn column_iter(&self, column: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.rows).map(move |row| self.get_cell(row, column))
    }

    // multiplies `self` by `other` and writes result to `out`
    // note if `out` has more rows or columns they are not overwriten (this is a feature since in `out` bias x's are stored)
    // after multiplication the result is mapped via activation not linear fn
    pub fn feed_forward(&self, other: &Self, out: &mut Self) {
        let m1 = self;
        let m2 = other;

        assert_eq!(m1.columns, m2.rows);

        for row in 0..m1.rows {
            for column in 0..m2.columns {
                *out.get_cell_mut(row, column) = sigmoid(
                    m1.row_iter(row)
                        .zip(m2.column_iter(row))
                        .map(|(l, r)| l * r)
                        .sum(),
                )
            }
        }
    }
}
