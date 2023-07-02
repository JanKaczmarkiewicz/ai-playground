use crate::f::F;

pub type Matrix = Vec<Vec<F>>;

pub fn create_matrix<G: FnMut() -> F>(rows: usize, columns: usize, mut gen_value: G) -> Matrix {
    Vec::from_iter((0..rows).map(|_| Vec::from_iter((0..columns).map(|_| gen_value()))))
}

pub fn matrix_multiply(m1: &Matrix, m2: &Matrix, out: &mut Matrix) {
    {
        let m1_nr_of_columns = m1[0].len();
        let m2_nr_of_rows = m2.len();

        assert_eq!(m1_nr_of_columns, m2_nr_of_rows);
    }

    let m1_nr_of_rows = m1.len();
    let m2_nr_of_columns = m2[0].len();
    assert_eq!(m1_nr_of_rows, out.len());
    assert_eq!(m2_nr_of_columns, out[0].len());

    for i in 0..m1_nr_of_rows {
        for j in 0..m2_nr_of_columns {
            out[i][j] = m1[i]
                .iter()
                .enumerate()
                .map(|(index, m1_cell)| m1_cell * m2[index][j])
                .sum()
        }
    }
}

pub fn matrix_add(m1: &mut Matrix, m2: &Matrix) {
    let m1_nr_of_rows = m1.len();
    let m2_nr_of_rows = m2.len();
    let m1_nr_of_columns = m1[0].len();
    let m2_nr_of_columns = m2[0].len();

    assert_eq!(m1_nr_of_rows, m2_nr_of_rows);
    assert_eq!(m1_nr_of_columns, m2_nr_of_columns);

    for i in 0..m1_nr_of_rows {
        for j in 0..m1_nr_of_columns {
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

pub fn matrix_subtract(m1: &Matrix, m2: &Matrix) -> Matrix { // TODO: optimize
    let m1_nr_of_rows = m1.len();
    let m2_nr_of_rows = m2.len();
    let m1_nr_of_columns = m1[0].len();
    let m2_nr_of_columns = m2[0].len();

    assert_eq!(m1_nr_of_rows, m2_nr_of_rows);
    assert_eq!(m1_nr_of_columns, m2_nr_of_columns);

    let mut out = create_matrix(m1_nr_of_rows, m1_nr_of_columns, || 0.0);

    for i in 0..m1_nr_of_rows {
        for j in 0..m1_nr_of_columns {
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
        let mut out = vec![vec![0.0], vec![0.0]];
        matrix_multiply(
            &vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            &vec![vec![1.0], vec![2.0]],
            &mut out,
        );

        assert_eq!(out, vec![vec![5.0], vec![11.0]]);
    }

    #[test]
    fn matrix_addition_simple() {
        let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        matrix_add(&mut a, &vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert_eq!(a, vec![vec![2.0, 4.0], vec![5.0, 8.0]],);
    }
}
