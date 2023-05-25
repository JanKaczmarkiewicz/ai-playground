use rand::Rng;

type F = f32;
type Matrix = Vec<Vec<F>>;

pub struct SimulationConfig {
    pub eps: F,
    pub rate: F,
    pub nr_of_iterations: usize,
    pub layers: &'static [usize],
    pub data: &'static [&'static [F]],
}

pub type Model = ();

fn random_float() -> F {
    let mut random = rand::thread_rng();
    random.gen_range(0.0..1.0)
}

fn create_matrix(rows: usize, columns: usize, gen_value: fn() -> F) -> Matrix {
    Vec::from_iter((0..rows).map(|_| Vec::from_iter((0..columns).map(|_| gen_value()))))
}

pub fn simulate(
    SimulationConfig {
        data,
        eps,
        rate,
        nr_of_iterations,
        layers,
    }: SimulationConfig,
) -> Model {
    let nr_of_inputs = data[0].len() - 1;

    let model = {
        let mut prev_layer = nr_of_inputs;
        layers
            .iter()
            .map(|layer| {
                let matrix = create_matrix(prev_layer, *layer, random_float);
                prev_layer = *layer;
                matrix
            })
            .collect::<Vec<_>>()
    };

    println!("{:?}", model);

    for _ in 0..nr_of_iterations {}
}

fn matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix {
    {
        let m1_nr_of_columns = m1[0].len();
        let m2_nr_of_rows = m2.len();

        assert_eq!(m1_nr_of_columns, m2_nr_of_rows);
    }
    let m1_nr_of_rows = m1.len();
    let m2_nr_of_columns = m2[0].len();

    let mut out = create_matrix(m1_nr_of_rows, m2_nr_of_columns, || 0.0);

    for i in 0..m1_nr_of_rows {
        for j in 0..m2_nr_of_columns {
            out[i][j] = m1[i]
                .iter()
                .enumerate()
                .map(|(index, m1_cell)| m1_cell * m2[index][j])
                .sum()
        }
    }

    out
}

fn matrix_addition(m1: Matrix, m2: Matrix) -> Matrix {
    let m1_nr_of_rows = m1.len();
    let m2_nr_of_rows = m2.len();
    let m1_nr_of_columns = m1[0].len();
    let m2_nr_of_columns = m2[0].len();

    assert_eq!(m1_nr_of_rows, m2_nr_of_rows);
    assert_eq!(m1_nr_of_columns, m2_nr_of_columns);

    let mut out = create_matrix(m1_nr_of_rows, m1_nr_of_columns, || 0.0);

    for i in 0..m1_nr_of_rows {
        for j in 0..m1_nr_of_columns {
            out[i][j] = m1[i][j] + m2[i][j];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::{matrix_multiply, matrix_addition};

    #[test]
    fn matrix_multiply_simple() {
        assert_eq!(
            matrix_multiply(
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![1.0], vec![2.0]],
            ),
            vec![vec![5.0], vec![11.0]]
        );
    }

    #[test]
    fn matrix_addition_simple() {
        assert_eq!(
            matrix_addition(
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![1.0, 2.0], vec![2.0, 4.0]],
            ),
            vec![vec![2.0, 4.0], vec![5.0, 8.0]]
        );
    }
}
