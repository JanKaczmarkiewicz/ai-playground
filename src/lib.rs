use std::f32::consts::E;
type F = f32;
type Matrix = Vec<Vec<F>>;

#[derive(Debug, Clone)]
pub struct LayerParams {
    weights: Matrix,
    biases: Matrix,
}

fn sigmoid(x: F) -> F {
    1.0 / (1.0 + E.powf(-x))
}

pub type Data<const D: usize, const DI: usize, const DO: usize> = [([F; DI], [F; DO]); D];

pub struct TrainConfig<
    G: FnMut() -> F,
    const H: usize,
    const D: usize,
    const DI: usize,
    const DO: usize,
> {
    pub eps: F,
    pub rate: F,
    pub nr_of_iterations: usize,
    pub generate_parameter: G,
    pub hidden_layers: [usize; H],
    pub data: Data<D, DI, DO>,
}

pub type Model = Vec<LayerParams>;

fn create_matrix<G: FnMut() -> F>(rows: usize, columns: usize, mut gen_value: G) -> Matrix {
    Vec::from_iter((0..rows).map(|_| Vec::from_iter((0..columns).map(|_| gen_value()))))
}

pub fn cost<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    model: &Model,
) -> F {
    let mut total_cost = 0.0;

    for (inputs, outputs) in data {
        let input = vec![Vec::from(*inputs)];

        let output = model.iter().fold(input, |acc, curr| {
            let a = matrix_multiply(&acc, &curr.weights);
            matrix_map(matrix_addition(a, &curr.biases), sigmoid)
        });

        total_cost += (outputs.last().unwrap() - output[0][0]).powi(2);
    }

    total_cost / data.len() as F
}

fn get_direction<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    eps: F,
    rate: F,
    model: &mut Model,
) -> Model {
    let current_cost = cost(data, model);

    let mut direction = model.clone();

    for i in 0..model.len() {
        for row_index in 0..model[i].weights.len() {
            for cell_index in 0..model[i].weights[row_index].len() {
                let temp_weight = model[i].weights[row_index][cell_index];
                model[i].weights[row_index][cell_index] = temp_weight + eps;
                direction[i].weights[row_index][cell_index] =
                    ((cost(data, &model) - current_cost) / eps) * rate;
                model[i].weights[row_index][cell_index] = temp_weight;
            }
        }

        for row_index in 0..model[i].biases.len() {
            for cell_index in 0..model[i].biases[row_index].len() {
                let temp_bias = model[i].biases[row_index][cell_index];
                model[i].biases[row_index][cell_index] = temp_bias + eps;
                direction[i].biases[row_index][cell_index] =
                    ((cost(data, &model) - current_cost) / eps) * rate;
                model[i].biases[row_index][cell_index] = temp_bias;
            }
        }
    }

    direction
}

fn initialize_random_model<G: FnMut() -> F>(layers: &[usize], mut generete_parameter: G) -> Model {
    (0..layers.len() - 1)
        .map(|i| {
            let rows = layers[i];
            let colums = layers[i + 1];

            LayerParams {
                biases: create_matrix(1, colums, &mut generete_parameter),
                weights: create_matrix(rows, colums, &mut generete_parameter),
            }
        })
        .collect::<Vec<_>>()
}

pub fn train<G: FnMut() -> F, const H: usize, const D: usize, const DI: usize, const DO: usize>(
    TrainConfig {
        data,
        eps,
        rate,
        generate_parameter,
        nr_of_iterations,
        hidden_layers,
    }: TrainConfig<G, H, D, DI, DO>,
) -> Model {
    let layers = {
        let (input, output) = data[0];
        let mut layers = Vec::from(hidden_layers);
        layers.insert(0, input.len());
        layers.push(output.len());
        layers
    };

    let mut model = initialize_random_model(&layers, generate_parameter);

    let data = Box::new(data);

    for _ in 0..nr_of_iterations {
        let direction = get_direction(&data, eps, rate, &mut model);

        for (layer, layer_direction) in model.iter_mut().zip(direction) {
            layer.weights = matrix_subtraction(&layer.weights, &layer_direction.weights);
            layer.biases = matrix_subtraction(&layer.biases, &layer_direction.biases);
        }
    }

    model
}

fn matrix_multiply(m1: &Matrix, m2: &Matrix) -> Matrix {
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

fn matrix_addition(mut m1: Matrix, m2: &Matrix) -> Matrix {
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

    m1
}

fn matrix_map(mut m: Matrix, func: fn(F) -> F) -> Matrix {
    for row in m.iter_mut() {
        for cell in row {
            *cell = func(*cell)
        }
    }

    m
}

fn matrix_subtraction(m1: &Matrix, m2: &Matrix) -> Matrix {
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
    use crate::{matrix_addition, matrix_multiply};

    #[test]
    fn matrix_multiply_simple() {
        assert_eq!(
            matrix_multiply(
                &vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                &vec![vec![1.0], vec![2.0]],
            ),
            vec![vec![5.0], vec![11.0]]
        );
    }

    #[test]
    fn matrix_addition_simple() {
        assert_eq!(
            matrix_addition(
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                &vec![vec![1.0, 2.0], vec![2.0, 4.0]],
            ),
            vec![vec![2.0, 4.0], vec![5.0, 8.0]]
        );
    }
}
