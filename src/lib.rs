mod f;
mod matrix;

use std::f32::consts::E;

use f::F;
use matrix::{
    create_matrix, matrix_add, matrix_map, matrix_subtract, matrix_multiply, Matrix,
};

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

struct Cost {
    multiplication_matrix_cache: Vec<Matrix>,
}

impl Cost {
    fn new(layers: &[usize]) -> Self {
        Cost {
            multiplication_matrix_cache: layers
                .iter()
                .skip(1)
                .map(|layer| create_matrix(1, *layer, || 0.0))
                .collect::<Vec<_>>(),
        }
    }

    fn cost<const D: usize, const DI: usize, const DO: usize>(
        &mut self,
        data: &Data<D, DI, DO>,
        model: &Model,
    ) -> F {
        let mut total_cost = 0.0;

        for (inputs, outputs) in data {
            let input = vec![Vec::from(*inputs)];
            let mut prev = &input;

            for (layer_params, mut cache) in model.iter().zip(&mut self.multiplication_matrix_cache)
            {
                matrix_multiply(prev, &layer_params.weights, &mut cache);
                matrix_add(&mut cache, &layer_params.biases);
                matrix_map(&mut cache, sigmoid);

                prev = cache;
            }

            total_cost += (outputs.last().unwrap() - prev[0][0]).powi(2); // TODO: handle multiple outputs
        }

        total_cost / data.len() as F
    }
}

fn get_direction<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    eps: F,
    rate: F,
    model: &mut Model,
    cost: &mut Cost,
) -> Model {
    let current_cost = cost.cost(data, model);

    let mut direction = model.clone();

    for i in 0..model.len() {
        for row_index in 0..model[i].weights.len() {
            for cell_index in 0..model[i].weights[row_index].len() {
                let temp_weight = model[i].weights[row_index][cell_index];
                model[i].weights[row_index][cell_index] = temp_weight + eps;
                direction[i].weights[row_index][cell_index] =
                    ((cost.cost(data, &model) - current_cost) / eps) * rate;
                model[i].weights[row_index][cell_index] = temp_weight;
            }
        }

        for row_index in 0..model[i].biases.len() {
            for cell_index in 0..model[i].biases[row_index].len() {
                let temp_bias = model[i].biases[row_index][cell_index];
                model[i].biases[row_index][cell_index] = temp_bias + eps;
                direction[i].biases[row_index][cell_index] =
                    ((cost.cost(data, &model) - current_cost) / eps) * rate;
                model[i].biases[row_index][cell_index] = temp_bias;
            }
        }
    }

    direction
}

fn initialize_model<G: FnMut() -> F>(layers: &[usize], mut generete_parameter: G) -> Model {
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

    let mut model = initialize_model(&layers, generate_parameter);
    
    let mut cost = Cost::new(&layers);

    for _ in 0..nr_of_iterations {
        let direction = get_direction(&data, eps, rate, &mut model, &mut cost);

        for (layer, layer_direction) in model.iter_mut().zip(direction) {
            layer.weights = matrix_subtract(&layer.weights, &layer_direction.weights);
            layer.biases = matrix_subtract(&layer.biases, &layer_direction.biases);
        }
    }

    model
}
