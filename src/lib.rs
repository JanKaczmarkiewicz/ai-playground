mod f;
mod matrix;

use std::f32::consts::E;

use f::F;
use matrix::{matrix_add, matrix_map, matrix_multiply, matrix_subtract, Matrix};

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
    pub rate: F,
    pub nr_of_iterations: usize,
    pub generate_parameter: G,
    pub hidden_layers: [usize; H],
    pub data: Data<D, DI, DO>,
}

pub type Model = Vec<LayerParams>;

pub fn cost<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    model: &Model,
) -> F {
    data.iter()
        .map(|(input, output)| -> F {
            let mut acc = Matrix(vec![input.to_vec()]);

            for layer_params in model.iter() {
                let mut cache = Matrix::create(1, layer_params.weights.columns(), || 0.0);

                matrix_multiply(&acc, &layer_params.weights, &mut cache);
                matrix_add(&mut cache, &layer_params.biases);
                matrix_map(&mut cache, sigmoid);

                acc = cache;
            }

            acc.first()
                .unwrap()
                .iter()
                .zip(output)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<F>()
                / output.len() as F
        })
        .sum::<F>()
        / data.len() as F
}

fn get_gradient_back_propagation<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    rate: F,
    model: &mut Model,
) -> Model {
    todo!()
}

fn initialize_model<G: FnMut() -> F>(layers: &[usize], mut generete_parameter: G) -> Model {
    (0..layers.len() - 1)
        .map(|i| {
            let rows = layers[i];
            let colums = layers[i + 1];

            LayerParams {
                biases: Matrix::create(1, colums, &mut generete_parameter),
                weights: Matrix::create(rows, colums, &mut generete_parameter),
            }
        })
        .collect::<Vec<_>>()
}

pub fn train<G: FnMut() -> F, const H: usize, const D: usize, const DI: usize, const DO: usize>(
    TrainConfig {
        data,
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

    for _ in 0..nr_of_iterations {
        let gradient = get_gradient_back_propagation(&data, rate, &mut model);
        
        for (layer, layer_gradient) in model.iter_mut().zip(gradient) {
            layer.weights = matrix_subtract(&layer.weights, &layer_gradient.weights);
            layer.biases = matrix_subtract(&layer.biases, &layer_gradient.biases);
        }

        println!("{:?}", cost(&data, &model));
    }

    model
}

#[cfg(test)]
mod tests {
    use crate::{cost, get_gradient_back_propagation, initialize_model};

    #[test]
    fn get_gradient_back_propagation_test() {
        let arch = [4, 3, 2, 1];
        let mut model = initialize_model(&arch, || 1.0);
        println!(
            "{:?}",
            get_gradient_back_propagation(&[([1.0, 0.2, 0.5, 0.7], [1.0])], 0.1, &mut model,)
        );
    }

    #[test]
    fn cost_simple_test() {
        let arch = [2, 1];
        let model = initialize_model(&arch, || 0.5);

        assert_eq!(cost(&[([1.0, 0.0], [1.0])], &model), 0.07232948);
    }
}
