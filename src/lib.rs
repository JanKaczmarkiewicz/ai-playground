mod f;
mod matrix;

use std::f32::consts::E;

use f::F;
use matrix::{create_matrix, matrix_add, matrix_map, matrix_multiply, matrix_subtract, Matrix};

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

pub struct Cost {
    outputs_for_data_cache: Option<Vec<Vec<Matrix>>>,
}

impl Cost {
    pub fn new() -> Self {
        Cost {
            outputs_for_data_cache: None,
        }
    }

    pub fn cost<const D: usize, const DI: usize, const DO: usize>(
        &mut self,
        data: &Data<D, DI, DO>,
        model: &Model,
    ) -> F {
        self.outputs_for_data(data, model)
            .iter()
            .zip(data)
            .map(|(row, (_, output))| (output.last().unwrap() - row.last().unwrap()[0][0]).powi(2))
            .sum::<F>()
            / data.len() as F
    }

    fn outputs_for_data<const D: usize, const DI: usize, const DO: usize>(
        &mut self,
        data: &Data<D, DI, DO>,
        model: &Model,
    ) -> &mut Vec<Vec<Matrix>> {
        let out_cache = self.outputs_for_data_cache.get_or_insert_with(|| {
            data.iter()
                .map(|_| {
                    model
                        .iter()
                        .map(|layer| create_matrix(1, layer.weights.len(), || 0.0))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });

        data.iter()
            .zip(out_cache.iter_mut())
            .for_each(|((input, _output), row_cache)| {
                let mut cache_iter = row_cache.iter_mut();
                let input_cache = cache_iter.next().unwrap();
                input_cache[0].clear();
                input_cache[0].extend(input.iter());
                let mut prev = input_cache;

                for (layer_params, mut cache) in model.iter().zip(cache_iter) {
                    matrix_multiply(prev, &layer_params.weights, &mut cache);
                    matrix_add(&mut cache, &layer_params.biases);
                    matrix_map(&mut cache, sigmoid);

                    prev = cache;
                }
            });

        out_cache
    }
}

fn get_gradient_back_propagation<const D: usize, const DI: usize, const DO: usize>(
    data: &Data<D, DI, DO>,
    rate: F,
    model: &mut Model,
    cost: &mut Cost,
) -> Model {
    let current_cost = cost.cost(data, model);

    let mut direction = model.clone();

    let xes = cost.outputs_for_data(data, model);

    let c = |v| v * (1.0 - v);

    for layer_index in 0..direction.len() {
        for row_index in 0..direction[layer_index].weights.len() {
            for cell_index in 0..direction[layer_index].weights[row_index].len() {
                let cell_direction: f32 = data
                    .iter()
                    .enumerate()
                    .map(|(data_index, (_, out))| -> F {
                        let recursive_part: F = {
                            let mut relevant_outputs =
                                xes[data_index][layer_index..].iter().map(|layer| &layer[0]);

                            let base_x = relevant_outputs.next().unwrap()[row_index];
                            let next_x = relevant_outputs
                                .next()
                                .map_or(1.0, |layer_output| c(layer_output[cell_index]));

                            let mut res = vec![base_x * next_x];

                            let base_x_str = format!("x[l{layer_index}, i{row_index}, v:{base_x}]");
                            let next_x_str =
                                format!("c(x[l{}, i{cell_index}, v:{base_x}])", layer_index + 1);

                            let mut res_str = vec![format!("{base_x_str} * {next_x_str}")];

                            for (xes_index, xes) in relevant_outputs.enumerate() {
                                res_str = xes
                                    .iter()
                                    .enumerate()
                                    .flat_map(|(i, x)| {
                                        res_str.iter().map(move |res_path| {
                                            let next_x_str = format!(
                                                "c(x[l{}, i{i}, v:{x}])",
                                                layer_index + xes_index + 2,
                                            );
                                            format!("{res_path} * {}", next_x_str)
                                        })
                                    })
                                    .collect::<Vec<_>>();

                                res = xes
                                    .iter()
                                    .flat_map(|x| res.iter().map(|res_path| res_path * c(*x)))
                                    .collect::<Vec<_>>();
                            }

                            let res = res.iter().sum();

                            println!(
                                "w[l{layer_index}][r{row_index}][c{cell_index}] = {res} = {}",
                                res_str.join(" + ")
                            );

                            res
                        };

                        2.0 * (out[0] - current_cost).powi(2) * recursive_part
                    })
                    .sum::<F>();

                direction[layer_index].weights[row_index][cell_index] = cell_direction * rate;
            }
        }

        // optimalization idea: first compute biases then based on that weights
        for cell_index in 0..direction[layer_index].biases[0].len() {
            let cell_direction: f32 = data
                .iter()
                .enumerate()
                .map(|(data_index, (_, out))| -> F {
                    let recursive_part: F = {
                        let mut relevant_outputs = xes[data_index][layer_index..]
                            .iter()
                            .map(|layer| &layer[0])
                            .skip(1); // TODO: check

                        let next_x = relevant_outputs
                            .next()
                            .map_or(1.0, |layer_output| c(layer_output[cell_index]));

                        let mut res = vec![c(next_x)];

                        for xes in relevant_outputs {
                            res = xes
                                .iter()
                                .flat_map(|x| res.iter().map(|res_path| res_path * c(*x)))
                                .collect::<Vec<_>>();
                        }

                        res.iter().sum()
                    };

                    2.0 * (out[0] - current_cost).powi(2) * recursive_part
                })
                .sum::<F>();

            direction[layer_index].biases[0][cell_index] = cell_direction * rate;
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

    let mut cost = Cost::new();

    for _ in 0..nr_of_iterations {
        let gradient = get_gradient_back_propagation(&data, rate, &mut model, &mut cost);
        println!("{:#?}", gradient);
        for (layer, layer_gradient) in model.iter_mut().zip(gradient) {
            layer.weights = matrix_subtract(&layer.weights, &layer_gradient.weights);
            layer.biases = matrix_subtract(&layer.biases, &layer_gradient.biases);
        }

        println!("{:?}", Cost::new().cost(&data, &model));
    }

    model
}

#[cfg(test)]
mod tests {
    use crate::{get_gradient_back_propagation, initialize_model, Cost};

    #[test]
    fn get_gradient_back_propagation_test() {
        let arch = [4, 3, 2, 1];
        let mut cost = Cost::new();
        let mut model = initialize_model(&arch, || 1.0);
        println!(
            "{:?}",
            get_gradient_back_propagation(
                &[
                    ([1.0, 0.2, 0.5, 0.7], [1.0]),
                    // ([0.5, 0.0, 0.3, 0.2], [0.5])
                ],
                0.1,
                &mut model,
                &mut cost,
            )
        );
    }
}
