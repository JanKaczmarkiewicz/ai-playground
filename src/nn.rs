use rand::Rng;

use crate::matrix::Matrix;

struct Layer {
    weights: Matrix,
    weights_gradient: Matrix,
    gradient_updated_times: usize,
    outputs: Matrix,
    deltas: Matrix,
}

impl Layer {
    fn new<G: FnMut() -> f64>(input_size: usize, output_size: usize, gen: G) -> Self {
        Self {
            weights: Matrix::from_generator(
                input_size + 1, /* +1 row for bias weigths */
                output_size,
                gen,
            ),
            gradient_updated_times: 0,
            weights_gradient: Matrix::from_generator(input_size + 1, output_size, || 0.0),
            outputs: Matrix::from_generator(
                1,
                output_size + 1, /* +1 row for bias x multiplier (1) */
                || 1.0,
            ),
            deltas: Matrix::from_generator(1, output_size, || 0.0),
        }
    }

    fn forward(&mut self, previus_layer_output: &Matrix) {
        previus_layer_output.feed_forward(&self.weights, &mut self.outputs)
    }

    fn backward(&mut self, next_layer: &Layer) {
        self.deltas
            .vec
            .iter_mut()
            .zip(self.outputs.row_iter(0))
            .enumerate()
            .for_each(|(i, (delta, a))| {
                *delta = a
                    * (1.0 - a)
                    * next_layer
                        .deltas
                        .column_iter(0)
                        .zip(next_layer.weights.column_iter(i))
                        .map(|(next_delta, weight)| next_delta * weight)
                        .sum::<f64>()
            })
    }

    fn flush_weights(&mut self) {
        let learning_rate = 0.01;

        for row in 0..self.weights.rows {
            for column in 0..self.weights.columns {
                *self.weights.get_cell_mut(row, column) -= learning_rate
                    * self.weights_gradient.get_cell(row, column)
                    / self.gradient_updated_times as f64;
                *self.weights_gradient.get_cell_mut(row, column) = 0.0
            }
        }

        self.gradient_updated_times = 0;
    }

    fn update_gradient(&mut self, previus_layer_output: &Matrix) {
        self.gradient_updated_times += 1;
        for row in 0..self.weights_gradient.rows {
            for column in 0..self.weights_gradient.columns {
                *self.weights_gradient.get_cell_mut(row, column) +=
                    self.deltas.get_cell(0, column) * previus_layer_output.get_cell(0, row)
            }
        }
    }
}

pub struct NeuronNetwork {
    layers: Vec<Layer>,
}

impl NeuronNetwork {
    pub fn new<G: FnMut() -> f64>(layers_sizes: &[usize], mut gen: G) -> Self {
        Self {
            layers: (0..layers_sizes.len() - 1)
                .map(|i| Layer::new(layers_sizes[i], layers_sizes[i + 1], &mut gen))
                .collect::<Vec<_>>(),
        }
    }

    pub fn random(layers_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        Self::new(layers_sizes, || rng.gen_range(0.0..1.0))
    }

    fn forward_pass(&mut self, inputs: &Matrix) {
        self.layers
            .iter_mut()
            .fold(inputs, |previus_layer_output, current_layer| {
                current_layer.forward(previus_layer_output);
                &current_layer.outputs
            });
    }

    fn backward_pass(&mut self, inputs: &Matrix, outputs: &[f64]) {
        let last_layer = self.layers.last_mut().unwrap();

        // last layer deltas
        last_layer
            .deltas
            .vec
            .iter_mut()
            .zip(&last_layer.outputs.vec)
            .zip(outputs)
            .for_each(|((delta, a), y)| *delta = 2.0 * (a - y).abs() * a * (1.0 - a));

        // other layers deltas
        self.layers
            .iter_mut()
            .rev()
            .reduce(|next_layer, current_layer| {
                current_layer.backward(next_layer);
                current_layer
            });

        self.layers
            .iter_mut()
            .fold(inputs, |previus_layer_output, current_layer| {
                current_layer.update_gradient(previus_layer_output);
                &current_layer.outputs
            });
    }

    fn flush_weights(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.flush_weights());
    }

    fn cost<'a>(&self, outputs_batch: impl Iterator<Item = &'a [f64]>) -> f64 {
        let mut cost = 0.0;

        let mut n = 0;

        for outputs in outputs_batch {
            cost += self
                .layers
                .last()
                .unwrap()
                .outputs
                .row_iter(0)
                .zip(outputs)
                .map(|(a, y)| (a - *y).powi(2))
                .sum::<f64>()
                / outputs.len() as f64;

            n += 1;
        }

        cost /= n as f64;
        cost
    }

    pub fn train(&mut self, data_batch: &[(&[f64], &[f64])], n: usize) {
        let data_batch = data_batch
            .iter()
            .map(|(inputs, outputs)| {
                (
                    Matrix::from_iter(inputs.iter().chain(&[1.0]).map(|x| *x)),
                    *outputs,
                )
            })
            .collect::<Vec<_>>();

        for _ in 0..n {
            for (inputs, outputs) in &data_batch {
                self.forward_pass(inputs);

                self.backward_pass(inputs, outputs);
            }

            self.flush_weights();

            println!(
                "cost: {}",
                self.cost(data_batch.iter().map(|(_, outputs)| *outputs))
            )
        }
    }
}

#[test]
fn feature() {
    let layers = [2, 1, 1];

    let mut nn = NeuronNetwork::new(&layers, || 0.5);

    assert_eq!(nn.layers[0].weights.columns, 1);
    assert_eq!(nn.layers[0].weights.rows, 3);
    assert_eq!(nn.layers[0].outputs.columns, 2);
    assert_eq!(nn.layers[0].outputs.rows, 1);

    assert_eq!(nn.layers[1].weights.columns, 1);
    assert_eq!(nn.layers[1].weights.rows, 2);
    assert_eq!(nn.layers[1].outputs.columns, 2);
    assert_eq!(nn.layers[1].outputs.rows, 1);

    let inputs = Matrix::from_slice(&[0.0, 1.0, 1.0 /* for bias */]);
    let outputs = &[1.0];

    nn.forward_pass(&inputs);

    let al0 = sigmoid(0.0 * 0.5 + 1.0 * 0.5 + 1.0 * 0.5);
    assert_eq!(nn.layers[0].outputs.get_cell(0, 0), al0);
    assert_eq!(nn.layers[0].outputs.get_cell(0, 1), 1.0);

    let al1 = sigmoid(al0 * 0.5 + 1.0 * 0.5);

    assert_eq!(nn.layers[1].outputs.get_cell(0, 0), al1);
    assert_eq!(nn.layers[1].outputs.get_cell(0, 1), 1.0);

    // backward_pass + flush_weights is not doing anything meaning full
    nn.backward_pass(&inputs, outputs);
    let dl1a0 = 2.0 * (1.0 - al1) * al1 * (1.0 - al1);

    assert_eq!(nn.layers[1].deltas.get_cell(0, 0), dl1a0);
    assert_eq!(nn.layers[1].weights_gradient.get_cell(0, 0), dl1a0 * al0);

    let dl0a0 = (dl1a0 * 0.5 * al0 * (1.0 - al0)) as f32;
    assert_eq!(nn.layers[0].deltas.get_cell(0, 0) as f32, dl0a0);
    assert_eq!(
        nn.layers[0].weights_gradient.get_cell(0, 0) as f32,
        dl0a0 * 0.0 /*x0 */
    );

    // TODO what is happening when there are multiple neurons after hidden layer?
    // TODO is weight computing algorythm: `flush_weights` correct?

    nn.flush_weights();

    // assert_eq!(nn.layers[1].weights.get_cell(0, 0), 0.0);

    // assert_eq!(nn.layers[1].weights.get_cell(0, 0), 0.0);
    // assert_eq!(nn.layers[1].weights.get_cell(0, 1), 1.0);
}
