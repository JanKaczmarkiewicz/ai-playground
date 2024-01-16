use rand::Rng;

use crate::{
    fns::sigmoid,
    matrix::{ColumnVec, Matrix},
};

#[derive(Debug)]
struct Layer {
    weights: Matrix,
    delta_weights: Matrix,
    gradient_updated_times: usize,
    outputs: ColumnVec,
    delta_outputs: ColumnVec,
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
            delta_weights: Matrix::from_generator(input_size + 1, output_size, || 0.0),
            outputs: ColumnVec::new(output_size, 0.0),
            delta_outputs: ColumnVec::new(output_size, 0.0),
        }
    }

    fn forward(&mut self, previus_layer_output: &mut ColumnVec) {
        previus_layer_output.push(1.0);
        previus_layer_output.mul(&self.weights, &mut self.outputs);
        previus_layer_output.pop();

        self.outputs.iter_mut().for_each(|x| *x = sigmoid(*x));
    }

    fn compute_output_deltas(&mut self, next_layer: &Layer) {
        self.delta_outputs
            .iter_mut()
            .zip(&*self.outputs)
            .enumerate()
            .for_each(|(i, (delta, a))| {
                *delta = a
                    * (1.0 - a)
                    * next_layer
                        .delta_outputs
                        .iter()
                        .zip(next_layer.weights.column_iter(i))
                        .map(|(next_delta, weight)| next_delta * weight)
                        .sum::<f64>()
            })
    }

    fn flush_weights(&mut self) {
        let learning_rate = 0.1;

        for row in 0..self.weights.rows {
            for column in 0..self.weights.columns {
                *self.weights.get_cell_mut(row, column) -= learning_rate
                    * self.delta_weights.get_cell(row, column)
                    / self.gradient_updated_times as f64;
                *self.delta_weights.get_cell_mut(row, column) = 0.0
            }
        }
        self.gradient_updated_times = 0;
    }

    fn update_gradient(&mut self, previus_layer_output: &mut ColumnVec) {
        previus_layer_output.push(1.0);

        for row in 0..self.delta_weights.rows {
            for column in 0..self.delta_weights.columns {
                *self.delta_weights.get_cell_mut(row, column) +=
                    self.delta_outputs[column] * previus_layer_output[row];
            }
        }
        self.gradient_updated_times += 1;

        previus_layer_output.pop();
    }
}

#[derive(Debug)]
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

    pub fn get_model(&self) -> Vec<&Matrix> {
        self.layers.iter().map(|layer| &layer.weights).collect()
    }

    fn forward_pass(&mut self, inputs: &mut ColumnVec) {
        self.layers
            .iter_mut()
            .fold(inputs, |previus_layer_output, current_layer| {
                current_layer.forward(previus_layer_output);
                &mut current_layer.outputs
            });
    }

    fn backward_pass(&mut self, inputs: &mut ColumnVec, outputs: &[f64]) {
        let last_layer = self.layers.last_mut().unwrap();

        // last layer output deltas
        last_layer
            .delta_outputs
            .iter_mut()
            .zip(&*last_layer.outputs)
            .zip(outputs)
            .for_each(|((delta, a), y)| *delta = 2.0 * (a - y) * a * (1.0 - a));

        // other layers deltas
        self.layers
            .iter_mut()
            .rev()
            .reduce(|next_layer, current_layer| {
                current_layer.compute_output_deltas(next_layer);
                current_layer
            });

        self.layers
            .iter_mut()
            .fold(inputs, |previus_layer_output, current_layer| {
                current_layer.update_gradient(previus_layer_output);
                &mut current_layer.outputs
            });
    }

    fn flush_weights(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.flush_weights());
    }

    pub fn cost<'a>(&mut self, data_batch: &mut [(ColumnVec, &[f64])]) -> f64 {
        let mut cost = 0.0;

        let n_of_samples = data_batch.len();

        for (inputs, outputs) in data_batch {
            self.forward_pass(inputs);

            cost += self
                .layers
                .last()
                .unwrap()
                .outputs
                .iter()
                .zip(*outputs)
                .map(|(a, y)| (a - *y).powi(2))
                .sum::<f64>()
                / outputs.len() as f64;
        }

        cost /= n_of_samples as f64;
        cost
    }

    pub fn train_step(&mut self, data_batch: &mut [(ColumnVec, &[f64])]) {
        for (inputs, outputs) in data_batch.iter_mut() {
            self.forward_pass(inputs);
            self.backward_pass(inputs, outputs);
        }
        self.flush_weights();
    }
}

#[test]
fn feature() {
    let layers = [2, 1, 1];

    let mut nn = NeuronNetwork::new(&layers, || 0.5);

    assert_eq!(nn.layers[0].weights.columns, 1);
    assert_eq!(nn.layers[0].weights.rows, 3);
    assert_eq!(nn.layers[0].outputs.len(), 1);

    assert_eq!(nn.layers[1].weights.columns, 1);
    assert_eq!(nn.layers[1].weights.rows, 2);
    assert_eq!(nn.layers[1].outputs.len(), 1);

    let input = &[0.0, 1.0];

    let mut inputs = ColumnVec::from_slice(input);
    let outputs = &[1.0];

    nn.forward_pass(&mut inputs);

    let al0 = sigmoid(inputs[0] * 0.5 + inputs[1] * 0.5 + 1.0 * 0.5);
    assert_eq!(nn.layers[0].outputs[0], al0);

    let al1 = sigmoid(al0 * 0.5 + 1.0 * 0.5);

    assert_eq!(nn.layers[1].outputs[0], al1);

    nn.backward_pass(&mut inputs, outputs);
    let dl1a0 = 2.0 * (al1 - 1.0) * al1 * (1.0 - al1);

    assert_eq!(nn.layers[1].delta_outputs[0], dl1a0);
    assert_eq!(nn.layers[1].delta_weights.get_cell(0, 0), dl1a0 * al0);

    let dl0a0 = dl1a0 * 0.5 * al0 * (1.0 - al0);
    assert_eq!(nn.layers[0].delta_outputs[0] as f32, dl0a0 as f32);
    assert_eq!(
        nn.layers[0].delta_weights.get_cell(0, 0) as f32,
        (dl0a0 * input[0]) as f32
    );

    nn.flush_weights();

    assert_eq!(nn.layers[1].weights.get_cell(0, 0), 0.5 - dl1a0 * al0 * 0.1);
    assert_eq!(nn.layers[1].weights.get_cell(0, 1), 0.5 - dl1a0 * 0.1);

    assert_eq!(
        nn.layers[0].weights.get_cell(0, 0),
        0.5 - dl0a0 * input[0] * al0 * 0.1
    );
    assert_eq!(
        nn.layers[0].weights.get_cell(0, 1),
        0.5 - dl0a0 * input[1] * 0.1
    );
    assert_eq!(nn.layers[0].weights.get_cell(0, 2), 0.5 - dl0a0 * 0.1);
}

#[test]
fn simple() {
    let layers = [2, 1];

    let mut nn = NeuronNetwork::new(&layers, || 0.5);

    assert_eq!(nn.layers[0].weights.columns, 1);
    assert_eq!(nn.layers[0].weights.rows, 3);
    assert_eq!(nn.layers[0].outputs.len(), 1);

    let mut inputs = ColumnVec::from_slice(&[0.0, 1.0]);
    let outputs: &[f64] = [1.0].as_slice();

    nn.forward_pass(&mut inputs);

    let al0 = sigmoid(0.0 * 0.5 + 1.0 * 0.5 + 1.0 * 0.5);
    assert_eq!(nn.layers[0].outputs[0], al0);

    for _ in 0..1000 {
        nn.forward_pass(&mut inputs);
        nn.backward_pass(&mut inputs, outputs);
    }
    nn.backward_pass(&mut inputs, outputs);

    let dl0a0 = 2.0 * (sigmoid(1.0) - 1.0) * sigmoid(1.0) * (1.0 - sigmoid(1.0));
    assert_eq!(nn.layers[0].delta_outputs[0] as f32, dl0a0 as f32);

    nn.flush_weights();

    assert_eq!(
        nn.layers[0].weights.get_cell(0, 0),
        0.5 - dl0a0 * inputs[0] * 0.1
    );
    assert_eq!(
        nn.layers[0].weights.get_cell(0, 1) as f32,
        (0.5 - dl0a0 * inputs[1] * 0.1) as f32
    );
    assert_eq!(
        nn.layers[0].weights.get_cell(0, 2) as f32,
        (0.5 - dl0a0 * 0.1) as f32
    );
}

// TODO: test cost fn
