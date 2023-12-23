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
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        println!("Layer:");
        println!("weights rows: {}, cols: {}", input_size + 1, output_size);
        println!("outputs rows: {}, cols: {}", 1, input_size + 1);
        println!("deltas rows: {}, cols: {}", 1, output_size);

        Self {
            weights: Matrix::from_generator(
                input_size + 1, /* +1 row for bias weigths */
                output_size,
                || rng.gen_range(0.0..1.0),
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
        // for last layer algorythm will be different

        for (i, delta) in self.deltas.vec.iter_mut().enumerate() {
            let a = self.outputs.get_cell(0, i);

            *delta = a
                * (1.0 - a)
                * next_layer
                    .deltas
                    .column_iter(0)
                    .zip(next_layer.weights.column_iter(i))
                    .map(|(next_delta, weight)| next_delta * weight)
                    .sum::<f64>()
        }
    }

    fn flush_weights(&mut self) {
        let learning_rate = 0.001;

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

pub fn simulation() {
    let data: [(Matrix, &[f64]); 4] = [
        (Matrix::from_slice(&[0.0, 0.0, 1.0]), &[0.0]),
        (Matrix::from_slice(&[0.0, 1.0, 1.0]), &[1.0]),
        (Matrix::from_slice(&[1.0, 1.0, 1.0]), &[1.0]),
        (Matrix::from_slice(&[1.0, 0.0, 1.0]), &[1.0]),
    ];

    let layers_sizes = [data[0].0.columns - 1, 3, data[0].1.len()];

    let mut layers = (0..layers_sizes.len() - 1)
        .map(|i| Layer::new(layers_sizes[i], layers_sizes[i + 1]))
        .collect::<Vec<_>>();

    for _ in 0..3000 {
        let mut cost = 0.0;

        for (inputs, outputs) in &data {
            // TODO: move forward pass and backward pass to NN structure

            // forward pass
            layers
                .iter_mut()
                .fold(inputs, |previus_layer_output, current_layer| {
                    current_layer.forward(previus_layer_output);
                    &current_layer.outputs
                });

            // backward pass
            let mut layers_iter_from_end_to_start = layers.iter_mut().rev();
            let last_layer = layers_iter_from_end_to_start.next().unwrap();

            // last layer deltas
            last_layer
                .deltas
                .vec
                .iter_mut()
                .zip(&last_layer.outputs.vec)
                .zip(*outputs)
                .for_each(|((delta, y), a)| *delta = 2.0 * (a - y).abs());

            // other layers deltas
            layers_iter_from_end_to_start.fold(last_layer, |next_layer_deltas, current_layer| {
                current_layer.backward(next_layer_deltas);
                current_layer
            });

            layers
                .iter_mut()
                .fold(inputs, |previus_layer_output, current_layer| {
                    current_layer.update_gradient(previus_layer_output);
                    &current_layer.outputs
                });

            cost += layers
                .last()
                .unwrap()
                .outputs
                .column_iter(0)
                .zip(*outputs)
                .map(|(a, y)| (a - *y).powi(2))
                .sum::<f64>()
                / outputs.len() as f64;
        }

        cost /= data.len() as f64;
        println!("{cost}");
        layers.iter_mut().for_each(|layer| {
            layer.flush_weights();
            // println!("{:?}", layer.weights)
        });
    }
}
