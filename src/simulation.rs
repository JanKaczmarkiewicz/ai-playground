use rand::Rng;

use crate::matrix::Matrix;

pub fn simulate() {
    let data: [(&[f64], &[f64]); 4] = [
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
    ];

    // initialize

    let layers = [data[0].0.len(), data[0].1.len()];

    let mut rng = rand::thread_rng();

    let mut weights = (0..layers.len() - 1)
        .map(|i| {
            Matrix::from_generator(
                layers[i] + 1, /* +1 row for bias weigths */
                layers[i + 1],
                || rng.gen_range(0.0..1.0),
            )
        })
        .collect::<Vec<_>>();

    let mut output_layers = (0..layers.len())
        .map(|i| {
            Matrix::from_generator(
                1,
                layers[i] + 1, /* +1 row for bias x multiplier (1) */
                || 1.0,
            )
        })
        .collect::<Vec<_>>();

    for _ in 0..1 {
        let mut cost = 0.0;

        let mut deltas = (1..layers.len())
            .map(|i| Matrix::from_generator(1, layers[i], || 0.0))
            .collect::<Vec<_>>();

        for (inputs, outputs) in data {
            // insert input
            output_layers[0]
                .vec
                .splice(..inputs.len(), inputs.iter().map(|x| *x));

            // for each layer multiply current layer by weights and write to next layer
            for l in 0..weights.len() {
                let (head, tail) = output_layers.split_at_mut(l + 1);

                let current_layer_output = &head[l];
                let next_layer_output = &mut tail[0];

                current_layer_output.feed_forward(&weights[l], next_layer_output);
            }

            // compute fresh delta

            // (no need to compute deltas over first layer since its input layer)
            let mut inner_deltas = (1..layers.len())
                .map(|i| Matrix::from_generator(1, layers[i], || 0.0))
                .collect::<Vec<_>>();

            // What is exatly delta?
            // Each weight derivetive can be computed by multiplying value
            // that weight is pointing to by corresponding neuron output
            // d/dC w = a * w

            // How to compute deltas?
            // define l index of last
            // for each ith neuron in layer:
            // last layer:
            //   d[l][i] = 2(a[l][i] - y[i]) * a[l][i] * (1 - a[l][i])
            for (i, delta) in inner_deltas.last_mut().unwrap().vec.iter_mut().enumerate() {
                let ali = output_layers[output_layers.len() - 1].get_cell(0, i);
                *delta = 2.0 * (ali - outputs[i]) * ali * (1.0 - ali);
            }
            // nth layer:
            //   d[n][i] = sum(d[n+1]) * a[n][i] * (1 - a[n][i])

            for l in (0..inner_deltas.len() - 1).rev() {
                let sum_of_prev_layer: f64 = inner_deltas[l + 1].column_iter(0).sum(); // sum(d[n+1]);

                for (i, delta) in inner_deltas[l].vec.iter_mut().enumerate() {
                    let ali = output_layers[output_layers.len() - 1].get_cell(0, i);
                    *delta = sum_of_prev_layer * ali * (1.0 - ali);
                }
            }

            println!("{output_layers:?}");

            // println!("{inner_deltas:?}");

            // sum inner delta with outer delta accumulator

            deltas
                .iter_mut()
                .zip(inner_deltas)
                .for_each(|(d, id)| d.add(&id));

            // I dont need to compute exact cost here I just need derivetives of weights in respect to Cost function

            cost += output_layers
                .last()
                .unwrap()
                .column_iter(0)
                .zip(outputs.iter())
                .map(|(a, y)| (a - *y).powi(2))
                .sum::<f64>()
                / outputs.len() as f64;
        }

        cost /= data.len() as f64;

        // println!("{cost}");

        // println!("{deltas:?}");

        // update each weight using deltas
        // after forward pass divide deltas by number of data to get avarage

        let learning_rate = 0.001;
        for ((weights, deltas), outputs) in weights
            .iter_mut()
            .zip(deltas)
            .zip(output_layers[..output_layers.len() - 2].iter())
        {
            for row in 0..weights.rows {
                for column in 0..weights.columns {
                    *weights.get_cell_mut(row, column) -=
                        learning_rate * deltas.get_cell(0, column) * outputs.get_cell(0, row)
                            / data.len() as f64
                }
            }
        }

        // compute cost
    }
}
