use rand::Rng;

use crate::matrix::Matrix;

pub fn simulate() {
    let data: [(&[f64], &[f64]); 4] = [
        (&[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0]),
        (&[0.0, 1.0, 1.0], &[0.0, 0.0, 1.0]),
        (&[1.0, 1.0, 1.0], &[0.0, 0.0, 1.0]),
        (&[1.0, 0.0, 1.0], &[0.0, 0.0, 1.0]),
    ];

    // initialize

    let layers = [data[0].0.len(), 2, 3, data[0].1.len()];

    let mut rng = rand::thread_rng();

    let weights = (0..layers.len() - 1)
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

    for (inputs, _outputs) in data {
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

        println!("{output_layers:?}");
    }
}
