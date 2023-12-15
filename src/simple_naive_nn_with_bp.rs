use std::f64::consts::E;

use rand::Rng;

fn forward_pass(data: &[(&[f64], f64)], weights: &[f64]) -> Vec<f64> {
    data.iter()
        .map(|(inputs, _)| {
            sigmoid(
                inputs
                    .iter()
                    .zip(weights)
                    .map(|(input, weight)| input * weight)
                    .sum::<f64>(),
            )
        })
        .collect()
}

fn cost_fn(data: &[(&[f64], f64)], weights: &[f64]) -> f64 {
    forward_pass(data, weights)
        .iter()
        .zip(data.iter().map(|(_, output)| output))
        .map(|(output, expected_output)| (output - expected_output).powi(2))
        .sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn simulate() {
    let data: [(&[f64], f64); 4] = [
        (&[0.0, 0.0, 1.0], 0.0),
        (&[0.0, 1.0, 1.0], 0.0),
        (&[1.0, 1.0, 1.0], 1.0),
        (&[1.0, 0.0, 1.0], 0.0),
    ];

    // initialize
    let mut rng = rand::thread_rng();
    let mut weights = (0..data[0].0.len())
        .map(|_| rng.gen_range(0.0..1.0))
        .collect::<Vec<_>>();

    for _ in 0..100000 {
        // forward pass
        let cost: f64 = cost_fn(&data, &weights);

        println!("{cost}");

        let mut d_weights = weights.clone();
        for weight in d_weights.iter_mut() {
            *weight = 0.0;
        }

        forward_pass(&data, &weights)
            .into_iter()
            .zip(data)
            .for_each(|(a, (inputs, out))| {
                let common = 2.0 * (a - out) * a * (1.0 - a);
                for k in 0..d_weights.len() {
                    d_weights[k] += common * inputs[k];
                }
            });

        let rate = 0.1;
        for (weight, new_weight) in weights.iter_mut().zip(d_weights) {
            *weight -= rate * new_weight;
        }
    }
}
