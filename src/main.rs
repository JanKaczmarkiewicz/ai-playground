use aiplay::nn::NeuronNetwork;

extern crate sdl2;

fn main() {
    // // // w* = 0.5
    // // // f(x) = sigmoid(x0 * w0 + x1 * w1 + 1 * w2)
    // // //

    // // // C(X):
    // // // sum
    // // //   for x in X:
    // // //     > (f(x) - y)^2

    // // // C'(X)/w0 = sum of X (2 * (f(x) - y) * f(x) * (1 - f(x)) * x0)

    // // // a = sigmoid(1.5)

    // // // C'(X)/w0 =
    // // //     sum of (
    // // //       0,
    // // //       0,
    // // //       2 * (sigmoid(1.5) - 1) * sigmoid(1.5) * (1 - sigmoid(1.5)) * x0,
    // // //       2 * (sigmoid(1) - 1) * sigmoid(1) * (1 - sigmoid(1)) * x0,
    // // //     )

    // let data: [(&[f64], &[f64]); 4] = [
    //     (&[0.0, 0.0], &[0.0]),
    //     (&[0.0, 1.0], &[1.0]),
    //     (&[1.0, 0.0], &[1.0]),
    //     (&[1.0, 1.0], &[1.0]),
    // ];

    // let layers = [data[0].0.len(), data[0].1.len()];

    // let mut nn = NeuronNetwork::random(&layers);

    // nn.train(&data, 1000);

    let data: [(&[f64], &[f64]); 4] = [
        (&[0.0], &[0.0]),
        (&[0.1], &[0.2]),
        (&[0.2], &[0.4]),
        (&[0.3], &[0.6]),
    ];

    // w* = 0.5
    // f(x) = sigmoid(x0 * w0 + x1 * w1 + 1 * w2)
    //

    // C(X):
    // sum
    //   for x in X:
    //     > (f(x) - y)^2

    // C'(X)/w0 = sum of X (2 * (f(x) - y) * f(x) * (1 - f(x)) * x0)

    // a = sigmoid(1.5)

    // C'(X)/w0 =
    //     sum of (
    //       0,
    //       0,
    //       2 * (sigmoid(1.5) - 1) * sigmoid(1.5) * (1 - sigmoid(1.5)) * x0,
    //       2 * (sigmoid(1) - 1) * sigmoid(1) * (1 - sigmoid(1)) * x0,
    //     )

    let layers = [data[0].0.len(), data[0].1.len()];

    let mut nn = NeuronNetwork::random(&layers);

    nn.train(&data, 1000);

    // let mut data_batch = data
    //     .iter()
    //     .map(|(inputs, outputs)| (ColumnVec::from_slice(inputs), *outputs))
    //     .collect::<Vec<_>>();

    // for _ in 0..1000 {
    //     for (inputs, outputs) in &mut data_batch {
    //         nn.forward_pass(inputs);
    //         nn.backward_pass(inputs, outputs);
    //     }
    //     nn.flush_weights();

    //     let outputs_batch = data_batch
    //         .iter()
    //         .map(|(_, outputs)| *outputs)
    //         .collect::<Vec<_>>();

    //     println!("cost: {}", nn.cost(&outputs_batch))
    // }

    // for (inputs, _) in &mut data_batch {
    //     nn.forward_pass(inputs);

    //     println!("{inputs:?} {:?}", &nn.layers.last().unwrap().outputs)
    // }
}
