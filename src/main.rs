use aiplay::nn::NeuronNetwork;

fn main() {
    let data: [(&[f64], &[f64]); 4] = [
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
    ];

    let layers = [data[0].0.len(), data[0].1.len()];

    let mut nn = NeuronNetwork::random(&layers);

    nn.train(&data, 10);
}
