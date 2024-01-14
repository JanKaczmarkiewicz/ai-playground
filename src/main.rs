use aiplay::{matrix::ColumnVec, nn::NeuronNetwork, visualize::Visualizer};

fn main() {
    let mut data: [(ColumnVec, &[f64]); 4] = [
        (ColumnVec::from_slice(&[0.0, 0.0]), &[0.0]),
        (ColumnVec::from_slice(&[0.0, 1.0]), &[1.0]),
        (ColumnVec::from_slice(&[1.0, 0.0]), &[0.0]),
        (ColumnVec::from_slice(&[1.0, 1.0]), &[1.0]),
    ];

    let layers = [data[0].0.len(), data[0].1.len()];

    let mut nn = NeuronNetwork::new(&layers, || 0.5);

    let mut visualizer = Visualizer::new().unwrap();

    for _ in 0..1000 {
        nn.train_step(&mut data);
        visualizer.draw_nn(&nn);

        println!("{}", nn.cost(&mut data))
    }
}
