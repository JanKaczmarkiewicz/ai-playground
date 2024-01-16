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
        let cost = nn.cost(&mut data);
        let model = nn.get_model();
        visualizer.draw_nn(model, cost);

        println!("{}", nn.cost(&mut data))
    }
}
