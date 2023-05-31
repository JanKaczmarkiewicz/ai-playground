use ai_playground::{cost, train, TrainConfig};

fn main() {
    let config = TrainConfig {
        eps: 0.01,
        rate: 0.01,
        nr_of_iterations: 2,
        layers: &[2, 1],
        data: &[
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
        ],
    };

    let model = train(config);
}

