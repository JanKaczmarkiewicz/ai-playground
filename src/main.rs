use ai_playground::{cost, train, TrainConfig};

fn main() {
    let config = TrainConfig {
        eps: 0.1,
        rate: 0.1,
        nr_of_iterations: 100000,
        layers: &[2, 1, 3, 2],
        data: &[
            &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
        ],
    };

    let model = train(config);
    println!()
}

