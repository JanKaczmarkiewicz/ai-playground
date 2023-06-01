use ai_playground::{cost, train, Data, TrainConfig};

fn main() {
    let data: Data = &[
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[1.0]),
        (&[1.0, 0.0], &[1.0]),
        (&[1.0, 1.0], &[0.0]),
    ];

    let config = TrainConfig {
        eps: 0.1,
        rate: 0.1,
        nr_of_iterations: 100000,
        hidden_layers: &[2],
        data,
    };

    let model = train(config);

    println!("cost: {}", cost(data, &model));
}
