#![feature(test)]
extern crate test;
use rand::Rng;
use test::Bencher;

use ai_playground::{ train, TrainConfig};

fn main() {
    let mut random = rand::thread_rng();
    let data = [
        ([0.0, 0.0, 1.0], [0.0]),
        ([0.0, 1.0, 0.0], [1.0]),
        ([1.0, 0.0, 1.0], [1.0]),
        ([1.0, 1.0, 1.0], [0.0]),
    ];

    let config = TrainConfig {
        eps: 0.1,
        rate: 0.1,
        generate_parameter: || random.gen_range(0.0..1.0),
        nr_of_iterations: 100000,
        hidden_layers: [4, 2, 3],
        data,
    };

    let model = train(config);
    
}

#[bench]
fn bench_xor(b: &mut Bencher) {
    // 16,336,874 ns/iter (+/- 291,784)
    //  9,050,475 ns/iter (+/- 421,420) after mat add allocation optimalization
    //  4,006,797 ns/iter (+/- 28,356) after mat multiplication prealocation optimalization
    b.iter(|| {
        let data = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ];

        let mut c = 0.0;

        let generate_parameter = || {
            c += 0.01;
            c
        };

        let config = TrainConfig {
            eps: 0.1,
            rate: 0.1,
            nr_of_iterations: 40,
            generate_parameter,
            hidden_layers: [2, 3, 5, 7],
            data,
        };

        train(config);
    })
}
