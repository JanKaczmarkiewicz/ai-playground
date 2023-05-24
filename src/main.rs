use rand::Rng;
use std::f64::consts::E;

const DATA: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn cost(w1: f64, w2: f64, b: f64) -> f64 {
    let mut result = 0.0;

    for sample in DATA.iter() {
        let x1 = sample[0];
        let x2 = sample[1];
        let expected_y = sample[2];

        let y = sigmoid(x1 * w1 + x2 * w2 + b);

        result += (y - expected_y).powf(2.0);
    }

    result /= DATA.len() as f64;

    result
}

fn main() {
    let mut w1 = rand::thread_rng().gen_range(0.0..1.0);
    let mut w2 = rand::thread_rng().gen_range(0.0..1.0);
    let mut b = rand::thread_rng().gen_range(0.0..1.0);

    let eps = 0.001;
    let rate = 0.001;

    for _ in 0..50000000 {
        let c = cost(w1, w2, b);
        let d_cost_w1 = (cost(w1 + eps, w2, b) - c) / eps;
        let d_cost_w2 = (cost(w1, w2 + eps, b) - c) / eps;
        let d_cost_b = (cost(w1, w2, b + eps) - c) / eps;

        w1 -= rate * d_cost_w1;
        w2 -= rate * d_cost_w2;
        b -= rate * d_cost_b;

        // println!("cost (w1, w2) = ({}, {})", d_cost_w1, d_cost_w2);
    }

    println!("w1: {w1}, w2: {w2}");

    for sample in DATA {
        let x1 = sample[0];
        let x2 = sample[1];
        let y = sample[2];

        println!("{x1} | {x2} = {y} ({})", sigmoid(x1 * w1 + x2 * w2 + b))
    }
}
