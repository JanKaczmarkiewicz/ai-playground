use rand;
use rand::Rng;
use std::f32::consts::E;

const DATA: [(f32, f32, f32); 4] = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

#[derive(Default, Debug)]
struct NeuronParams {
    w1: f32,
    w2: f32,
    b: f32,
}

#[derive(Default, Debug)]
struct XorModel {
    first: NeuronParams,
    second: NeuronParams,
    third: NeuronParams,
}

fn neuron_network(m: &XorModel, x1: f32, x2: f32) -> f32 {
    let first_neuron_y = sigmoid(m.first.w1 * x1 + m.first.w2 * x2 + m.first.b);
    let second_neuron_y = sigmoid(m.second.w1 * x1 + m.second.w2 * x2 + m.second.b);
    let third_neuron_y =
        sigmoid(m.third.w1 * first_neuron_y + m.third.w2 * second_neuron_y + m.third.b);

    // println!("first_neuron_y: {first_neuron_y}, second_neuron_y: {second_neuron_y}, third_neuron_y: {third_neuron_y}");
    return third_neuron_y;
}

fn cost(m: &XorModel) -> f32 {
    let mut result = 0.0;

    for (x1, x2, y) in DATA {
        let cost_y = neuron_network(&m, x1, x2);
        result += (y - cost_y).powi(2);
    }

    result /= DATA.len() as f32;

    result
}

fn subtract_models(m1: &mut XorModel, m2: &XorModel) {
    m1.first.w1 -= m2.first.w1;
    m1.first.w2 -= m2.first.w2;
    m1.first.b -= m2.first.b;
    m1.second.w1 -= m2.second.w1;
    m1.second.w2 -= m2.second.w2;
    m1.second.b -= m2.second.b;
    m1.third.w1 -= m2.third.w1;
    m1.third.w2 -= m2.third.w2;
    m1.third.b -= m2.third.b;
}

fn get_dir(m: &mut XorModel, eps: f32, rate: f32) -> XorModel {
    let c = cost(&m);
    let mut dir_m = XorModel::default();

    let temp = m.first.w1;
    m.first.w1 += eps;
    dir_m.first.w1 = ((cost(&m) - c) / eps) * rate;
    m.first.w1 = temp;

    let temp = m.first.w2;
    m.first.w2 += eps;
    dir_m.first.w2 = ((cost(&m) - c) / eps) * rate;
    m.first.w2 = temp;

    let temp = m.first.b;
    m.first.b += eps;
    dir_m.first.b = ((cost(&m) - c) / eps) * rate;
    m.first.b = temp;

    // second

    let temp = m.second.w1;
    m.second.w1 += eps;
    dir_m.second.w1 = ((cost(&m) - c) / eps) * rate;
    m.second.w1 = temp;

    let temp = m.second.w2;
    m.second.w2 += eps;
    dir_m.second.w2 = ((cost(&m) - c) / eps) * rate;
    m.second.w2 = temp;

    let temp = m.second.b;
    m.second.b += eps;
    dir_m.second.b = ((cost(&m) - c) / eps) * rate;
    m.second.b = temp;

    // third

    let temp = m.third.w1;
    m.third.w1 += eps;
    dir_m.third.w1 = ((cost(&m) - c) / eps) * rate;
    m.third.w1 = temp;

    let temp = m.third.w2;
    m.third.w2 += eps;
    dir_m.third.w2 = ((cost(&m) - c) / eps) * rate;
    m.third.w2 = temp;

    let temp = m.third.b;
    m.third.b += eps;
    dir_m.third.b = ((cost(&m) - c) / eps) * rate;
    m.third.b = temp;

    dir_m
}

fn main() {
    let mut random = rand::thread_rng();

    let mut m = XorModel {
        first: NeuronParams {
            w1: random.gen_range(0.0..1.0),
            w2: random.gen_range(0.0..1.0),
            b: random.gen_range(0.0..1.0),
        },
        second: NeuronParams {
            w1: random.gen_range(0.0..1.0),
            w2: random.gen_range(0.0..1.0),
            b: random.gen_range(0.0..1.0),
        },
        third: NeuronParams {
            w1: random.gen_range(0.0..1.0),
            w2: random.gen_range(0.0..1.0),
            b: random.gen_range(0.0..1.0),
        },
    };

    let eps = 0.1;
    let rate = 0.1;

    for _ in 0..10000000 {
        let dir = get_dir(&mut m, eps, rate);
        subtract_models(&mut m, &dir);
    }

    println!("cost: {:?}", cost(&m));

    for (x1, x2, y) in DATA {
        let result = neuron_network(&m, x1, x2);
        println!("{x1} ^ {x2} = {y} ({result})");
    }
}
