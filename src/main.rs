use ai_playground::{simulate, SimulationConfig};

fn main() {
    let model = simulate(SimulationConfig {
        eps: 0.01,
        rate: 0.01,
        nr_of_iterations: 1000,
        layers: &[2, 1],
        data: &[
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
        ],
    });
}
