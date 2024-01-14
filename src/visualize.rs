extern crate sdl2;

use sdl2::{
    pixels::Color,
    rect::{Point, Rect},
    render::Canvas,
    video::Window,
    EventPump,
};
use std::time::Duration;

use crate::nn::NeuronNetwork;

// create plot widget that is capable of updating its content
pub struct Visualizer {
    canvas: Canvas<Window>,
    event_pump: EventPump,
}

const SIZE: i32 = 800;

impl Visualizer {
    pub fn new() -> Result<Self, String> {
        let sdl_context = sdl2::init()?;

        let canvas = sdl_context
            .video()?
            .window("graph", SIZE as u32, SIZE as u32)
            .position_centered()
            .build()
            .map_err(|e| e.to_string())?
            .into_canvas()
            .build()
            .map_err(|e| e.to_string())?;

        let event_pump = sdl_context.event_pump()?;

        Ok(Self { canvas, event_pump })
    }

    pub fn draw_nn(&mut self, nn: &NeuronNetwork) {
        for event in self.event_pump.poll_iter() {
            match event {
                _ => {}
            }
        }

        self.canvas.set_draw_color(Color::BLACK);
        self.canvas.clear();

        const PADDING_LEFT: f64 = 50.0;
        const BASELINE: f64 = SIZE as f64 / 2.0;
        const NEURON_RADIUS: f64 = 40.0;
        const SPACE_BETWEEN_NEURONS: f64 = 20.0;
        const SPACE_BETWEEN_LAYERS: f64 = 200.0;

        let model = nn.get_model();

        let get_neuron_position = |l: usize, n: usize| {
            let x_pos = { PADDING_LEFT as i32 + SPACE_BETWEEN_LAYERS as i32 * l as i32 };

            let y_pos = {
                (BASELINE
                    + ((model[l].columns - 1) as f64 / 2.0 - n as f64)
                        * (2.0 * NEURON_RADIUS + SPACE_BETWEEN_NEURONS)) as i32
            };

            (x_pos, y_pos)
        };

        model.iter().enumerate().for_each(|(i, weights)| {
            let nr_of_neurons = weights.rows;
            let nr_of_neurons_in_next_layer = weights.columns;

            for j in 0..nr_of_neurons {
                let neuron_center = Point::from(get_neuron_position(i, j));

                self.canvas.set_draw_color(Color::WHITE);
                self.canvas.draw_rect(Rect::from_center(
                    neuron_center,
                    NEURON_RADIUS as u32 * 2,
                    NEURON_RADIUS as u32 * 2,
                ));

                let mut weight_start = neuron_center;
                weight_start.x += NEURON_RADIUS as i32;

                for k in 0..nr_of_neurons_in_next_layer {
                    let weight_destination = Point::from((
                        PADDING_LEFT as i32 + SPACE_BETWEEN_LAYERS as i32 * (i as i32 + 1)
                            - NEURON_RADIUS as i32,
                        (BASELINE
                            + ((nr_of_neurons_in_next_layer as i32
                                - 1
                                - if is_last_layer { 0 } else { 1 })
                                as f64
                                / 2.0
                                - k as f64)
                                * (2.0 * NEURON_RADIUS + SPACE_BETWEEN_NEURONS))
                            as i32,
                    ));

                    self.canvas.set_draw_color(Color::RGB(
                        0,
                        (255.0 * weights.get_cell(j, k)) as u8,
                        0,
                    ));
                    self.canvas.draw_line(weight_start, weight_destination);
                }
            }

            if is_last_layer {
                for j in 0..nr_of_neurons_in_next_layer {
                    let neuron_center = Point::from((
                        PADDING_LEFT as i32 + SPACE_BETWEEN_LAYERS as i32 * (i as i32 + 1),
                        (BASELINE
                            + ((nr_of_neurons_in_next_layer - 1) as f64 / 2.0 - j as f64)
                                * (2.0 * NEURON_RADIUS + SPACE_BETWEEN_NEURONS))
                            as i32,
                    ));

                    self.canvas.set_draw_color(Color::WHITE);
                    self.canvas.draw_rect(Rect::from_center(
                        neuron_center,
                        NEURON_RADIUS as u32 * 2,
                        NEURON_RADIUS as u32 * 2,
                    ));
                }
            }
        });

        self.canvas.present();
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
    }
}
