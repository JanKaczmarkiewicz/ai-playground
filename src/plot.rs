extern crate sdl2;

use sdl2::{pixels::Color, rect::Rect, render::Canvas, video::Window, EventPump};
use std::time::Duration;

// create plot widget that is capable of updating its content
pub struct Plot {
    canvas: Canvas<Window>,
    event_pump: EventPump,
    size: u32,
}

impl Plot {
    pub fn new() -> Result<Self, String> {
        let sdl_context = sdl2::init()?;

        let size = 800;

        let canvas = sdl_context
            .video()?
            .window("rust-sdl2 demo: Events", size, size)
            .position_centered()
            .resizable()
            .build()
            .map_err(|e| e.to_string())?
            .into_canvas()
            .build()
            .map_err(|e| e.to_string())?;

        let event_pump = sdl_context.event_pump()?;

        Ok(Self {
            canvas,
            event_pump,
            size,
        })
    }

    // x and y are values from (-1..1) to (0..800)
    // eg: (-0.2, 0.2) ->
    fn value_to_point(&self, (x, y): (f64, f64)) -> (i32, i32) {
        (
            ((x + 1.0) * self.size as f64 / 2.0) as i32,
            (self.size as f64 - (y + 1.0) * self.size as f64 / 2.0) as i32,
        )
    }

    pub fn update<G: FnMut(f64) -> f64>(&mut self, mut f: G, data_batch: &[(&[f64], &[f64])]) {
        for event in self.event_pump.poll_iter() {
            match event {
                _ => {}
            }
        }

        self.canvas.set_draw_color(Color::RGB(255, 255, 255));
        self.canvas.clear();

        self.canvas.set_draw_color(Color::RGB(255, 0, 0));

        let size = self.size as i32;

        self.canvas.draw_line((0, size / 2), (size, size / 2));
        self.canvas.draw_line((size / 2, 0), (size / 2, size));

        self.canvas.set_draw_color(Color::RGB(0, 0, 0));

        let number_of_dividers = 20;

        let distance_between = size / number_of_dividers;
        let line_length = 10; // px

        (0..number_of_dividers).for_each(|prev_x| {
            {
                // horizontal
                let x_start = distance_between * prev_x;
                let y_start = (size - line_length) / 2;

                let x_end = x_start;
                let y_end = y_start + line_length;

                self.canvas.draw_line((x_start, y_start), (x_end, y_end));
            }

            {
                // vertical
                let x_start = (size - line_length) / 2;
                let y_start = distance_between * prev_x;

                let x_end = x_start + line_length;
                let y_end = y_start;

                self.canvas.draw_line((x_start, y_start), (x_end, y_end));
            }
        });

        let nr_of_points = 1000;

        (0..nr_of_points)
            .map(|i| (i - nr_of_points / 2) as f64 / 10.0)
            .reduce(|prev_x, x: f64| {
                self.canvas.draw_line(
                    self.value_to_point((x, f(x))),
                    self.value_to_point((prev_x, f(prev_x))),
                );

                x
            });

        self.canvas.set_draw_color(Color::RGB(0, 255, 0));

        data_batch.iter().for_each(|(x, y)| {
            let x = x[0];
            let y = y[0];

            self.canvas
                .draw_rect(Rect::from_center(self.value_to_point((x, y)), 5, 5));
        });

        self.canvas.present();
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
    }
}
