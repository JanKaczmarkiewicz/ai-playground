extern crate sdl2;

use std::time::Duration;

pub fn render() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let _ = video_subsystem
        .window("", 800, 600)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    loop {
        std::thread::sleep(Duration::from_millis(100));

        let {weights, outputs} = get_updated_state();
    }
}
