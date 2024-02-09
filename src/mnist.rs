use std::fs::File;
use std::io::prelude::*;
use std::io::SeekFrom;

const IMAGE_SIZE: usize = 28 * 28;
const N: usize = 1000;

pub struct Image(pub f64, pub [f64; IMAGE_SIZE]);

pub fn load() -> Result<Vec<Image>, ()> {
    let mut image_file = File::open("./src/data/mnist/train-images-idx3-ubyte").unwrap();
    let mut label_file = File::open("./src/data/mnist/train-labels-idx1-ubyte").unwrap();

    image_file.seek(SeekFrom::Start(16));
    label_file.seek(SeekFrom::Start(8));

    let mut image_data_buffer = [0; IMAGE_SIZE * N];
    let mut label_data_buffer = [0; N];

    image_file.read_exact(&mut image_data_buffer);
    label_file.read_exact(&mut label_data_buffer);

    Ok(image_data_buffer
        .array_chunks::<IMAGE_SIZE>()
        .zip(label_data_buffer)
        .map(|(s, l)| (Image(l as f64 / 10.0, s.map(|x| x as f64 / 255.0).to_owned())))
        .collect())
}
