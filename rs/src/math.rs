pub static PI: f32 = std::f32::consts::PI;

// TODO: consider passing around rng everywhere instead

pub fn rand32() -> f32 {
	rand::random()
}

pub fn deg_to_rad(deg: f32) -> f32 {
	deg * PI / 180.
}
