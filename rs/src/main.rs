#![allow(clippy::identity_op)]
#![allow(clippy::uninlined_format_args)]

mod math;
mod ray;
mod scene;
mod vec;

#[cfg(not(feature = "gui"))]
mod cli;

#[cfg(feature = "gui")]
mod gui;

#[cfg(not(feature = "gui"))]
fn main() {
	if let Err(err) = cli::run() {
		eprintln!("ERROR: {}", err);
		std::process::exit(1);
	}
}

#[cfg(feature = "gui")]
fn main() {
	let options = eframe::NativeOptions::default();
	let _ = eframe::run_native(
		"Raytracer",
		options,
		Box::new(|cc| Box::new(gui::App::new(cc))),
	);
}
