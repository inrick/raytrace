use std::sync::mpsc::{sync_channel, Receiver};
use std::time::Instant;

use eframe::egui;
use egui::{Color32, ColorImage};

use crate::ray::{raytrace, save_file, Args, Image};

enum RenderState {
	Idle,
	Running(Instant, Receiver<Message>),
}

pub struct App {
	render_state: RenderState,
	filename: String,
	save_status: Option<String>,
	nsamples: u32,
	threads: u32,
	info: Option<String>,
	result: Option<ImageResult>,
}

struct Message {
	image: Image,
	t0: Instant,
	t1: Instant,
}

struct ImageResult {
	image: Image,
	texture: egui::TextureHandle,
}

impl From<&Image> for ColorImage {
	fn from(img: &Image) -> ColorImage {
		let size = [img.nx as usize, img.ny as usize];
		let mut cimg = ColorImage::new(size, Color32::BLACK);
		for (i, pixel) in cimg.pixels.iter_mut().enumerate() {
			let r = img.buf[3 * i];
			let g = img.buf[3 * i + 1];
			let b = img.buf[3 * i + 2];
			*pixel = Color32::from_rgb(r, g, b);
		}
		cimg
	}
}

impl Default for App {
	fn default() -> Self {
		Self {
			render_state: RenderState::Idle,
			filename: "out.png".to_owned(),
			save_status: None,
			nsamples: 10,
			threads: 8,
			info: None,
			result: None,
		}
	}
}

impl eframe::App for App {
	fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
		egui::CentralPanel::default().show(ctx, |ui| {
			if let RenderState::Running(_, rx) = &self.render_state {
				if let Ok(Message { image, t0, t1 }) = rx.try_recv() {
					self.info = Some(format!(
						"Rendering took {:.2} seconds",
						t1.duration_since(t0).as_secs_f32()
					));
					let cimg = ColorImage::from(&image);
					self.result = Some(ImageResult {
						image,
						texture: ui.ctx().load_texture("ray", cimg),
					});
					self.render_state = RenderState::Idle;
				} else {
					// While the renderer is running, keep repainting until we get a
					// result.
					ui.ctx().request_repaint();
				}
			}

			ui.horizontal(|ui| {
				ui.label("Samples");
				ui.add(egui::Slider::new(&mut self.nsamples, 1..=200));
				ui.label("Threads");
				ui.add(egui::Slider::new(&mut self.threads, 1..=16));
			});
			ui.horizontal(|ui| {
				ui.scope(|ui| {
					ui.set_enabled(matches!(self.render_state, RenderState::Idle));
					if ui.button("Run").clicked() {
						self.save_status = None;
						let args = Args {
							nsamples: self.nsamples,
							threads: self.threads,
						};
						let (tx, rx) = sync_channel(0);
						self.render_state = RenderState::Running(Instant::now(), rx);
						std::thread::spawn(move || {
							let t0 = Instant::now();
							let image = raytrace(&args, 600, 300);
							let t1 = Instant::now();
							tx.send(Message { image, t0, t1 }).unwrap()
						});
					}
				});
				if let RenderState::Running(t0, _) = self.render_state {
					ui.label(format!("{:.2} seconds...", t0.elapsed().as_secs_f32()));
				} else if let Some(info) = &self.info {
					ui.label(info);
				}
			});
			if let Some(state) = &self.result {
				ui.vertical(|ui| {
					ui.image(&state.texture, state.texture.size_vec2());
					ui.horizontal(|ui| {
						ui.text_edit_singleline(&mut self.filename);
						if ui.button("Save to file").clicked() {
							let save_status = save_file(&state.image, &self.filename);
							self.save_status = Some(format!(
								"{} {}",
								chrono::prelude::Local::now().format("[%H:%M:%S]"),
								match save_status {
									Ok(()) =>
										format!("File \"{}\" saved successfully", &self.filename),
									Err(err) => format!("ERROR, could not save file: {}", err),
								}
							));
						}
					});
					if let Some(status) = &self.save_status {
						ui.label(status);
					}
				});
			}
		});
	}
}
