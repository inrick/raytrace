use std::ops::{Deref, DerefMut};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::time::Instant;

use eframe::egui::{self, Color32, ColorImage, Key, Modifiers};
use eframe::epaint::FontId;

use crate::ray::{
	raytrace, save_file, small_scene, Args, Camera, Image, Scene,
};
use crate::vec::{vec, Vec3};

enum RenderState {
	Idle,
	Running(Instant, Receiver<Message>),
}

pub struct App {
	render_state: RenderState,
	jobs: Sender<(Args, SyncSender<Message>)>,
	filename: String,
	save_status: Option<String>,
	actions: Action,
	info: Option<String>,
	result: Option<ImageResult>,
	windows: Windows,
	log: Log,

	// Raytrace arguments
	nsamples: u32,
	threads: u32,
	scene: Scene,

	look_from: Vec3,
	look_at: Vec3,
}

#[derive(Default)]
struct Log(Vec<String>);

impl Deref for Log {
	type Target = Vec<String>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl DerefMut for Log {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

impl Log {
	fn log(&mut self, level: &str, msg: &str) {
		let msg = format!(
			"{} :: {} :: {}",
			chrono::prelude::Local::now().format("%H:%M:%S.%3f"),
			level,
			msg
		);
		self.0.push(msg);
	}

	fn info(&mut self, msg: &str) {
		self.log("INFO", msg);
	}

	fn error(&mut self, msg: &str) {
		self.log("ERROR", msg);
	}
}

#[derive(Default)]
struct Windows {
	settings: bool,
	log: bool,
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

#[derive(Copy, Clone)]
struct Action(u32);

const ACTION_NONE: Action = Action(0b0);
const ACTION_RENDER: Action = Action(0b1);
const ACTION_TOGGLE_SETTINGS_WINDOW: Action = Action(0b10);
const ACTION_TOGGLE_LOG_WINDOW: Action = Action(0b100);

impl Action {
	fn matches(self, a: Action) -> bool {
		self.0 & a.0 > 0
	}

	fn set(&mut self, a: Action) {
		self.0 |= a.0;
	}
}

fn set_egui_style(ctx: &egui::Context) {
	use egui::FontFamily::{Monospace, Proportional};
	let mut fonts = egui::FontDefinitions::default();
	fonts.font_data.insert(
		"fira_sans".into(),
		egui::FontData::from_static(include_bytes!("../data/FiraSans-Regular.ttf")),
	);
	fonts.font_data.insert(
		"fira_mono".into(),
		egui::FontData::from_static(include_bytes!("../data/FiraMono-Regular.ttf")),
	);
	fonts
		.families
		.entry(Proportional)
		.or_default()
		.insert(0, "fira_sans".into());
	fonts
		.families
		.entry(Monospace)
		.or_default()
		.insert(0, "fira_mono".into());
	let mut style = ctx.style().deref().clone();
	style
		.text_styles
		.iter_mut()
		.for_each(|(key, val)| match &key {
			egui::TextStyle::Body => *val = FontId::new(18., Proportional),
			egui::TextStyle::Button => *val = FontId::new(18., Proportional),
			egui::TextStyle::Monospace => *val = FontId::new(17., Monospace),
			_ => (),
		});
	ctx.set_pixels_per_point(1.);
	ctx.set_fonts(fonts);
	ctx.set_style(style);
}

impl App {
	pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
		set_egui_style(&cc.egui_ctx);
		let (tx, rx) = channel::<(Args, SyncSender<_>)>();
		std::thread::spawn(move || {
			while let Ok((args, tx)) = rx.recv() {
				let t0 = Instant::now();
				let image = raytrace(&args, 600, 300);
				let t1 = Instant::now();
				tx.send(Message { image, t0, t1 }).unwrap();
			}
		});
		Self {
			render_state: RenderState::Idle,
			jobs: tx,
			filename: "out.png".to_owned(),
			save_status: None,
			actions: ACTION_RENDER, // Run render on startup
			info: None,
			result: None,
			windows: Windows::default(),
			log: Log::default(),
			nsamples: 10,
			threads: 8,
			look_from: vec(10., 2.5, 5.),
			look_at: vec(-4., 0., -2.),
			scene: small_scene(),
		}
	}
}

impl eframe::App for App {
	fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
		// Process keys
		if ctx.input().key_pressed(Key::Enter) {
			self.actions.set(ACTION_RENDER);
		}
		if ctx.input().modifiers.matches(Modifiers::CTRL) {
			if ctx.input().key_pressed(Key::Num1) {
				self.actions.set(ACTION_TOGGLE_SETTINGS_WINDOW);
			} else if ctx.input().key_pressed(Key::Num2) {
				self.actions.set(ACTION_TOGGLE_LOG_WINDOW);
			}
		}

		egui::Window::new("Settings")
			.open(&mut self.windows.settings)
			.show(ctx, |ui| {
				ctx.settings_ui(ui);
			});
		egui::Window::new("Log")
			.open(&mut self.windows.log)
			.default_size([500., 300.])
			.show(ctx, |ui| {
				if ui.button("Clear log").clicked() {
					self.log.clear();
				}
				ui.separator();
				egui::ScrollArea::vertical()
					.auto_shrink([false, false])
					.show(ui, |ui| {
						for row in self.log.iter() {
							ui.label(row);
						}
					});
			});

		egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
			egui::menu::bar(ui, |ui| {
				ui.menu_button("File", |ui| {
					if ui.button("Quit").clicked() {
						frame.quit();
					}
				});
				ui.menu_button("Windows", |ui| {
					if ui.button("Settings").clicked() {
						self.actions.set(ACTION_TOGGLE_SETTINGS_WINDOW);
						ui.close_menu();
					}
					if ui.button("Log").clicked() {
						self.actions.set(ACTION_TOGGLE_LOG_WINDOW);
						ui.close_menu();
					}
				});
			});
		});

		if let RenderState::Running(_, rx) = &self.render_state {
			if let Ok(Message { image, t0, t1 }) = rx.try_recv() {
				let msg = format!(
					"Rendering took {:.2} seconds",
					t1.duration_since(t0).as_secs_f32()
				);
				self.log.info(&msg);
				self.info = Some(msg);
				let cimg = ColorImage::from(&image);
				self.result = Some(ImageResult {
					image,
					texture: ctx.load_texture("ray", cimg),
				});
				self.render_state = RenderState::Idle;
			} else {
				// While the renderer is running, keep repainting until we get a
				// result.
				ctx.request_repaint();
			}
		}

		egui::CentralPanel::default().show(ctx, |ui| {
			ui.horizontal(|ui| {
				ui.vertical(|ui| {
					ui.label("Camera");
					ui.horizontal(|ui| {
						ui.vec_slider("From", &mut self.look_from);
						ui.vec_slider("At", &mut self.look_at);
						ui.vertical(|ui| {
							ui.add(
								egui::Slider::new(&mut self.nsamples, 1..=200).text("Samples"),
							);
							ui.add(
								egui::Slider::new(&mut self.threads, 1..=16).text("Threads"),
							);
							ui.horizontal(|ui| {
								ui.scope(|ui| {
									ui.set_enabled(matches!(
										self.render_state,
										RenderState::Idle
									));
									if ui.button("Render").clicked() {
										self.actions.set(ACTION_RENDER);
									}
								});
								if let RenderState::Running(t0, _) = self.render_state {
									ui.add(egui::Spinner::new());
									ui.label(format!(
										"{:.2} seconds...",
										t0.elapsed().as_secs_f32()
									));
								} else if let Some(info) = &self.info {
									ui.label(info);
								}
							});
						});
					});
				});
			});
			if let Some(state) = &self.result {
				ui.vertical(|ui| {
					ui.image(&state.texture, state.texture.size_vec2());
					ui.horizontal(|ui| {
						ui.text_edit_singleline(&mut self.filename);
						if ui.button("Save to file").clicked() {
							let save_status = save_file(&state.image, &self.filename);
							let msg;
							match save_status {
								Ok(()) => {
									msg =
										format!("File \"{}\" saved successfully", &self.filename);
									self.log.info(&msg);
								}
								Err(err) => {
									msg = format!(
										"Could not save file \"{}\": {}",
										self.filename, err
									);
									self.log.error(&msg);
								}
							};
							self.save_status = Some(msg);
						}
					});
					if let Some(status) = &self.save_status {
						ui.label(status);
					}
				});
			}
		});

		// Process any actions triggered this frame
		if self.actions.matches(ACTION_RENDER)
			&& matches!(self.render_state, RenderState::Idle)
		{
			self.log.info(&format!(
				"Starting render using {} samples and {} threads",
				self.nsamples, self.threads
			));
			self.save_status = None;
			let args = Args {
				nsamples: self.nsamples,
				threads: self.threads,
				cam: new_camera(600, 300, self.look_from, self.look_at),
				scene: self.scene.clone(),
			};
			let (tx, rx) = sync_channel(0);
			self.render_state = RenderState::Running(Instant::now(), rx);
			self.jobs.send((args, tx)).unwrap();
		}
		if self.actions.matches(ACTION_TOGGLE_SETTINGS_WINDOW) {
			self.windows.settings = !self.windows.settings;
		}
		if self.actions.matches(ACTION_TOGGLE_LOG_WINDOW) {
			self.windows.log = !self.windows.log;
		}

		// Clear actions
		self.actions = ACTION_NONE;
	}
}

trait UiExtensions {
	fn vec_slider(&mut self, name: &str, u: &mut Vec3);
}

impl UiExtensions for egui::Ui {
	fn vec_slider(&mut self, label: &str, u: &mut Vec3) {
		self.group(|ui| {
			ui.vertical(|ui| {
				ui.label(label);
				ui.horizontal(|ui| {
					ui.label("x");
					ui.add(egui::DragValue::new(&mut u.x).speed(0.05));
				});
				ui.horizontal(|ui| {
					ui.label("y");
					ui.add(egui::DragValue::new(&mut u.y).speed(0.05));
				});
				ui.horizontal(|ui| {
					ui.label("z");
					ui.add(egui::DragValue::new(&mut u.z).speed(0.05));
				});
			});
		});
	}
}

fn new_camera(nx: u32, ny: u32, look_from: Vec3, look_at: Vec3) -> Camera {
	let dist_to_focus = (look_from - look_at).norm();
	let aperture = 0.05;

	let (nxf, nyf) = (nx as f32, ny as f32);

	Camera::new(
		look_from,
		look_at,
		vec(0., 1., 0.),
		20.,
		nxf / nyf,
		aperture,
		dist_to_focus,
	)
}
