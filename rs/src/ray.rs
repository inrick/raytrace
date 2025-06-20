use std::{ffi::OsStr, fs::File, io::Write, path::PathBuf};

use crate::math::{deg_to_rad, rand32};
use crate::vec::*;

#[derive(Debug)]
pub struct Config {
	pub nsamples: u32,
	pub threads: u32,
	pub nx: u32,
	pub ny: u32,
	pub max_depth: u32,
}

#[derive(Debug)]
pub struct CameraConfig {
	pub fov: f32,
	pub look_from: Vec3,
	pub look_at: Vec3,
	pub v_up: Vec3,
	pub defocus_angle: f32,
	pub focus_dist: f32,
}

#[derive(Debug)]
pub struct Args {
	pub cfg: Config,
	pub cam: Camera,
	pub scene: Scene,
}

struct ThreadArgs<'a> {
	buf: &'a mut [u8],
	cam: &'a Camera,
	scene: &'a Scene,
	x0: u32,
	x1: u32,
	y0: u32,
	y1: u32,
}

pub struct Image {
	pub buf: Vec<u8>,
	pub nx: u32,
	pub ny: u32,
}

type Error = Box<dyn std::error::Error>;
type Result<T> = ::std::result::Result<T, Error>;

type ImageWriter = fn(&mut File, &[u8], u32, u32) -> Result<()>;

pub fn save_file(img: &Image, filename: &str) -> Result<()> {
	let extension = PathBuf::from(filename)
		.extension()
		.and_then(OsStr::to_str)
		.map(|s| s.to_lowercase())
		.ok_or("missing file extension")?;
	let image_writer: ImageWriter = match extension.as_ref() {
		"ppm" => ppm_write,
		"png" => |f, buf, x, y| {
			image::write_buffer_with_format(
				f,
				buf,
				x,
				y,
				image::ColorType::Rgb8,
				image::ImageOutputFormat::Png,
			)?;
			Ok(())
		},
		"jpg" | "jpeg" => |f, buf, x, y| {
			image::write_buffer_with_format(
				f,
				buf,
				x,
				y,
				image::ColorType::Rgb8,
				image::ImageOutputFormat::Jpeg(90),
			)?;
			Ok(())
		},
		unknown => {
			return Err(
				format!(
					"unknown image output format for file extension '{}' \
        (only know ppm/png/jpg)",
					unknown,
				)
				.into(),
			)
		}
	};
	let mut f = File::create(filename)?;
	image_writer(&mut f, &img.buf, img.nx, img.ny)
}

pub fn raytrace(args: &Args) -> Image {
	let Config {
		nsamples,
		threads,
		nx,
		ny,
		max_depth,
	} = args.cfg;
	if threads == 0 || nsamples == 0 || max_depth == 0 {
		panic!("number of samples/threads/depth must be positive");
	}
	let cam = &args.cam;
	let scene = &args.scene;

	let mut buf = vec![0; (3 * nx * ny) as usize];
	std::thread::scope(|s| {
		let mut ypos = 0u32;
		for i in 0..threads {
			let chunk = (cam.image_height - ypos) / (threads - i);
			let buf_alias =
				unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.len()) };
			let render_args = ThreadArgs {
				buf: buf_alias,
				cam,
				scene,
				x0: 0,
				x1: cam.image_width,
				y0: ypos,
				y1: ypos + chunk,
			};
			s.spawn(move || {
				render(render_args);
			});
			ypos += chunk;
		}
	});

	Image { buf, nx, ny }
}

fn linear_to_gamma(lin: f32) -> f32 {
	if lin > 0. {
		lin.sqrt()
	} else {
		lin
	}
}

#[allow(clippy::identity_op)]
fn render(args: ThreadArgs) {
	let ThreadArgs {
		buf,
		cam,
		scene,
		x0,
		x1,
		y0,
		y1,
	} = args;
	let stride = cam.image_width;
	for j in y0..y1 {
		for i in x0..x1 {
			let color = ray_color_at_ij(cam, scene, i, j);
			let pos = 3 * (j * stride + i) as usize;
			buf[pos + 0] = (255. * linear_to_gamma(color.x)) as u8;
			buf[pos + 1] = (255. * linear_to_gamma(color.y)) as u8;
			buf[pos + 2] = (255. * linear_to_gamma(color.z)) as u8;
		}
	}
}

fn ppm_write(f: &mut File, buf: &[u8], x: u32, y: u32) -> Result<()> {
	f.write_all(format!("P6\n{} {} 255\n", x, y).as_bytes())?;
	f.write_all(buf)?;
	Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
	pub origin: Vec3,
	pub dir: Vec3,
	pub time: f32,
}

impl Ray {
	fn eval(self, t: f32) -> Vec3 {
		self.origin + self.dir * t
	}
}

#[derive(Copy, Clone, Debug)]
pub enum Material {
	Matte { albedo: Vec3 },
	Metal { albedo: Vec3, fuzz: f32 },
	Dielectric { ref_idx: f32 },
}

impl Default for Material {
	fn default() -> Self {
		Material::Matte {
			albedo: Vec3::default(),
		}
	}
}

#[derive(Default)]
struct HitRecord {
	t: f32,
	p: Vec3,
	normal: Vec3,
	mat: Material,
	front_face: bool,
}

impl HitRecord {
	// NOTE: outward_normal is assumed to be of unit length
	pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
		self.front_face = r.dir.dot(outward_normal) < 0.;
		self.normal = if self.front_face {
			outward_normal
		} else {
			-outward_normal
		};
	}
}

#[derive(Debug, Clone)]
pub struct Sphere {
	center: Ray,
	radius: f32,
	mat: Material,
}

impl Sphere {
	pub fn new(center: Ray, radius: f32, mat: Material) -> Self {
		Sphere {
			center,
			radius,
			mat,
		}
	}

	pub fn new_static(center: Vec3, radius: f32, mat: Material) -> Self {
		Sphere {
			center: Ray {
				origin: center,
				dir: Vec3::default(),
				time: 0.,
			},
			radius,
			mat,
		}
	}

	pub fn bounding_box(&self) -> Aabb {
		let center = self.center;
		let r_vec = vec(self.radius, self.radius, self.radius);
		let box1 =
			Aabb::new_from_vec(center.eval(0.) - r_vec, center.eval(0.) + r_vec);
		let box2 =
			Aabb::new_from_vec(center.eval(1.) - r_vec, center.eval(1.) + r_vec);
		Aabb::new_from_bbox(box1, box2)
	}
}

#[derive(Default, Copy, Clone)]
pub struct SceneHandle(u32);

#[derive(Debug, Clone)]
pub struct Scene {
	spheres: Vec<Sphere>,
}

impl Scene {
	pub fn new(spheres: Vec<Sphere>) -> Scene {
		Scene { spheres }
	}
}

fn ray_color(scene: &Scene, depth: u32, r0: &Ray) -> Vec3 {
	let mut rec: HitRecord = HitRecord::default();
	let mut r = *r0;
	let mut color = ONES;
	for _ in 0..depth {
		if !scene.hit(Interval::new(0.001, f32::MAX), &mut r, &mut rec) {
			let t = 0.5 * (r.dir.normalize().y + 1.);
			color = color * lerp(t, ONES, vec(0.75, 0.95, 1.0));
			break;
		}
		let (attenuation, scattered) = scatter(&r, &rec);
		r = scattered;
		color = color * attenuation;
	}
	color
}

impl Scene {
	fn hit(
		&self,
		mut interval: Interval,
		r: &mut Ray,
		rec: &mut HitRecord,
	) -> bool {
		let mut hit = false;
		for h in self.handles() {
			if self.hit_obj(h, interval, r, rec) {
				hit = true;
				interval.max = rec.t;
			}
		}
		hit
	}

	fn handles(&self) -> impl Iterator<Item = SceneHandle> {
		(0..self.spheres.len()).map(|i| SceneHandle(i as u32))
	}

	#[allow(non_snake_case)]
	fn hit_obj(
		&self,
		h: SceneHandle,
		interval: Interval,
		r: &Ray,
		rec: &mut HitRecord,
	) -> bool {
		let sphere = &self.spheres[h.0 as usize];
		let center = sphere.center.eval(r.time);
		let oc = center - r.origin;
		// Note:
		//
		//   (-b +- sqrt(b^2 - 4ac))/(2a)  =>  (h +- sqrt(h^2 - ac))/a
		//
		// through the substitution b = -2h.
		let a = r.dir.dot(r.dir);
		let h = oc.dot(r.dir);
		let c = oc.dot(oc) - sphere.radius * sphere.radius;
		let D = h * h - a * c;
		if D > 0. {
			for t in [(h - D.sqrt()) / a, (h + D.sqrt()) / a] {
				if interval.surrounds(t) {
					rec.t = t;
					rec.p = r.eval(t);
					let outward_normal = (rec.p - center) / sphere.radius;
					rec.set_face_normal(r, outward_normal);
					rec.mat = sphere.mat; // TODO: save a handle instead?
					return true;
				}
			}
		}
		false
	}
}

#[derive(Debug, Clone)]
pub struct Camera {
	image_width: u32,
	image_height: u32,
	samples_per_pixel: u32,
	max_depth: u32,
	defocus_angle: f32,
	center: Vec3,
	pixel_delta_u: Vec3,
	pixel_delta_v: Vec3,
	pixel_00_loc: Vec3,
	defocus_disk_u: Vec3,
	defocus_disk_v: Vec3,
}

impl Camera {
	pub fn new(cfg: &Config, ccfg: &CameraConfig) -> Self {
		let focus_dist = if ccfg.focus_dist != 0. {
			ccfg.focus_dist
		} else {
			(ccfg.look_from - ccfg.look_at).norm()
		};
		let aspect_ratio = (cfg.nx as f32) / (cfg.ny as f32);
		let image_width = cfg.nx;
		let image_height = cfg.ny;
		let samples_per_pixel = cfg.nsamples;
		let max_depth = cfg.max_depth;

		let defocus_angle = ccfg.defocus_angle;

		let center = ccfg.look_from;
		let theta = deg_to_rad(ccfg.fov);
		let h = (theta / 2.).tan();
		let viewport_height = 2. * h * focus_dist;
		let viewport_width = viewport_height * aspect_ratio;

		// Camera plane in uv coordinates, with w perpendicular to uv such that
		// "VUp" orients the camera rotation.
		let w = (ccfg.look_from - ccfg.look_at).normalize();
		let u = ccfg.v_up.cross(w);
		let v = w.cross(u);

		let viewport_u = u * viewport_width;
		let viewport_v = v * -viewport_height;
		let pixel_delta_u = viewport_u / (image_width as f32);
		let pixel_delta_v = viewport_v / (image_height as f32);
		let upper_left =
			center - w * focus_dist - viewport_u / 2. - viewport_v / 2.;
		let pixel_00_loc = upper_left + (pixel_delta_u + pixel_delta_v) * 2.;

		let defocus_radius = deg_to_rad(ccfg.defocus_angle / 2.).tan();
		let defocus_disk_u = u * defocus_radius;
		let defocus_disk_v = v * defocus_radius;

		Camera {
			image_width,
			image_height,
			samples_per_pixel,
			max_depth,
			defocus_angle,
			center,
			pixel_delta_u,
			pixel_delta_v,
			pixel_00_loc,
			defocus_disk_u,
			defocus_disk_v,
		}
	}

	fn sample_ray_around_ij(&self, i: u32, j: u32) -> Ray {
		let offset = sample_square();
		let pixel_sample = self.pixel_00_loc
			+ self.pixel_delta_u * (i as f32 + offset.x)
			+ self.pixel_delta_v * (j as f32 + offset.y);
		let origin = if self.defocus_angle > 0. {
			self.defocus_disk_sample()
		} else {
			self.center
		};
		Ray {
			origin,
			dir: pixel_sample - self.center,
			time: rand32(),
		}
	}

	fn defocus_disk_sample(&self) -> Vec3 {
		let p = random_in_unit_disk();
		self.center + self.defocus_disk_u * p.x + self.defocus_disk_v * p.y
	}
}

fn ray_color_at_ij(cam: &Camera, scene: &Scene, i: u32, j: u32) -> Vec3 {
	let mut color = Vec3::default();
	for _ in 0..cam.samples_per_pixel {
		let ray = cam.sample_ray_around_ij(i, j);
		color = color + ray_color(scene, cam.max_depth, &ray);
	}
	color = color / (cam.samples_per_pixel as f32);
	color
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
	let r0 = (1. - ref_idx) / (1. + ref_idx);
	let r0 = r0 * r0;
	r0 + (1. - r0) * (1. - cosine).powi(5)
}

fn refract(uv: Vec3, n: Vec3, cos_theta: f32, etai_over_etat: f32) -> Vec3 {
	let r_out_perp = (uv + n * cos_theta) * etai_over_etat;
	let perp_dot = r_out_perp.dot(r_out_perp);
	let r_out_par = n * (-f32::sqrt(1. - perp_dot));
	r_out_perp + r_out_par
}

fn scatter(r: &Ray, rec: &HitRecord) -> (Vec3, Ray) {
	use Material::*;
	match rec.mat {
		Matte { albedo } => {
			let mut scatter_dir = random_in_unit_ball() + rec.normal;
			// Guard against the random unit vector pointing opposite to the Normal
			if scatter_dir.near_zero() {
				scatter_dir = rec.normal;
			}
			let scattered = Ray {
				origin: rec.p,
				dir: scatter_dir,
				time: r.time,
			};
			(albedo, scattered)
		}
		Metal { albedo, fuzz } => {
			let reflected = r.dir.reflect(rec.normal);
			let dir = reflected.normalize() + random_in_unit_ball() * fuzz;
			let scattered = if dir.dot(rec.normal) <= 0. {
				*r
			} else {
				Ray {
					origin: rec.p,
					dir,
					time: r.time,
				}
			};
			(albedo, scattered)
		}
		Dielectric { ref_idx } => {
			let ref_idx = if rec.front_face {
				1. / ref_idx
			} else {
				ref_idx
			};
			let unit_dir = r.dir.normalize();
			let cos_theta = f32::min(-unit_dir.dot(rec.normal), 1.);
			let sin_theta = (1. - cos_theta * cos_theta).sqrt();
			let cannot_refract = ref_idx * sin_theta > 1.;
			let dir = if cannot_refract || rand32() < schlick(cos_theta, ref_idx) {
				unit_dir.reflect(rec.normal)
			} else {
				refract(unit_dir, rec.normal, cos_theta, ref_idx)
			};
			let scattered = Ray {
				origin: rec.p,
				dir,
				time: r.time,
			};
			(ONES, scattered)
		}
	}
}

#[cfg(not(feature = "gui"))]
pub fn camera_default(cfg: &Config) -> Camera {
	let cam_cfg = CameraConfig {
		fov: 20.,
		look_from: vec(10., 2.5, 5.),
		look_at: vec(-4., 0., -2.),
		v_up: vec(0., 1., 0.),
		defocus_angle: 1.,
		focus_dist: 0.,
	};

	Camera::new(cfg, &cam_cfg)
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Interval {
	pub min: f32,
	pub max: f32,
}

impl Interval {
	pub fn new(min: f32, max: f32) -> Self {
		Self { min, max }
	}

	pub fn new_unordered(a: f32, b: f32) -> Self {
		Self {
			min: a.min(b),
			max: a.max(b),
		}
	}

	pub fn new_union(a: Self, b: Self) -> Self {
		Self {
			min: f32::min(a.min, b.min),
			max: f32::max(a.max, b.max),
		}
	}

	pub fn size(&self) -> f32 {
		self.max - self.min
	}

	pub fn contains(&self, x: f32) -> bool {
		self.min <= x && x <= self.max
	}

	pub fn surrounds(&self, x: f32) -> bool {
		self.min < x && x < self.max
	}

	pub fn clamp(&self, x: f32) -> f32 {
		f32::min(self.max, f32::max(self.min, x))
	}

	pub fn expand(&self, delta: f32) -> Self {
		Self {
			min: self.min - delta / 2.,
			max: self.max + delta / 2.,
		}
	}
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Aabb {
	pub x: Interval,
	pub y: Interval,
	pub z: Interval,
}

impl Aabb {
	pub fn new_from_vec(a: Vec3, b: Vec3) -> Self {
		Self {
			x: Interval::new_unordered(a.x, b.x),
			y: Interval::new_unordered(a.y, b.y),
			z: Interval::new_unordered(a.z, b.z),
		}
	}

	pub fn new_from_bbox(b0: Self, b1: Self) -> Self {
		Self {
			x: Interval::new_union(b0.x, b1.x),
			y: Interval::new_union(b0.y, b1.y),
			z: Interval::new_union(b0.z, b1.z),
		}
	}

	pub fn axis_interval(&self, axis: i32) -> Interval {
		match axis {
			0 => self.x,
			1 => self.y,
			2 => self.z,
			_ => unreachable!("don't do this"),
		}
	}
}

#[derive(Debug)]
pub struct BvhTree {
	pub scene: Scene,
}

#[derive(Default, Debug, Copy, Clone)]
pub struct Handle(u32);

#[derive(Debug, Copy, Clone)]
struct Node {
	left: Handle,
	right: Handle,
}

impl BvhTree {
	fn new(scene: Scene) -> Self {
		BvhTree { scene }
	}
}
