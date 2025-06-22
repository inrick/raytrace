use std::cmp::Ordering;
use std::num::NonZero;
use std::{ffi::OsStr, fs::File, io::Write, path::PathBuf};

use crate::math::{deg_to_rad, rand32};
use crate::vec::*;

#[derive(Debug)]
pub struct Config {
	pub nsamples: NonZero<u32>,
	pub threads: u32,
	pub nx: NonZero<u32>,
	pub ny: NonZero<u32>,
	pub max_depth: NonZero<u32>,
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

struct ThreadArgs<'a, T: Hittable> {
	buf: &'a mut [u8],
	cam: &'a Camera,
	scene: &'a T,
	y0: u32,
	y1: u32,
}

pub struct Image {
	pub buf: Vec<u8>,
	pub nx: NonZero<u32>,
	pub ny: NonZero<u32>,
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
	image_writer(&mut f, &img.buf, img.nx.get(), img.ny.get())
}

pub fn raytrace(args: &Args) -> Image {
	let Config {
		nsamples: _,
		mut threads,
		nx,
		ny,
		max_depth: _,
	} = args.cfg;
	if threads == 0 {
		threads = match std::thread::available_parallelism() {
			Ok(n) => 2 * n.get() as u32,
			_ => 1,
		};
	};
	let cam = &args.cam;
	//let scene = &args.scene;
	let bvh = BvhTree::new(&args.scene);
	let scene = &bvh;

	let mut buf: Vec<u8> =
		vec![0; (3 * cam.image_width * cam.image_height) as usize];
	std::thread::scope(|s| {
		let mut ypos = 0u32;
		let mut buf = buf.as_mut_slice();
		for i in 0..threads {
			let chunk_rows = (cam.image_height - ypos) / (threads - i);
			let chunk_size = (3 * cam.image_width * chunk_rows) as usize;
			let (buf_thread, buf_next) = buf.split_at_mut(chunk_size);
			buf = buf_next;
			let render_args = ThreadArgs {
				buf: buf_thread,
				cam,
				scene,
				y0: ypos,
				y1: ypos + chunk_rows,
			};
			s.spawn(move || {
				render(render_args);
			});
			ypos += chunk_rows;
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
fn render<T: Hittable>(args: ThreadArgs<T>) {
	let ThreadArgs {
		buf,
		cam,
		scene,
		y0,
		y1,
	} = args;
	let mut pos = 0;
	for j in y0..y1 {
		for i in 0..cam.image_width {
			let color = ray_color_at_ij(cam, scene, i, j);
			buf[pos + 0] = (255. * linear_to_gamma(color.x)) as u8;
			buf[pos + 1] = (255. * linear_to_gamma(color.y)) as u8;
			buf[pos + 2] = (255. * linear_to_gamma(color.z)) as u8;
			pos += 3;
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

trait Hittable {
	fn hit(&self, interval: Interval, r: &Ray, rec: &mut HitRecord) -> bool;
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct SceneHandle(u32);

#[derive(Debug, Clone)]
pub struct Scene {
	spheres: Vec<Sphere>,
}

impl Scene {
	pub fn new(spheres: Vec<Sphere>) -> Scene {
		Scene { spheres }
	}

	pub fn comparator_axis<'a>(
		&'a self,
		axis: i32,
	) -> impl FnMut(&SceneHandle, &SceneHandle) -> Ordering + 'a {
		move |h0, h1| {
			let a = self.obj_bbox(*h0);
			let b = self.obj_bbox(*h1);
			bbox_compare(a, b, axis)
		}
	}

	pub fn obj_bbox(&self, h: SceneHandle) -> Aabb {
		self[h].bounding_box()
	}

	fn handles(&self) -> impl Iterator<Item = SceneHandle> {
		(0..self.spheres.len()).map(|i| SceneHandle(i as u32))
	}

	fn hit_obj_write_rec(
		&self,
		h: SceneHandle,
		interval: Interval,
		r: &Ray,
		rec: &mut HitRecord,
	) -> bool {
		match self.hit_obj(h, interval, r) {
			Some(t) => {
				let sphere = &self[h];
				let center = sphere.center.eval(r.time);
				rec.t = t;
				rec.p = r.eval(t);
				let outward_normal = (rec.p - center) / sphere.radius;
				rec.set_face_normal(r, outward_normal);
				rec.mat = sphere.mat;
				true
			}
			None => false,
		}
	}

	#[allow(non_snake_case)]
	fn hit_obj(
		&self,
		h: SceneHandle,
		interval: Interval,
		r: &Ray,
	) -> Option<f32> {
		let sphere = &self[h];
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
					return Some(t);
				}
			}
		}
		None
	}
}

impl Hittable for Scene {
	fn hit(&self, mut interval: Interval, r: &Ray, rec: &mut HitRecord) -> bool {
		let mut hit: Option<(f32, SceneHandle)> = None;
		for h in self.handles() {
			if let Some(t) = self.hit_obj(h, interval, r) {
				hit = Some((t, h));
				interval.max = t;
			}
		}
		if let Some((t, h)) = hit {
			let sphere = &self[h];
			let center = sphere.center.eval(r.time);
			rec.t = t;
			rec.p = r.eval(t);
			let outward_normal = (rec.p - center) / sphere.radius;
			rec.set_face_normal(r, outward_normal);
			rec.mat = sphere.mat;
			true
		} else {
			false
		}
	}
}

pub fn bbox_compare(a: Aabb, b: Aabb, axis: i32) -> Ordering {
	let ia = a.axis(axis);
	let ib = b.axis(axis);
	if ia.min < ib.min {
		Ordering::Less
	} else if ia.min > ib.min {
		Ordering::Greater
	} else {
		Ordering::Equal
	}
}

fn ray_color<T: Hittable>(scene: &T, max_depth: u32, mut r: Ray) -> Vec3 {
	let mut rec: HitRecord = HitRecord::default();
	let mut color = ONES;
	for _ in 0..max_depth {
		if !scene.hit(Interval::new(0.001, f32::MAX), &r, &mut rec) {
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

impl std::ops::Index<SceneHandle> for Scene {
	type Output = Sphere;

	fn index(&self, index: SceneHandle) -> &Self::Output {
		&self.spheres[index.0 as usize]
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
		let image_width = cfg.nx.get();
		let image_height = cfg.ny.get();
		let aspect_ratio = (image_width as f32) / (image_height as f32);
		let samples_per_pixel = cfg.nsamples.get();
		let max_depth = cfg.max_depth.get();

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

	fn sample_ray_around_ij(&self, time: f32, i: u32, j: u32) -> Ray {
		let offset = sample_square();
		let pixel_sample = self.pixel_00_loc
			+ self.pixel_delta_u * (i as f32 + offset.x)
			+ self.pixel_delta_v * (j as f32 + offset.y);
		let origin = if self.defocus_angle > 0. {
			self.defocus_disk_sample()
		} else {
			self.center
		};
		let dir = pixel_sample - self.center;
		Ray { origin, dir, time }
	}

	fn defocus_disk_sample(&self) -> Vec3 {
		let p = random_in_unit_disk();
		self.center + self.defocus_disk_u * p.x + self.defocus_disk_v * p.y
	}
}

fn ray_color_at_ij<T: Hittable>(
	cam: &Camera,
	scene: &T,
	i: u32,
	j: u32,
) -> Vec3 {
	let mut color = Vec3::default();
	for _ in 0..cam.samples_per_pixel {
		let time = rand32();
		let ray = cam.sample_ray_around_ij(time, i, j);
		color = color + ray_color(scene, cam.max_depth, ray);
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
	pub const EMPTY: Self = Self {
		min: f32::MAX,
		max: f32::MIN,
	};

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

#[derive(Debug, Copy, Clone)]
pub struct Aabb {
	pub x: Interval,
	pub y: Interval,
	pub z: Interval,
}

impl Aabb {
	pub const EMPTY: Self = Self {
		x: Interval::EMPTY,
		y: Interval::EMPTY,
		z: Interval::EMPTY,
	};

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

	pub fn axis(&self, index: i32) -> Interval {
		match index {
			0 => self.x,
			1 => self.y,
			2 => self.z,
			_ => unreachable!("don't do this"),
		}
	}

	pub fn hit(&self, mut ray_t: Interval, ray: &Ray) -> bool {
		for axis in 0..3 {
			let ax = self.axis(axis);
			let adinv = 1. / ray.dir.axis(axis);

			let v = ray.origin.axis(axis);
			let t0 = (ax.min - v) * adinv;
			let t1 = (ax.max - v) * adinv;

			let (t0, t1) = (t0.min(t1), t0.max(t1));
			ray_t.min = ray_t.min.max(t0);
			ray_t.max = ray_t.max.min(t1);

			if ray_t.max <= ray_t.min {
				return false;
			}
		}
		true
	}

	pub fn longest_axis(&self) -> i32 {
		if self.x.size() > self.y.size() {
			if self.x.size() > self.z.size() {
				0
			} else {
				2
			}
		} else {
			if self.y.size() > self.z.size() {
				1
			} else {
				2
			}
		}
	}
}

impl std::ops::Index<i32> for Aabb {
	type Output = Interval;

	fn index(&self, index: i32) -> &Self::Output {
		match index {
			0 => &self.x,
			1 => &self.y,
			2 => &self.z,
			_ => unreachable!("don't do this"),
		}
	}
}

#[derive(Debug)]
pub struct BvhTree<'a> {
	start: Handle,
	scene: &'a Scene,
	nodes: Vec<BvhNode>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Handle {
	Empty,
	BvhHandle(u32),
	SceneHandle(SceneHandle),
}

#[derive(Debug, Copy, Clone)]
struct BvhNode {
	left: Handle,
	right: Handle,
	bbox: Aabb,
}

impl<'a> BvhTree<'a> {
	fn new(scene: &'a Scene) -> BvhTree<'a> {
		let mut bvh = Self {
			start: Handle::Empty,
			scene,
			nodes: Vec::new(),
		};
		let mut handles: Vec<SceneHandle> = scene.handles().collect();
		bvh.start = bvh.add_node(&mut handles);
		bvh
	}

	fn add_node(&mut self, handles: &mut [SceneHandle]) -> Handle {
		match handles {
			&mut [] => Handle::Empty,
			&mut [h] => Handle::SceneHandle(h),
			_ => {
				let mut bbox = Aabb::EMPTY;
				for &h in handles.iter() {
					bbox = Aabb::new_from_bbox(bbox, self.scene.obj_bbox(h));
				}
				let axis = bbox.longest_axis();
				let comparator = self.scene.comparator_axis(axis);
				handles.sort_by(comparator);

				let mid = handles.len() / 2;
				let (handles_left, handles_right) = handles.split_at_mut(mid);
				let left = self.add_node(handles_left);
				let right = self.add_node(handles_right);
				self.push_node(BvhNode { left, right, bbox })
			}
		}
	}

	fn push_node(&mut self, node: BvhNode) -> Handle {
		self.nodes.push(node);
		Handle::BvhHandle(self.nodes.len() as u32 - 1)
	}

	fn hit_rec(
		&self,
		h: Handle,
		mut interval: Interval,
		r: &Ray,
		rec: &mut HitRecord,
	) -> bool {
		match h {
			Handle::Empty => false,
			Handle::BvhHandle(h) => {
				let node = self.nodes[h as usize];
				node.bbox.hit(interval, r) && {
					let hit_left = self.hit_rec(node.left, interval, r, rec);
					if hit_left {
						interval.max = rec.t;
					}
					let right_is_different = node.left != node.right;
					let hit_right =
						right_is_different && self.hit_rec(node.right, interval, r, rec);
					hit_left || hit_right
				}
			}
			Handle::SceneHandle(h) => {
				self.scene.hit_obj_write_rec(h, interval, r, rec)
			}
		}
	}
}

impl Hittable for BvhTree<'_> {
	fn hit(&self, interval: Interval, r: &Ray, rec: &mut HitRecord) -> bool {
		self.hit_rec(self.start, interval, r, rec)
	}
}
