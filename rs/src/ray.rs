use std::{ffi::OsStr, fs::File, io::Write, path::PathBuf};

use crate::vec::*;

#[derive(Debug)]
pub struct Args {
	pub nsamples: u32,
	pub threads: u32,
	pub cam: Camera,
	pub scene: Scene,
	pub nx: u32,
	pub ny: u32,
	pub depth: u32,
}

pub struct RenderArgs<'a> {
	pub buf: &'a mut [u8],
	pub cam: &'a Camera,
	pub scene: &'a Scene,
	pub nsamples: u32,
	pub nx: u32,
	pub ny: u32,
	pub depth: u32,
	pub ymin: f32,
	pub ymax: f32,
}

pub struct Image {
	pub buf: Vec<u8>,
	pub nx: u32,
	pub ny: u32,
}

fn rand32() -> f32 {
	rand::random()
}

static PI: f32 = std::f32::consts::PI;

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

pub fn raytrace(args: &Args, nx: u32, ny: u32) -> Image {
	let nsamples = args.nsamples;
	let threads = args.threads;
	if threads == 0 || nsamples == 0 {
		panic!("number of samples and threads must be positive");
	}

	let mut buf = vec![0; (3 * nx * ny) as usize];
	std::thread::scope(|s| {
		let mut ny_pos: u32 = 0;
		let mut bufpos: usize = 0;
		for i in 0..threads {
			let ny_remaining = ny - ny_pos;
			let ny_th = ny_remaining / (threads - i);
			let len_th = 3 * nx * ny_th;
			let ymax = (ny_remaining) as f32 / ny as f32;
			let ymin = (ny_remaining - ny_th) as f32 / ny as f32;
			let (cam, sc) = (&args.cam, &args.scene);
			// Couldn't find a way to do different sized chunks with buf.chunks_mut
			// so did the split manually instead.
			let bufchunk = unsafe {
				&mut *std::ptr::slice_from_raw_parts_mut(
					buf.as_mut_ptr().add(bufpos),
					len_th as usize,
				)
			};
			let render_args = RenderArgs {
				buf: bufchunk,
				cam,
				scene: sc,
				nsamples,
				nx,
				ny: ny_th,
				depth: args.depth,
				ymin,
				ymax,
			};
			s.spawn(move || {
				render(render_args);
			});
			ny_pos += ny_th;
			bufpos += len_th as usize;
		}
	});

	Image { buf, nx, ny }
}

#[allow(clippy::identity_op)]
fn render(args: RenderArgs) {
	let RenderArgs {
		buf,
		cam,
		scene,
		nsamples,
		nx,
		ny,
		depth,
		ymin,
		ymax,
	} = args;
	assert_eq!(buf.len(), (3 * nx * ny) as usize);
	let yheight = ymax - ymin;
	let mut bi = 0;
	for j in (0..ny).rev() {
		for i in 0..nx {
			let mut color = Vec3::default();
			for _ in 0..nsamples {
				let x = (i as f32 + rand32()) / nx as f32;
				let y = ymin + yheight * (j as f32 + rand32()) / ny as f32;
				let r = cam.ray_at(x, y);
				color = color + scene.color(depth, &r);
			}
			color = (color / nsamples as f32).sqrt();
			buf[bi + 0] = (255. * color.x) as u8;
			buf[bi + 1] = (255. * color.y) as u8;
			buf[bi + 2] = (255. * color.z) as u8;
			bi += 3;
		}
	}
}

fn ppm_write(f: &mut File, buf: &[u8], x: u32, y: u32) -> Result<()> {
	f.write_all(format!("P6\n{} {} 255\n", x, y).as_bytes())?;
	f.write_all(buf)?;
	Ok(())
}

#[derive(Clone, Copy)]
struct Ray {
	origin: Vec3,
	dir: Vec3,
}

impl Ray {
	fn eval(self, t: f32) -> Vec3 {
		self.origin + self.dir * t
	}
}

#[derive(Copy, Clone, Debug)]
enum Material {
	Matte { albedo: Vec3 },
	Metal { albedo: Vec3, fuzz: f32 },
	Dielectric { ref_idx: f32 },
}

struct HitRecord {
	t: f32,
	p: Vec3,
	normal: Vec3,
	mat: Material,
}

impl Default for HitRecord {
	fn default() -> Self {
		HitRecord {
			t: 0.,
			p: Vec3::default(),
			normal: Vec3::default(),
			mat: Material::Matte {
				albedo: Vec3::default(),
			},
		}
	}
}

#[derive(Debug, Clone)]
struct Sphere {
	center: Vec3,
	radius: f32,
	mat: Material,
}

impl Sphere {
	#[allow(non_snake_case)]
	fn hit(&self, interval: Interval, r: &Ray, rec: &mut HitRecord) -> bool {
		let oc = r.origin - self.center;
		let a = r.dir.dot(r.dir);
		let b = oc.dot(r.dir);
		let c = oc.dot(oc) - self.radius * self.radius;
		let D = b * b - a * c;
		if D > 0. {
			for t in [(-b - D.sqrt()) / a, (-b + D.sqrt()) / a] {
				if interval.surrounds(t) {
					rec.t = t;
					rec.p = r.eval(t);
					rec.normal = (rec.p - self.center) / self.radius;
					rec.mat = self.mat;
					return true;
				}
			}
		}
		false
	}
}

#[derive(Debug, Clone)]
pub struct Scene {
	spheres: Vec<Sphere>,
}

impl Scene {
	fn color(&self, depth: u32, r0: &Ray) -> Vec3 {
		let mut rec: HitRecord = HitRecord::default();
		let mut r = *r0;
		let mut color = ONES;
		for _ in 0..depth {
			if !self.hit(Interval::new(0.001, f32::MAX), &mut r, &mut rec) {
				let t = 0.5 * (r.dir.normalize().y + 1.);
				color = color * (vec(0.75, 0.95, 1.0) * t + ONES * (1. - t));
				break;
			}
			let (attenuation, scattered) = scatter(&r, rec.p, rec.normal, rec.mat);
			r = scattered;
			color = color * attenuation;
		}
		color
	}

	fn hit(
		&self,
		mut interval: Interval,
		r: &mut Ray,
		rec: &mut HitRecord,
	) -> bool {
		let mut hit = false;
		for sphere in &self.spheres {
			if sphere.hit(interval, r, rec) {
				hit = true;
				interval.max = rec.t;
			}
		}
		hit
	}
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Camera {
	lower_left_corner: Vec3,
	horiz: Vec3,
	vert: Vec3,
	origin: Vec3,
	u: Vec3,
	v: Vec3,
	w: Vec3,
	lens_radius: f32,
}

impl Camera {
	pub fn new(
		look_from: Vec3,
		look_at: Vec3,
		v_up: Vec3,
		vfov: f32,
		aspect: f32,
		aperture: f32,
		focus_dist: f32,
	) -> Self {
		let theta = vfov * PI / 180.;
		let half_height = (theta / 2.).tan();
		let half_width = aspect * half_height;
		let w = (look_from - look_at).normalize();
		let u = v_up.cross(w).normalize();
		let v = w.cross(u);
		let lower_left_corner = look_from
			- u * focus_dist * half_width
			- v * focus_dist * half_height
			- w * focus_dist;

		Camera {
			lower_left_corner,
			horiz: u * 2. * half_width * focus_dist,
			vert: v * 2. * half_height * focus_dist,
			origin: look_from,
			u,
			v,
			w,
			lens_radius: aperture / 2.,
		}
	}

	fn ray_at(&self, x: f32, y: f32) -> Ray {
		let rd = random_in_unit_ball() * self.lens_radius;
		let offset = self.u * rd.x + self.v * rd.y;
		let dir = self.lower_left_corner + (self.horiz * x + self.vert * y)
			- self.origin
			- offset;
		Ray {
			origin: self.origin + offset,
			dir,
		}
	}
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
	let r0 = (1. - ref_idx) / (1. + ref_idx);
	let r0 = r0 * r0;
	r0 + (1. - r0) * (1. - cosine).powi(5)
}

#[allow(non_snake_case)]
fn refract(
	u: Vec3,
	normal: Vec3,
	ni_over_nt: f32,
	refracted: &mut Vec3,
) -> bool {
	let un = u.normalize();
	let dt = un.dot(normal);
	let D = 1. - ni_over_nt * ni_over_nt * (1. - dt * dt);
	if D > 0. {
		let v = (un - normal * dt) * ni_over_nt - normal * D.sqrt();
		*refracted = v;
		return true;
	}
	false
}

fn scatter(r: &Ray, p: Vec3, normal: Vec3, mat: Material) -> (Vec3, Ray) {
	use Material::*;
	match mat {
		Matte { albedo } => {
			let target = p + normal + random_in_unit_ball();
			let scattered = Ray {
				origin: p,
				dir: target - p,
			};
			(albedo, scattered)
		}
		Metal { albedo, fuzz } => {
			let reflected = r.dir.normalize().reflect(normal);
			let dir = reflected + random_in_unit_ball() * fuzz;
			let scattered: Ray = if dir.dot(normal) > 0. {
				Ray { origin: p, dir }
			} else {
				*r
			};
			(albedo, scattered)
		}
		Dielectric { ref_idx } => {
			let d = r.dir.dot(normal);
			let outward_normal: Vec3;
			let ni_over_nt: f32;
			let cosine: f32;
			if d > 0. {
				outward_normal = -normal;
				ni_over_nt = ref_idx;
				cosine = ref_idx * d / r.dir.norm();
			} else {
				outward_normal = normal;
				ni_over_nt = 1. / ref_idx;
				cosine = -d / r.dir.norm();
			}
			let mut dir = Vec3::default();
			if !refract(r.dir, outward_normal, ni_over_nt, &mut dir)
				|| rand32() < schlick(cosine, ref_idx)
			{
				dir = reflect(r.dir, normal);
			}
			let scattered = Ray { origin: p, dir };
			(ONES, scattered)
		}
	}
}

#[cfg(not(feature = "gui"))]
pub fn camera_default(nx: u32, ny: u32) -> Camera {
	let look_from = vec(10., 2.5, 5.);
	let look_at = vec(-4., 0., -2.);
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

pub fn small_scene() -> Scene {
	let nspheres = 3 + 360 / 15;
	let mut spheres = Vec::with_capacity(nspheres);

	spheres.push(Sphere {
		center: vec(0., -1000., 0.),
		radius: 1000.,
		mat: Material::Matte {
			albedo: vec(0.88, 0.96, 0.7),
		},
	});
	spheres.push(Sphere {
		center: vec(1.5, 1., 0.),
		radius: 1.,
		mat: Material::Dielectric { ref_idx: 1.5 },
	});
	spheres.push(Sphere {
		center: vec(-1.5, 1., 0.),
		radius: 1.,
		mat: Material::Metal {
			albedo: vec(0.8, 0.9, 0.8),
			fuzz: 0.,
		},
	});

	for deg in (0..360).step_by(15) {
		let x = ((deg as f32) * PI / 180.).sin();
		let z = ((deg as f32) * PI / 180.).cos();
		let r0 = 3.;
		let r1 = 0.33 + x * z / 9.;
		spheres.push(Sphere {
			center: vec(r0 * x, r1, r0 * z),
			radius: r1,
			mat: Material::Matte {
				albedo: vec(x, 0.5 + x * z / 2., z),
			},
		});
	}

	Scene { spheres }
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
