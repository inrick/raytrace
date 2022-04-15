mod vec;
use vec::*;

use std::vec::Vec as Array;
use std::vec as array;
use std::io::Write;
use std::io;
use std::fs::File;

fn rand32() -> f32 {
  rand::random()
}

static PI: f32 = std::f32::consts::PI;

fn main() {
  let mut f = File::create("out.ppm").unwrap();
  if let Err(err) = run(&mut f, 10, 600, 300) {
      eprintln!("ERROR: {}", err);
  }
}

fn run(f: &mut File, nsamples: i32, nx: i32, ny: i32) -> io::Result<()> {
  let look_from = vec(10., 2.5, 5.);
  let look_at = vec(-4., 0., -2.);
  let dist_to_focus = norm(look_from - look_at);
  let aperture = 0.05;

  let (nxf, nyf) = (nx as f32, ny as f32);

  let cam = Camera::new(
    look_from, look_at, vec(0., 1., 0.),
    20., nxf/nyf, aperture, dist_to_focus,
  );

  let sc = small_scene();

  let mut buf = array![0; (3*nx*ny) as usize];
  let mut bi = 0;
  for j in (0..ny).rev() {
    for i in 0..nx {
      let mut color = Vec::default();
      for _ in 0..nsamples {
        let x = (i as f32 + rand32()) / nxf;
        let y = (j as f32 + rand32()) / nyf;
        let r = cam.ray_at(x, y);
        color = color + sc.color(&r);
      }
      color = (color / nsamples as f32).sqrt();
      buf[bi+0] = (255.*color.x) as u8;
      buf[bi+1] = (255.*color.y) as u8;
      buf[bi+2] = (255.*color.z) as u8;
      bi += 3;
    }
  }

  ppm_write(f, &buf, nx, ny)?;
  Ok(())
}

fn ppm_write(f: &mut File, buf: &Array<u8>, x: i32, y: i32) -> io::Result<()> {
  f.write(format!("P6\n{} {} 255\n", x, y).as_bytes())?;
  f.write_all(buf)?;
  Ok(())
}

#[derive(Clone, Copy)]
struct Ray {
  origin: Vec, dir: Vec
}

impl Ray {
  fn eval(self, t: f32) -> Vec {
    self.origin + self.dir*t
  }
}

#[derive(Copy, Clone)]
enum Material {
  Matte      { albedo: Vec },
  Metal      { albedo: Vec, fuzz: f32 },
  Dielectric { ref_idx: f32 },
}

struct HitRecord {
  t:      f32,
  p:      Vec,
  normal: Vec,
  mat:    Material,
}

impl Default for HitRecord {
  fn default() -> Self {
    HitRecord{
      t: 0.,
      p: Vec::default(),
      normal: Vec::default(),
      mat: Material::Matte{albedo: Vec::default()},
    }
  }
}

struct Sphere {
  center: Vec,
  radius: f32,
  mat:    Material,
}

impl Sphere {
  #[allow(non_snake_case)]
  fn hit(&self, tmin: f32, tmax: f32, r: &Ray, rec: &mut HitRecord) -> bool {
    let oc = r.origin - self.center;
    let a = dot(r.dir, r.dir);
    let b = dot(oc, r.dir);
    let c = dot(oc, oc) - self.radius*self.radius;
    let D = b*b - a*c;
    if D > 0. {
      for t in [(-b - D.sqrt()) / a, (-b + D.sqrt()) / a] {
        if tmin < t && t < tmax {
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

struct Scene {
  spheres: Array<Sphere>,
}

impl Scene {
  fn color(&self, r0: &Ray) -> Vec {
    let mut rec: HitRecord = HitRecord::default();
    let mut r = *r0;
    let mut color = ONES;
    for _depth in 0..50 {
      if !self.hit(0.001, f32::MAX, &mut r, &mut rec) {
        let t = 0.5 * (normalize(r.dir).y + 1.);
        color = color * (vec(0.75, 0.95, 1.0)*t + ONES*(1.-t));
        break;
      }
      let (attenuation, scattered) = scatter(&r, rec.p, rec.normal, rec.mat);
      r = scattered;
      color = color * attenuation;
    }
    color
  }

  fn hit(&self, tmin: f32, tmax: f32, r: &mut Ray, rec: &mut HitRecord) -> bool {
    let mut hit = false;
    let mut closest = tmax;
    for sph in &self.spheres {
      if sph.hit(tmin, closest, r, rec) {
        hit = true;
        closest = rec.t;
      }
    }
    hit
  }
}

#[allow(dead_code)]
struct Camera {
  lower_left_corner: Vec,
  horiz:             Vec,
  vert:              Vec,
  origin:            Vec,
  u:                 Vec,
  v:                 Vec,
  w:                 Vec,
  lens_radius:       f32,
}

impl Camera {
  fn new(
    look_from: Vec, look_at: Vec, v_up: Vec,
    vfov: f32, aspect: f32, aperture: f32, focus_dist: f32,
  ) -> Self {
    let theta = vfov * PI / 180.;
    let half_height = (theta / 2.).tan();
    let half_width = aspect*half_height;
    let w = normalize(look_from - look_at);
    let u = normalize(cross(v_up, w));
    let v = cross(w, u);
    let lower_left_corner =
      look_from
      - u*focus_dist*half_width
      - v*focus_dist*half_height
      - w*focus_dist;

    Camera{
      lower_left_corner,
      horiz: u*2.*half_width*focus_dist,
      vert: v*2.*half_height*focus_dist,
      origin: look_from,
      u, v, w,
      lens_radius: aperture / 2.,
    }
  }

  fn ray_at(&self, x: f32, y: f32) -> Ray {
    let rd = random_in_unit_ball() * self.lens_radius;
    let offset = self.u*rd.x + self.v*rd.y;
    let dir =
      self.lower_left_corner
      + (self.horiz*x + self.vert*y)
      - self.origin - offset;
    Ray{origin: self.origin + offset, dir}
  }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
  let r0 = (1.-ref_idx)/(1.+ref_idx);
  let r0 = r0*r0;
  r0 + (1.-r0)*(1.-cosine).powi(5)
}

#[allow(non_snake_case)]
fn refract(u: Vec, normal: Vec, ni_over_nt: f32, refracted: &mut Vec) -> bool {
  let un = normalize(u);
  let dt = dot(un, normal);
  let D = 1. - ni_over_nt*ni_over_nt*(1.-dt*dt);
  if D > 0. {
    let v = (un - normal*dt)*ni_over_nt - normal*D.sqrt();
    *refracted = v;
    return true;
  }
  false
}

fn scatter(r: &Ray, p: Vec, normal: Vec, mat: Material) -> (Vec, Ray) {
  use Material::*;
  match mat {
    Matte{albedo} => {
      let target = p + normal + random_in_unit_ball();
      let scattered = Ray{origin: p, dir: target-p};
      (albedo, scattered)
    }
    Metal{albedo, fuzz} => {
      let reflected = reflect(normalize(r.dir), normal);
      let dir = reflected + random_in_unit_ball()*fuzz;
      let scattered: Ray;
      if dot(dir, normal) > 0. {
        scattered = Ray{origin: p, dir};
      } else {
        scattered = *r;
      }
      (albedo, scattered)
    }
    Dielectric{ref_idx} => {
      let d = dot(r.dir, normal);
      let outward_normal: Vec;
      let ni_over_nt: f32;
      let cosine: f32;
      if d > 0. {
        outward_normal = -normal;
        ni_over_nt = ref_idx;
        cosine = ref_idx * d / norm(r.dir);
      } else {
        outward_normal = normal;
        ni_over_nt = 1. / ref_idx;
        cosine = -d / norm(r.dir);
      }
      let mut dir = Vec::default();
      if !refract(r.dir, outward_normal, ni_over_nt, &mut dir)
        || rand32() < schlick(cosine, ref_idx) {
        dir = reflect(r.dir, normal);
      }
      let scattered = Ray{origin: p, dir};
      (ONES, scattered)
    }
  }
}

fn small_scene() -> Scene {
  let nspheres = 3 + 360/15;
  let mut spheres = Array::with_capacity(nspheres);

  spheres.push(Sphere{
    center: vec(0., -1000., 0.),
    radius: 1000.,
    mat: Material::Matte{albedo: vec(0.88, 0.96, 0.7)},
  });
  spheres.push(Sphere{
    center: vec(1.5, 1., 0.),
    radius: 1.,
    mat: Material::Dielectric{ref_idx: 1.5},
  });
  spheres.push(Sphere{
    center: vec(-1.5, 1., 0.),
    radius: 1.,
    mat: Material::Metal{albedo: vec(0.8, 0.9, 0.8), fuzz: 0.},
  });

  for deg in (0..360).step_by(15) {
    let x = ((deg as f32) * PI / 180.).sin();
    let z = ((deg as f32) * PI / 180.).cos();
    let r0 = 3.;
    let r1 = 0.33 + x*z/9.;
    spheres.push(Sphere{
      center: vec(r0*x, r1, r0*z),
      radius: r1,
      mat: Material::Matte{albedo: vec(x, 0.5+x*z/2., z)},
    });
  }

  Scene{spheres}
}
