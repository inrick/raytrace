use std::ops::{Add, Mul, Sub, Div, Neg};

#[derive(Default, Debug, Clone, Copy)]
pub struct Vec {
  pub x: f32, pub y: f32, pub z: f32
}

pub static ONES: Vec = Vec{x: 1., y: 1., z: 1.};

pub fn vec(x: f32, y: f32, z: f32) -> Vec {
  Vec{x, y, z}
}

impl Add for Vec {
  type Output = Self;
  fn add(self, v: Self) -> Self {
    vec(self.x+v.x, self.y+v.y, self.z+v.z)
  }
}

impl Sub for Vec {
  type Output = Self;
  fn sub(self, v: Self) -> Self {
    vec(self.x-v.x, self.y-v.y, self.z-v.z)
  }
}

impl Mul for Vec {
  type Output = Self;
  fn mul(self, v: Self) -> Self {
    vec(self.x*v.x, self.y*v.y, self.z*v.z)
  }
}

impl Mul<f32> for Vec {
  type Output = Self;
  fn mul(self, k: f32) -> Self {
    vec(self.x*k, self.y*k, self.z*k)
  }
}

impl Div for Vec {
  type Output = Self;
  fn div(self, v: Self) -> Self {
    vec(self.x/v.x, self.y/v.y, self.z/v.z)
  }
}

impl Div<f32> for Vec {
  type Output = Self;
  fn div(self, k: f32) -> Self {
    vec(self.x/k, self.y/k, self.z/k)
  }
}

impl Neg for Vec {
  type Output = Self;
  fn neg(self) -> Self {
    vec(-self.x, -self.y, -self.z)
  }
}

impl Vec {
  pub fn sqrt(self) -> Vec {
    vec(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
  }
}


pub fn cross(u: Vec, v: Vec) -> Vec {
  vec(
    u.y*v.z - u.z*v.y,
    u.z*v.x - u.x*v.z,
    u.x*v.y - u.y*v.x,
  )
}

pub fn dot(u: Vec, v: Vec) -> f32 {
  u.x*v.x + u.y*v.y + u.z*v.z
}

pub fn norm(u: Vec) -> f32 {
  dot(u, u).sqrt()
}

pub fn normalize(u: Vec) -> Vec {
  u/norm(u)
}

pub fn reflect(u: Vec, normal: Vec) -> Vec {
  u - normal*2.*dot(u, normal)
}

fn rand32() -> f32 {
  rand::random()
}

pub fn random_in_unit_ball() -> Vec {
  loop {
    let u = vec(rand32(), rand32(), rand32()) * 2. - ONES;
    if norm(u) < 1. {
      return u;
    }
  }
}
