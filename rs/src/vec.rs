use crate::math::rand32;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Default, Debug, Clone, Copy)]
pub struct Vec3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
}

pub static ONES: Vec3 = Vec3 {
	x: 1.,
	y: 1.,
	z: 1.,
};

static ONES_XY: Vec3 = Vec3 {
	x: 1.,
	y: 1.,
	z: 0.,
};

pub fn vec(x: f32, y: f32, z: f32) -> Vec3 {
	Vec3::new(x, y, z)
}

impl Add for Vec3 {
	type Output = Self;
	fn add(self, v: Self) -> Self {
		vec(self.x + v.x, self.y + v.y, self.z + v.z)
	}
}

impl Sub for Vec3 {
	type Output = Self;
	fn sub(self, v: Self) -> Self {
		vec(self.x - v.x, self.y - v.y, self.z - v.z)
	}
}

impl Mul for Vec3 {
	type Output = Self;
	fn mul(self, v: Self) -> Self {
		vec(self.x * v.x, self.y * v.y, self.z * v.z)
	}
}

impl Mul<f32> for Vec3 {
	type Output = Self;
	fn mul(self, k: f32) -> Self {
		vec(self.x * k, self.y * k, self.z * k)
	}
}

impl Div for Vec3 {
	type Output = Self;
	fn div(self, v: Self) -> Self {
		vec(self.x / v.x, self.y / v.y, self.z / v.z)
	}
}

impl Div<f32> for Vec3 {
	type Output = Self;
	fn div(self, k: f32) -> Self {
		vec(self.x / k, self.y / k, self.z / k)
	}
}

impl Neg for Vec3 {
	type Output = Self;
	fn neg(self) -> Self {
		vec(-self.x, -self.y, -self.z)
	}
}

impl Vec3 {
	pub fn new(x: f32, y: f32, z: f32) -> Self {
		Self { x, y, z }
	}

	pub fn sqrt(self) -> Vec3 {
		vec(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
	}

	pub fn norm(self) -> f32 {
		dot(self, self).sqrt()
	}

	pub fn normalize(self) -> Vec3 {
		self / self.norm()
	}

	pub fn dot(self, v: Self) -> f32 {
		dot(self, v)
	}

	pub fn cross(self, v: Self) -> Self {
		cross(self, v)
	}

	pub fn reflect(self, v: Self) -> Self {
		reflect(self, v)
	}

	pub fn near_zero(self) -> bool {
		self.x.abs() <= f32::EPSILON
			&& self.y.abs() <= f32::EPSILON
			&& self.z.abs() <= f32::EPSILON
	}

	pub fn axis(&self, index: i32) -> f32 {
		match index {
			0 => self.x,
			1 => self.y,
			2 => self.z,
			_ => unreachable!("don't do this"),
		}
	}
}

pub fn cross(u: Vec3, v: Vec3) -> Vec3 {
	vec(
		u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x,
	)
}

pub fn dot(u: Vec3, v: Vec3) -> f32 {
	u.x * v.x + u.y * v.y + u.z * v.z
}

pub fn reflect(u: Vec3, normal: Vec3) -> Vec3 {
	u - normal * 2. * dot(u, normal)
}

pub fn lerp(t: f32, u: Vec3, v: Vec3) -> Vec3 {
	u * (1. - t) + v * t
}

pub fn sample_square() -> Vec3 {
	vec(rand32() - 0.5, rand32() + 0.5, 0.)
}

pub fn random_in_unit_ball() -> Vec3 {
	loop {
		let u = vec(rand32(), rand32(), rand32()) * 2. - ONES;
		let norm = u.norm();
		if f32::EPSILON < norm && norm < 1. {
			return u;
		}
	}
}

pub fn random_in_unit_disk() -> Vec3 {
	loop {
		let u = vec(rand32(), rand32(), 0.) * 2. - ONES_XY;
		if u.norm() < 1. {
			return u;
		}
	}
}
