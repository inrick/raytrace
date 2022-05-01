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

pub fn vec(x: f32, y: f32, z: f32) -> Vec3 {
	Vec3 { x, y, z }
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
	pub fn sqrt(self) -> Vec3 {
		vec(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
	}

	pub fn norm(self) -> f32 {
		dot(self, self).sqrt()
	}

	pub fn normalize(self) -> Vec3 {
		self / self.norm()
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

fn rand32() -> f32 {
	rand::random()
}

pub fn random_in_unit_ball() -> Vec3 {
	loop {
		let u = vec(rand32(), rand32(), rand32()) * 2. - ONES;
		if u.norm() < 1. {
			return u;
		}
	}
}
