use crate::math::{deg_to_rad, rand32};
use crate::ray::{Material, Ray, Scene, Sphere};
use crate::vec::vec;

pub fn small_scene() -> Scene {
	let nspheres = 3 + 360 / 15;
	let mut spheres = Vec::with_capacity(nspheres);

	spheres.push(Sphere::new_static(
		vec(0., -1000., 0.),
		1000.,
		Material::Matte {
			albedo: vec(0.88, 0.96, 0.7),
		},
	));
	spheres.push(Sphere::new_static(
		vec(1.5, 1., 0.),
		1.,
		Material::Dielectric { ref_idx: 1.5 },
	));
	spheres.push(Sphere::new_static(
		vec(-1.5, 1., 0.),
		1.,
		Material::Metal {
			albedo: vec(0.8, 0.9, 0.8),
			fuzz: 0.,
		},
	));

	for deg in (0..360).step_by(15) {
		let x = deg_to_rad(deg as f32).sin();
		let z = deg_to_rad(deg as f32).cos();
		let r0 = 3.;
		let r1 = 0.33 + x * z / 9.;
		spheres.push(Sphere::new_static(
			vec(r0 * x, r1, r0 * z),
			r1,
			Material::Matte {
				albedo: vec(x, 0.5 + x * z / 2., z),
			},
		));
	}

	Scene::new(spheres)
}

pub fn small_scene_moving() -> Scene {
	let nspheres = 3 + 360 / 15;
	let mut spheres = Vec::with_capacity(nspheres);

	spheres.push(Sphere::new_static(
		vec(0., -1000., 0.),
		1000.,
		Material::Matte {
			albedo: vec(0.88, 0.96, 0.7),
		},
	));
	spheres.push(Sphere::new_static(
		vec(1.5, 1., 0.),
		1.,
		Material::Dielectric { ref_idx: 1.5 },
	));
	spheres.push(Sphere::new_static(
		vec(-1.5, 1., 0.),
		1.,
		Material::Metal {
			albedo: vec(0.8, 0.9, 0.8),
			fuzz: 0.,
		},
	));

	for deg in (0..360).step_by(15) {
		let x = deg_to_rad(deg as f32).sin();
		let z = deg_to_rad(deg as f32).cos();
		let r0 = 3.;
		let r1 = 0.33 + x * z / 9.;
		let dir = vec(0., rand32() / 2., 0.);
		spheres.push(Sphere::new(
			Ray {
				origin: vec(r0 * x, r1, r0 * z),
				dir,
				time: 0.,
			},
			r1,
			Material::Matte {
				albedo: vec(x, 0.5 + x * z / 2., z),
			},
		));
	}

	Scene::new(spheres)
}
