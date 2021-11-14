package main

import (
	"math/rand"
)

var (
	Zeros = Vec{0, 0, 0}
	Ones  = Vec{1, 1, 1}
)

type Vec struct {
	X, Y, Z float32
}

func RandomInUnitBall() Vec {
	var u Vec
	var norm float32
	for {
		u = Vec{X: rand.Float32(), Y: rand.Float32(), Z: rand.Float32()}
		u = Kmul(2, u)
		u = Sub(u, Ones)
		norm = Norm(u)
		if Epsilon <= norm && norm < 1 {
			break
		}
	}
	return u
}

func Neg(u Vec) Vec {
	return Vec{X: -u.X, Y: -u.Y, Z: -u.Z}
}

func Add(u, v Vec) Vec {
	return Vec{X: u.X + v.X, Y: u.Y + v.Y, Z: u.Z + v.Z}
}

func Sub(u, v Vec) Vec {
	return Vec{X: u.X - v.X, Y: u.Y - v.Y, Z: u.Z - v.Z}
}

func Mul(u, v Vec) Vec {
	return Vec{X: u.X * v.X, Y: u.Y * v.Y, Z: u.Z * v.Z}
}

func Kmul(k float32, u Vec) Vec {
	return Vec{X: k * u.X, Y: k * u.Y, Z: k * u.Z}
}

func Kdiv(u Vec, k float32) Vec {
	return Vec{X: u.X / k, Y: u.Y / k, Z: u.Z / k}
}

func Sqrt(u Vec) Vec {
	return Vec{X: Sqrt32(u.X), Y: Sqrt32(u.Y), Z: Sqrt32(u.Z)}
}

func Dot(u, v Vec) float32 {
	return u.X*v.X + u.Y*v.Y + u.Z*v.Z
}

func Cross(u, v Vec) Vec {
	return Vec{
		X: u.Y*v.Z - u.Z*v.Y,
		Y: u.Z*v.X - u.X*v.Z,
		Z: u.X*v.Y - u.Y*v.X,
	}
}

func Norm(u Vec) float32 {
	return Sqrt32(Dot(u, u))
}

func Normalize(u Vec) Vec {
	unorm := Norm(u)
	if unorm <= Epsilon {
		panic("cannot normalize a zero vector")
	}
	return Kdiv(u, unorm)
}

func Reflect(u, normal Vec) Vec {
	v := Kmul(2*Dot(u, normal), normal)
	return Sub(u, v)
}
