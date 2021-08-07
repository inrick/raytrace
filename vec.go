package main

import (
	"math/rand"
)

var (
	VecZero = Vec{0, 0, 0}
	VecOne  = Vec{1, 1, 1}
)

type Vec struct {
	X, Y, Z float32
}

func RandomInUnitBall() Vec {
	var u Vec
	var norm float32
	for norm >= 1 || Abs32(norm) < Epsilon {
		u = Vec{X: rand.Float32(), Y: rand.Float32(), Z: rand.Float32()}
		u = u.Kmul(2)
		u = u.Sub(VecOne)
		norm = u.Norm()
	}
	return u
}

func (u Vec) Neg() Vec {
	return Vec{X: -u.X, Y: -u.Y, Z: -u.Z}
}

func (u Vec) Add(v Vec) Vec {
	return Vec{X: u.X + v.X, Y: u.Y + v.Y, Z: u.Z + v.Z}
}

func (u Vec) Sub(v Vec) Vec {
	return Vec{X: u.X - v.X, Y: u.Y - v.Y, Z: u.Z - v.Z}
}

func (u Vec) Mul(v Vec) Vec {
	return Vec{X: u.X * v.X, Y: u.Y * v.Y, Z: u.Z * v.Z}
}

func (u Vec) Kmul(k float32) Vec {
	return Vec{X: u.X * k, Y: u.Y * k, Z: u.Z * k}
}

func (u Vec) Kdiv(k float32) Vec {
	return Vec{X: u.X / k, Y: u.Y / k, Z: u.Z / k}
}

func (u Vec) Sqrt() Vec {
	return Vec{X: Sqrt32(u.X), Y: Sqrt32(u.Y), Z: Sqrt32(u.Z)}
}

func (u Vec) Dot(v Vec) float32 {
	return u.X*v.X + u.Y*v.Y + u.Z*v.Z
}

func (u Vec) Cross(v Vec) Vec {
	return Vec{
		X: u.Y*v.Z - u.Z*v.Y,
		Y: u.Z*v.X - u.X*v.Z,
		Z: u.X*v.Y - u.Y*v.X,
	}
}

func (u Vec) Norm() float32 {
	return Sqrt32(u.Dot(u))
}

func (u Vec) Normalize() Vec {
	unorm := u.Norm()
	if unorm <= Epsilon {
		panic("cannot normalize a zero vector")
	}
	return u.Kdiv(unorm)
}

func (u Vec) Reflect(normal Vec) Vec {
	v := normal.Kmul(2 * u.Dot(normal))
	return u.Sub(v)
}
