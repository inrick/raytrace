//go:build math64

package main

import (
	"math"
	"math/rand"
)

type float = float64

const MaxFloat = math.MaxFloat64

var Epsilon float

func init() {
	Epsilon = math.Nextafter(1, 2) - 1
}

func Rand() float {
	return rand.Float64()
}

func Abs(x float) float {
	return math.Abs(x)
}

func Pow(x, y float) float {
	return math.Pow(x, y)
}

func Sqrt(x float) float {
	return math.Sqrt(x)
}

func Cos(x float) float {
	return math.Cos(x)
}

func Sin(x float) float {
	return math.Sin(x)
}

func Tan(x float) float {
	return math.Tan(x)
}
