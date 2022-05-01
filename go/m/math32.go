//go:build !math64

package m // import "ray/m"

import (
	"math"
	"math/rand"
)

// Exported name
type Float = float

type float = float32

const MaxFloat = math.MaxFloat32

var Epsilon float

func init() {
	Epsilon = math.Nextafter32(1, 2) - 1
}

func Rand() float {
	return rand.Float32()
}

func Abs(x float) float {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

func Pow(x, y float) float {
	return float(math.Pow(float64(x), float64(y)))
}

// tinygo does not support the go assembler so just use a simple wrapper for
// sqrt as well.
//func Sqrt32(x float32) float32
/*
#include "textflag.h"

// func Sqrt32(x float32) float32
TEXT ·Sqrt32(SB), NOSPLIT, $0
	XORPS  X0, X0 // break dependency
	SQRTSS x+0(FP), X0
	MOVSS  X0, ret+8(FP)
	RET
*/

func Sqrt(x float) float {
	return float(math.Sqrt(float64(x)))
}

func Cos(x float) float {
	return float(math.Cos(float64(x)))
}

func Sin(x float) float {
	return float(math.Sin(float64(x)))
}

func Tan(x float) float {
	return float(math.Tan(float64(x)))
}
