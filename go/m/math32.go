//go:build !math64

package m // import "ray/m"

import (
	"math"
)

// Exported name
type Float = float32

const MaxFloat = math.MaxFloat32

var Epsilon Float

func init() {
	Epsilon = math.Nextafter32(1, 2) - 1
}

func Abs(x Float) Float {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
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
