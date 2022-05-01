package main

import "math"

var Epsilon float32

func init() {
	Epsilon = math.Nextafter32(1, 2) - 1
}

func Abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

func Pow32(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
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

func Sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func Cos32(x float32) float32 {
	return float32(math.Cos(float64(x)))
}

func Sin32(x float32) float32 {
	return float32(math.Sin(float64(x)))
}

func Tan32(x float32) float32 {
	return float32(math.Tan(float64(x)))
}
