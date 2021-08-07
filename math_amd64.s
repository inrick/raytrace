#include "textflag.h"

// func Sqrt32(x float32) float32
TEXT ·Sqrt32(SB), NOSPLIT, $0
	XORPS  X0, X0 // break dependency
	SQRTSS x+0(FP), X0
	MOVSS  X0, ret+8(FP)
	RET
