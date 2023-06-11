package m

import "math"

// Taken from Eskil Steenberg's talk "How I program C":
// https://www.youtube.com/watch?v=443UNeGrFoM#t=2h09m55s
func frandi(index uint32) uint32 {
	index = (index << 13) ^ index
	return (index*(index*index*15731+789221) + 1376312589) & 0x7fffffff
}

// Go does not support thread locals so instead we use a separate state per
// goroutine.
type RandState uint32

func NewRand() *RandState {
	s := RandState(123456789)
	return &s
}

func (rnd *RandState) Rand() Float {
	result := frandi(uint32(*rnd))
	*rnd = RandState(result + 1)
	return Float(result) / Float(0x7fff_ffff)
}

func Pow(x, y Float) Float {
	return Float(math.Pow(float64(x), float64(y)))
}

func Sqrt(x Float) Float {
	return Float(math.Sqrt(float64(x)))
}

func Cos(x Float) Float {
	return Float(math.Cos(float64(x)))
}

func Sin(x Float) Float {
	return Float(math.Sin(float64(x)))
}

func Tan(x Float) Float {
	return Float(math.Tan(float64(x)))
}
