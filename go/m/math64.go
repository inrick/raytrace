//go:build math64

package m // import "ray/m"

import (
	"math"
)

// Exported name
type Float = float

type float = float64

const MaxFloat = math.MaxFloat64

var Epsilon float

func init() {
	Epsilon = math.Nextafter(1, 2) - 1
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
