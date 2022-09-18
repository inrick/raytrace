//go:build math64

package m // import "ray/m"

import (
	"math"
)

// Exported name
type Float = float64

const MaxFloat = math.MaxFloat64

var Epsilon Float

func init() {
	Epsilon = math.Nextafter(1, 2) - 1
}

func Abs(x Float) Float {
	return math.Abs(x)
}
