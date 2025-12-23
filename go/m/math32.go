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
