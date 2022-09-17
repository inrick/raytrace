package m

// Taken from Eskil Steenberg's talk "How I program C":
// https://www.youtube.com/watch?v=443UNeGrFoM#t=2h09m55s
func frandi(index uint32) uint32 {
	index = (index << 13) ^ index
	return (index*(index*index*15731+789221) + 1376312589) & 0x7fffffff
}

// TODO: data race, though harmless
var seed uint32 = 123456789

func Rand() float {
	result := frandi(seed)
	// Offset next seed by 1 to avoid a fixed point. The final image still looks
	// good. Without the offset it looks awful.
	seed = result + 1
	return float(result) / float(0x7fff_ffff)
}
