package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
)

func main() {
	var NSamples, SizeX, SizeY int
	flag.IntVar(&NSamples, "n", 10, "number of samples")
	flag.IntVar(&SizeX, "x", 600, "picture width")
	flag.IntVar(&SizeY, "y", 300, "picture height")
	flag.Parse()

	Raytrace(os.Stdout, NSamples, SizeX, SizeY)
}

func Raytrace(w io.Writer, nsamples, nx, ny int) {
	lookFrom := Vec{10, 2.5, 5}
	lookAt := Vec{-4, 0, -2}
	distToFocus := lookFrom.Sub(lookAt).Norm()
	aperture := float32(.05)

	nxf, nyf := float32(nx), float32(ny)

	cam := CameraNew(
		lookFrom, lookAt, Vec{0, 1, 0},
		20, nxf/nyf, aperture, distToFocus,
	)

	sc := SmallScene()

	buf := make([]byte, nx*ny*3)
	bi := 0
	for j := ny; j > 0; j-- {
		for i := 0; i < nx; i++ {
			var color Vec
			for s := 0; s < nsamples; s++ {
				x := (float32(i) + rand.Float32()) / nxf
				y := (float32(j-1) + rand.Float32()) / nyf
				r := cam.RayAtXY(x, y)
				color = color.Add(Color(r, sc))
			}
			color = color.Kdiv(float32(nsamples))
			color = color.Sqrt()
			buf[bi+0] = byte(255 * color.X)
			buf[bi+1] = byte(255 * color.Y)
			buf[bi+2] = byte(255 * color.Z)
			bi += 3
		}
	}

	PpmWrite(w, buf, nx, ny)
}

type Ray struct {
	Origin, Dir Vec
}

func (r Ray) Eval(t float32) Vec {
	return r.Origin.Add(r.Dir.Kmul(t))
}

func Color(r Ray, sc Scene) Vec {
	var rec HitRecord
	color := VecOne
	for depth := 0; depth < 50; depth++ {
		// apparently one clips slightly above 0 to avoid "shadow acne"
		if !sc.Hit(r, .001, math.MaxFloat32, &rec) {
			t := .5 * (r.Dir.Normalize().Y + 1)
			color = color.Mul(VecOne.Kmul(1 - t).Add(Vec{.75, .95, 1.0}.Kmul(t)))
			break
		}
		var scattered Ray
		var attenuation Vec
		if Scatter(r, rec, &attenuation, &scattered) {
			r = scattered
			color = attenuation.Mul(color)
		} else {
			color = VecZero
		}
	}
	return color
}

type MaterialKind int

const (
	KindMatte MaterialKind = iota
	KindMetal
	KindDielectric
)

type Material struct {
	Kind   MaterialKind
	Albedo Vec     // for Matte and Metal
	F      float32 // fuzz when kind is Metal, refIdx when Dielectric
}

type HitRecord struct {
	T      float32
	P      Vec
	Normal Vec
	Mat    Material
}

type Sphere struct {
	Center Vec
	Radius float32
	Mat    Material
}

func (s Sphere) Hit(r Ray, tmin, tmax float32, rec *HitRecord) bool {
	oc := r.Origin.Sub(s.Center)
	a := r.Dir.Dot(r.Dir)
	b := oc.Dot(r.Dir)
	c := oc.Dot(oc) - s.Radius*s.Radius
	D := b*b - a*c // NOTE: 4 cancels because b is no longer mult by 2
	if D > 0 {
		for _, t := range []float32{
			(-b - Sqrt32(D)) / a,
			(-b + Sqrt32(D)) / a,
		} {
			if tmin < t && t < tmax {
				rec.T = t
				rec.P = r.Eval(t)
				// (p-c)/r
				rec.Normal = rec.P.Sub(s.Center).Kdiv(s.Radius)
				rec.Mat = s.Mat
				return true
			}
		}
	}
	return false
}

type Scene struct {
	Spheres []Sphere
}

func (sc Scene) Hit(r Ray, tmin, tmax float32, rec *HitRecord) bool {
	var tmp HitRecord
	hit := false
	closest := tmax

	for _, s := range sc.Spheres {
		if s.Hit(r, tmin, closest, &tmp) {
			hit = true
			closest = tmp.T
		}
	}
	*rec = tmp
	return hit
}

type Camera struct {
	LowerLeftCorner     Vec
	Horiz, Vert, Origin Vec
	U, V, W             Vec
	LensRadius          float32
}

func CameraNew(
	lookFrom, lookAt, vup Vec,
	vfov, aspect, aperture, focusDist float32,
) Camera {
	theta := vfov * math.Pi / 180
	halfHeight := Tan32(theta / 2)
	halfWidth := aspect * halfHeight
	w := lookFrom.Sub(lookAt).Normalize()
	u := vup.Cross(w).Normalize()
	v := w.Cross(u)

	x := u.Kmul(halfWidth * focusDist)
	x = lookFrom.Sub(x)
	x = x.Sub(v.Kmul(halfHeight * focusDist))
	x = x.Sub(w.Kmul(focusDist))
	lowerLeftCorner := x

	return Camera{
		LowerLeftCorner: lowerLeftCorner,
		Horiz:           u.Kmul(2 * halfWidth * focusDist),
		Vert:            v.Kmul(2 * halfHeight * focusDist),
		Origin:          lookFrom,
		U:               u,
		V:               v,
		W:               w,
		LensRadius:      aperture / 2,
	}
}

func (c *Camera) RayAtXY(x, y float32) Ray {
	rd := RandomInUnitBall().Kmul(c.LensRadius)
	offset := c.U.Kmul(rd.X).Add(c.V.Kmul(rd.Y))

	dir := c.Horiz.Kmul(x).Add(c.Vert.Kmul(y))
	dir = c.LowerLeftCorner.Add(dir)
	dir = dir.Sub(c.Origin)
	dir = dir.Sub(offset)
	return Ray{Origin: c.Origin.Add(offset), Dir: dir}
}

func Schlick(cosine, refIdx float32) float32 {
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*Pow32(1-cosine, 5)
}

func Refract(u, n Vec, niOverNt float32, refracted *Vec) bool {
	un := u.Normalize()
	dt := un.Dot(n)
	D := 1 - niOverNt*niOverNt*(1-dt*dt)
	if D > 0 {
		v := n.Kmul(dt)
		v = un.Sub(v)
		v = v.Kmul(niOverNt)
		v = v.Sub(n.Kmul(Sqrt32(D)))
		*refracted = v
		return true
	}
	return false
}

func Scatter(rayIn Ray, rec HitRecord, attenuation *Vec, scattered *Ray) bool {
	switch rec.Mat.Kind {
	case KindMatte:
		target := rec.P.Add(rec.Normal).Add(RandomInUnitBall())
		*attenuation = rec.Mat.Albedo
		*scattered = Ray{Origin: rec.P, Dir: target.Sub(rec.P)}
		return true
	case KindMetal:
		reflected := rayIn.Dir.Normalize().Reflect(rec.Normal)
		dir := reflected.Add(RandomInUnitBall().Kmul(rec.Mat.F))
		*attenuation = rec.Mat.Albedo
		*scattered = Ray{Origin: rec.P, Dir: dir}
		return dir.Dot(rec.Normal) > 0
	case KindDielectric:
		refIdx := rec.Mat.F
		var outwardNormal Vec
		var niOverNt, cosine float32
		if rayIn.Dir.Dot(rec.Normal) > 0 {
			outwardNormal = rec.Normal.Neg()
			niOverNt = refIdx
			cosine = refIdx * rayIn.Dir.Dot(rec.Normal) / rayIn.Dir.Norm()
		} else {
			outwardNormal = rec.Normal
			niOverNt = 1 / refIdx
			cosine = -rayIn.Dir.Dot(rec.Normal) / rayIn.Dir.Norm()
		}
		var refracted Vec
		r := Ray{Origin: rec.P}
		if Refract(rayIn.Dir, outwardNormal, niOverNt, &refracted) &&
			rand.Float32() >= Schlick(cosine, refIdx) {
			r.Dir = refracted
		} else {
			r.Dir = rayIn.Dir.Reflect(rec.Normal)
		}
		*attenuation = VecOne
		*scattered = r
		return true
	default:
		panic(nil)
	}
}

func SmallScene() Scene {
	nspheres := 3 + 360/15
	spheres := make([]Sphere, nspheres)

	spheres[0] = Sphere{
		Center: Vec{0, -1000, 0},
		Radius: 1000,
		Mat:    Material{Kind: KindMatte, Albedo: Vec{.88, .96, .7}},
	}
	spheres[1] = Sphere{
		Center: Vec{1.5, 1, 0},
		Radius: 1,
		Mat:    Material{Kind: KindDielectric, F: 1.5},
	}
	spheres[2] = Sphere{
		Center: Vec{-1.5, 1, 0},
		Radius: 1,
		Mat:    Material{Kind: KindMetal, Albedo: Vec{.8, .9, .8}, F: 0},
	}

	i := 3
	for deg := 0; deg < 360; deg += 15 {
		var x, z, R0, R1 float32
		x = Sin32(float32(deg) * math.Pi / 180)
		z = Cos32(float32(deg) * math.Pi / 180)
		R0 = 3
		R1 = .33 + x*z/9
		spheres[i] = Sphere{
			Center: Vec{R0 * x, R1, R0 * z},
			Radius: R1,
			Mat:    Material{Kind: KindMatte, Albedo: Vec{x, .5 + x*z/2, z}},
		}
		i++
	}

	return Scene{spheres}
}

func PpmWrite(w io.Writer, buf []byte, x, y int) {
	fmt.Fprintf(w, "P6\n%d %d 255\n", x, y)
	N := len(buf)
	var written int
	for {
		n, err := w.Write(buf)
		written += n
		if err != nil {
			panic(err)
		}
		if written == N {
			break
		}
	}
}
