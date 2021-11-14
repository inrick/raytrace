package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"
)

func main() {
	var nsamples, x, y int
	var cpuprof, outputFile string
	flag.IntVar(&nsamples, "n", 10, "number of samples")
	flag.IntVar(&x, "x", 600, "picture width")
	flag.IntVar(&y, "y", 300, "picture height")
	flag.StringVar(&cpuprof, "cpuprof", "", "file to dump cpu profile")
	flag.StringVar(&outputFile, "o", "-", "output file")
	flag.Parse()

	if cpuprof != "" {
		f, err := os.Create(cpuprof)
		if err != nil {
			log.Fatalf("could not open cpuprof file %q: %v", cpuprof, err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	var w io.Writer
	switch outputFile {
	case "-":
		w = os.Stdout
	default:
		f, err := os.Create(outputFile)
		if err != nil {
			log.Fatalf("could not open output file %q: %v", outputFile, err)
		}
		defer f.Close()
		w = f
	}

	t0 := time.Now()
	Run(w, nsamples, x, y)
	t1 := time.Now().Sub(t0)

	fmt.Fprintf(os.Stderr, "raytracing took %.3f seconds\n", t1.Seconds())
}

func Run(w io.Writer, nsamples, nx, ny int) {
	lookFrom := Vec{10, 2.5, 5}
	lookAt := Vec{-4, 0, -2}
	distToFocus := Norm(Sub(lookFrom, lookAt))
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
				color = Add(color, sc.Color(&r))
			}
			color = Kdiv(color, float32(nsamples))
			color = Sqrt(color)
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

func (r *Ray) Eval(t float32) Vec {
	return Add(r.Origin, Kmul(t, r.Dir))
}

func (sc Scene) Color(r0 *Ray) Vec {
	var rec HitRecord
	r := *r0
	color := Ones // At infinity
	for depth := 0; depth < 50; depth++ {
		// apparently one clips slightly above 0 to avoid "shadow acne"
		if !sc.Hit(.001, math.MaxFloat32, &r, &rec) {
			t := .5 * (Normalize(r.Dir).Y + 1)
			color = Mul(color, Add(Kmul(t, Vec{.75, .95, 1.0}), Kmul(1-t, Ones)))
			break
		}
		attenuation, scattered := Scatter(&r, &rec)
		r = scattered
		color = Mul(color, attenuation)
	}
	return color
}

type MaterialKind int32

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

// Out parameter rec is written when Hit returns true, indicating a hit.
func (s *Sphere) Hit(tmin, tmax float32, r *Ray, rec *HitRecord) bool {
	oc := Sub(r.Origin, s.Center)
	a := Dot(r.Dir, r.Dir)
	b := Dot(oc, r.Dir)
	c := Dot(oc, oc) - s.Radius*s.Radius
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
				rec.Normal = Kdiv(Sub(rec.P, s.Center), s.Radius)
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

func (sc Scene) Hit(tmin, tmax float32, r *Ray, rec *HitRecord) bool {
	hit := false
	closest := tmax

	for i := range sc.Spheres {
		if sc.Spheres[i].Hit(tmin, closest, r, rec) {
			hit = true
			closest = rec.T
		}
	}
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
	w := Normalize(Sub(lookFrom, lookAt))
	u := Normalize(Cross(vup, w))
	v := Cross(w, u)

	x := Kmul(halfWidth*focusDist, u)
	x = Sub(lookFrom, x)
	x = Sub(x, Kmul(halfHeight*focusDist, v))
	x = Sub(x, Kmul(focusDist, w))
	lowerLeftCorner := x

	return Camera{
		LowerLeftCorner: lowerLeftCorner,
		Horiz:           Kmul(2*halfWidth*focusDist, u),
		Vert:            Kmul(2*halfHeight*focusDist, v),
		Origin:          lookFrom,
		U:               u,
		V:               v,
		W:               w,
		LensRadius:      aperture / 2,
	}
}

func (c *Camera) RayAtXY(x, y float32) Ray {
	rd := Kmul(c.LensRadius, RandomInUnitBall())
	offset := Add(Kmul(rd.X, c.U), Kmul(rd.Y, c.V))

	dir := Add(Kmul(x, c.Horiz), Kmul(y, c.Vert))
	dir = Add(c.LowerLeftCorner, dir)
	dir = Sub(dir, c.Origin)
	dir = Sub(dir, offset)
	return Ray{Origin: Add(c.Origin, offset), Dir: dir}
}

func Schlick(cosine, refIdx float32) float32 {
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*Pow32(1-cosine, 5)
}

func Refract(u, n Vec, niOverNt float32, refracted *Vec) bool {
	un := Normalize(u)
	dt := Dot(un, n)
	D := 1 - niOverNt*niOverNt*(1-dt*dt)
	if D > 0 {
		v := Sub(Kmul(niOverNt, Sub(un, Kmul(dt, n))), Kmul(Sqrt32(D), n))
		*refracted = v
		return true
	}
	return false
}

func Scatter(
	r0 *Ray,
	rec *HitRecord,
) (attenuation Vec, scattered Ray) {
	switch rec.Mat.Kind {
	case KindMatte:
		target := Add(Add(rec.P, rec.Normal), RandomInUnitBall())
		attenuation = rec.Mat.Albedo
		scattered.Origin = rec.P
		scattered.Dir = Sub(target, rec.P)
		return attenuation, scattered

	case KindMetal:
		reflected := Reflect(Normalize(r0.Dir), rec.Normal)
		fuzz := rec.Mat.F
		dir := Add(reflected, Kmul(fuzz, RandomInUnitBall()))
		attenuation = rec.Mat.Albedo
		if Dot(dir, rec.Normal) > 0 {
			scattered.Origin = rec.P
			scattered.Dir = dir
		} else {
			scattered = *r0
		}
		return attenuation, scattered

	case KindDielectric:
		refIdx := rec.Mat.F
		var outwardNormal Vec
		var niOverNt, cosine float32
		if Dot(r0.Dir, rec.Normal) > 0 {
			outwardNormal = Neg(rec.Normal)
			niOverNt = refIdx
			cosine = refIdx * Dot(r0.Dir, rec.Normal) / Norm(r0.Dir)
		} else {
			outwardNormal = rec.Normal
			niOverNt = 1 / refIdx
			cosine = -Dot(r0.Dir, rec.Normal) / Norm(r0.Dir)
		}
		var dir Vec
		if !(Refract(r0.Dir, outwardNormal, niOverNt, &dir) &&
			rand.Float32() >= Schlick(cosine, refIdx)) {
			dir = Reflect(r0.Dir, rec.Normal)
		}
		attenuation = Ones
		scattered.Origin = rec.P
		scattered.Dir = dir
		return attenuation, scattered

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
