package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"sync"
	"time"

	"ray/m"
)

type float = m.Float

func main() {
	log.SetFlags(0)

	var nsamples, x, y int
	var cpuprof, outputFile string
	flag.IntVar(&nsamples, "n", 10, "number of samples")
	flag.IntVar(&x, "x", 600, "picture width")
	flag.IntVar(&y, "y", 300, "picture height")
	flag.StringVar(&cpuprof, "cpuprof", "", "file to dump cpu profile")
	flag.StringVar(&outputFile, "o", "out.png", "output file")
	flag.Parse()

	if nsamples <= 0 {
		log.Fatalf("number of samples has to be positive")
	}

	if cpuprof != "" {
		f, err := os.Create(cpuprof)
		if err != nil {
			log.Fatalf("could not open cpuprof file %q: %v", cpuprof, err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	var write ImageWriter
	var w io.Writer
	switch outputFile {
	case "-":
		write = WritePPM
		w = os.Stdout
	default:
		switch ext := strings.ToLower(filepath.Ext(outputFile)); ext {
		case ".png":
			write = WritePNG
		case ".jpg", ".jpeg":
			write = WriteJPG
		case ".ppm":
			write = WritePPM
		default:
			log.Fatalf(
				"unsupported file extension %q (supported: ppm, png, jpg/jpeg)",
				ext,
			)
		}
		f, err := os.Create(outputFile)
		if err != nil {
			log.Fatalf("could not open output file %q: %v", outputFile, err)
		}
		defer f.Close()
		w = f
	}

	t0 := time.Now()
	err := Run(write, w, nsamples, x, y)
	t1 := time.Now().Sub(t0)

	if err != nil {
		log.Fatalf("ERROR: %v", err)
	}

	log.Printf("raytracing took %.3f seconds\n", t1.Seconds())
}

type ImageWriter func(io.Writer, []byte, int, int) error

func Run(write ImageWriter, w io.Writer, nsamples, nx, ny int) error {
	lookFrom := Vec{10, 2.5, 5}
	lookAt := Vec{-4, 0, -2}
	distToFocus := Norm(Sub(lookFrom, lookAt))
	aperture := float(.05)

	nxf, nyf := float(nx), float(ny)

	cam := CameraNew(
		lookFrom, lookAt, Vec{0, 1, 0},
		20, nxf/nyf, aperture, distToFocus,
	)

	sc := SmallScene()

	buf := make([]byte, 3*nx*ny)

	bufpos := 0
	rowlen := 3 * nx
	var wg sync.WaitGroup
	wg.Add(ny)
	for i := 0; i < ny; i++ {
		ymax := float(ny-i-0) / nyf
		ymin := float(ny-i-1) / nyf
		bufchunk := buf[bufpos : bufpos+rowlen]
		go func() {
			Render(bufchunk, cam, sc, nsamples, nx, 1, ymin, ymax)
			wg.Done()
		}()
		bufpos += rowlen
	}
	wg.Wait()

	return write(w, buf, nx, ny)
}

func Render(buf []byte, cam Camera, sc Scene, nsamples, nx, ny int, ymin, ymax float) {
	yheight := ymax - ymin
	bi := 0
	for j := ny; j > 0; j-- {
		for i := 0; i < nx; i++ {
			var color Vec
			for s := 0; s < nsamples; s++ {
				x := (float(i) + m.Rand()) / float(nx)
				y := ymin + (yheight*(float(j-1)+m.Rand()))/float(ny)
				r := cam.RayAtXY(x, y)
				color = Add(color, sc.Color(&r))
			}
			color = Kdiv(color, float(nsamples))
			color = Sqrt(color)
			buf[bi+0] = byte(255 * color.X)
			buf[bi+1] = byte(255 * color.Y)
			buf[bi+2] = byte(255 * color.Z)
			bi += 3
		}
	}
}

type Ray struct {
	Origin, Dir Vec
}

func (r *Ray) Eval(t float) Vec {
	return Add(r.Origin, Kmul(t, r.Dir))
}

func (sc Scene) Color(r0 *Ray) Vec {
	var rec HitRecord
	r := *r0
	color := Ones // At infinity
	for depth := 0; depth < 50; depth++ {
		// apparently one clips slightly above 0 to avoid "shadow acne"
		if !sc.Hit(.001, m.MaxFloat, &r, &rec) {
			t := .5 * (Normalize(r.Dir).Y + 1)
			color = Mul(color, Add(Kmul(t, Vec{.75, .95, 1.0}), Kmul(1-t, Ones)))
			break
		}
		attenuation, scattered := Scatter(&r, rec.P, rec.Normal, rec.Mat)
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
	Albedo Vec   // for Matte and Metal
	F      float // fuzz when kind is Metal, refIdx when Dielectric
}

type HitRecord struct {
	T      float
	P      Vec
	Normal Vec
	Mat    Material
}

type Sphere struct {
	Center Vec
	Radius float
	Mat    Material
}

// Out parameter rec is written when Hit returns true, indicating a hit.
func (s *Sphere) Hit(tmin, tmax float, r *Ray, rec *HitRecord) bool {
	oc := Sub(r.Origin, s.Center)
	a := Dot(r.Dir, r.Dir)
	b := Dot(oc, r.Dir)
	c := Dot(oc, oc) - s.Radius*s.Radius
	D := b*b - a*c // NOTE: 4 cancels because b is no longer mult by 2
	if D > 0 {
		for _, t := range []float{
			(-b - m.Sqrt(D)) / a,
			(-b + m.Sqrt(D)) / a,
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

func (sc Scene) Hit(tmin, tmax float, r *Ray, rec *HitRecord) bool {
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
	LensRadius          float
}

func CameraNew(
	lookFrom, lookAt, vup Vec,
	vfov, aspect, aperture, focusDist float,
) Camera {
	theta := vfov * math.Pi / 180
	halfHeight := m.Tan(theta / 2)
	halfWidth := aspect * halfHeight
	w := Normalize(Sub(lookFrom, lookAt))
	u := Normalize(Cross(vup, w))
	v := Cross(w, u)

	llc := Kmul(halfWidth*focusDist, u)
	llc = Sub(lookFrom, llc)
	llc = Sub(llc, Kmul(halfHeight*focusDist, v))
	llc = Sub(llc, Kmul(focusDist, w))

	return Camera{
		LowerLeftCorner: llc,
		Horiz:           Kmul(2*halfWidth*focusDist, u),
		Vert:            Kmul(2*halfHeight*focusDist, v),
		Origin:          lookFrom,
		U:               u,
		V:               v,
		W:               w,
		LensRadius:      aperture / 2,
	}
}

func (c *Camera) RayAtXY(x, y float) Ray {
	rd := Kmul(c.LensRadius, RandomInUnitBall())
	offset := Add(Kmul(rd.X, c.U), Kmul(rd.Y, c.V))

	dir := Add(Kmul(x, c.Horiz), Kmul(y, c.Vert))
	dir = Add(c.LowerLeftCorner, dir)
	dir = Sub(dir, c.Origin)
	dir = Sub(dir, offset)
	return Ray{Origin: Add(c.Origin, offset), Dir: dir}
}

func Schlick(cosine, refIdx float) float {
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*m.Pow(1-cosine, 5)
}

func Refract(u, n Vec, niOverNt float, refracted *Vec) bool {
	un := Normalize(u)
	dt := Dot(un, n)
	D := 1 - niOverNt*niOverNt*(1-dt*dt)
	if D > 0 {
		v := Sub(Kmul(niOverNt, Sub(un, Kmul(dt, n))), Kmul(m.Sqrt(D), n))
		*refracted = v
		return true
	}
	return false
}

func Scatter(r *Ray, p, normal Vec, mat Material) (Vec, Ray) {
	switch mat.Kind {
	case KindMatte:
		// p+normal+random
		target := Add(Add(p, normal), RandomInUnitBall())
		scattered := Ray{Origin: p, Dir: Sub(target, p)}
		return mat.Albedo, scattered

	case KindMetal:
		reflected := Reflect(Normalize(r.Dir), normal)
		dir := Add(reflected, Kmul(mat.F, RandomInUnitBall()))
		var scattered Ray
		if Dot(dir, normal) > 0 {
			scattered.Origin = p
			scattered.Dir = dir
		} else {
			scattered = *r
		}
		return mat.Albedo, scattered

	case KindDielectric:
		refIdx := mat.F
		var outwardNormal Vec
		var niOverNt, cosine float
		dot := Dot(r.Dir, normal)
		if dot > 0 {
			outwardNormal = Neg(normal)
			niOverNt = refIdx
			cosine = refIdx * dot / Norm(r.Dir)
		} else {
			outwardNormal = normal
			niOverNt = 1 / refIdx
			cosine = -dot / Norm(r.Dir)
		}
		var dir Vec
		if !Refract(r.Dir, outwardNormal, niOverNt, &dir) ||
			m.Rand() < Schlick(cosine, refIdx) {
			dir = Reflect(r.Dir, normal)
		}
		scattered := Ray{Origin: p, Dir: dir}
		return Ones, scattered

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
		var x, z, R0, R1 float
		x = m.Sin(float(deg) * math.Pi / 180)
		z = m.Cos(float(deg) * math.Pi / 180)
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

func WritePPM(w io.Writer, buf []byte, width, height int) error {
	fmt.Fprintf(w, "P6\n%d %d 255\n", width, height)
	_, err := w.Write(buf)
	return err
}

func bufToImage(buf []byte, width, height int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for i := 0; i < len(buf); i += 3 {
		x := i / 3 % width
		y := i / 3 / width
		img.Set(x, y, color.NRGBA{buf[i+0], buf[i+1], buf[i+2], 255})
	}
	return img
}

func WriteJPG(w io.Writer, buf []byte, width, height int) error {
	img := bufToImage(buf, width, height)
	return jpeg.Encode(w, img, &jpeg.Options{Quality: 90})
}

func WritePNG(w io.Writer, buf []byte, width, height int) error {
	img := bufToImage(buf, width, height)
	return png.Encode(w, img)
}
