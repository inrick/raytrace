using Printf
using LinearAlgebra: ×, norm, normalize

const Elem = Float32
const Vec = Vector{Elem}

eps = 1e-8

struct Camera
  lower_left_corner::Vec
  horiz::Vec
  vert::Vec
  origin::Vec
  u::Vec
  v::Vec
  w::Vec
  lens_radius::Elem
end

function camera_new(look_from, look_at, vup, vfov, aspect, aperture, focus_dist)
  θ = vfov * π/180
  hh = tan(θ/2)  # half height
  hw = aspect*hh # half width
  w = normalize(look_from - look_at)
  u = normalize(vup × w)
  v = w × u

  lower_left_corner = (
    look_from
    - hw*focus_dist*u
    - hh*focus_dist*v
    -    focus_dist*w
  )

  return Camera(
    lower_left_corner,
    2*hw*focus_dist*u,
    2*hh*focus_dist*v,
    look_from,
    u,
    v,
    w,
    aperture/2,
  )
end

function ray_at_xy(c::Camera, x::Elem, y::Elem)
  rd = c.lens_radius * random_in_unit_ball()
  offset = rd[1]*c.u + rd[2]*c.v
  dir = c.lower_left_corner + (x*c.horiz + y*c.vert) - c.origin - offset
  return Ray(c.origin + offset, dir)
end

struct Ray
  origin::Vec
  dir::Vec
end

function eval_ray(r::Ray, t::Elem)::Vec
  return r.origin + t*r.dir
end

abstract type Material end
struct Matte <: Material
  albedo::Vec
end
struct Metal <: Material
  albedo::Vec
  fuzz::Elem
end
struct Dielectric <: Material
  ref_idx::Elem
end

function refract(u::Vec, normal::Vec, ni_over_nt::Elem)::Union{Vec,Nothing}
  un = normalize(u)
  dt = un'*normal
  D = 1 - ni_over_nt^2*(1-dt^2)
  if D > 0
    return ni_over_nt*(un - dt * normal) - sqrt(D)*normal
  else
    return nothing
  end
end

function reflect(u::Vec, normal::Vec)::Vec
  return u - 2*(u'*normal)*normal
end

function schlick(cosine::Elem, ref_idx::Elem)::Elem
  r0 = ((1 - ref_idx) / (1 + ref_idx))^2
  return r0 + (1-r0)*(1-cosine)^5
end

function scatter(r0::Ray, p::Vec, normal::Vec, mat::Matte)
  target = p + normal + random_in_unit_ball()
  scattered = Ray(p, target - p)
  return mat.albedo, scattered
end

function scatter(r0::Ray, p::Vec, normal::Vec, mat::Metal)
  reflected = reflect(normalize(r0.dir), normal)
  dir = reflected + mat.fuzz * random_in_unit_ball()
  if dir'*normal > 0
    scattered = Ray(p, dir)
  else
    scattered = r0
  end
  return mat.albedo, scattered
end

function scatter(r0::Ray, p::Vec, normal::Vec, mat::Dielectric)
  ref_idx = mat.ref_idx
  if r0.dir'*normal > 0
    outward_normal = -normal
    ni_over_nt = ref_idx
    cosine = ref_idx * (r0.dir'*normal) / norm(r0.dir)
  else
    outward_normal = normal
    ni_over_nt = 1/ref_idx
    cosine = -(r0.dir'*normal) / norm(r0.dir)
  end
  dir = refract(r0.dir, outward_normal, ni_over_nt)
  if dir == nothing || rand(Elem) < schlick(cosine, ref_idx)
    dir = reflect(r0.dir, normal)
  end
  scattered = Ray(p, dir)
  return ones(Elem, 3), scattered
end

mutable struct HitRecord
  t::Elem
  p::Vec
  normal::Vec
  material::Material
end

function scatter(r0::Ray, rec::HitRecord)
  attenuation, scattered = scatter(r0, rec.p, rec.normal, rec.material)
end

struct Sphere
  center::Vec
  radius::Elem
  material::Material
end

struct Scene
  spheres::Vector{Sphere}
end

function scene_color(sc::Scene, r::Ray)::Vec
  rec = HitRecord(0, zeros(3), zeros(3), Matte(zeros(3)))
  color = ones(Elem, 3) # at infinity
  for depth = 1:50
    if !hit(sc, Elem(.001), prevfloat(typemax(Elem)), r, rec)
      t = .5 * (normalize(r.dir)[2] + 1)
      color = color .* (t*Elem[.75, .95, 1.] + (1-t)*ones(Elem, 3))
      break
    end
    attenuation, scattered = scatter(r, rec)
    r = scattered
    color = color.*attenuation
  end
  return color
end

function write_ppm(fname::String, buf::Array{UInt8, 3}, w::Int, h::Int)
  open(fname, "w") do io
    @printf io "P6\n%d %d 255\n" w h
    write(io, buf[:])
  end
end

function random_in_unit_ball()::Vec
  while true
    u::Vec = 2*rand(3) - ones(3)
    if eps <= norm(u) <= 1
      return u
    end
  end
end

function hit(s::Sphere, tmin::Elem, tmax::Elem, r::Ray, rec::HitRecord)::Bool
  oc = r.origin - s.center
  a = r.dir'*r.dir
  b = oc'*r.dir
  c = oc'*oc - s.radius^2
  D = b^2 - a*c # NOTE: 4 cancels because b is no longer mult by 2
  if D > 0
    for t = [(-b-sqrt(D))/a, (-b+sqrt(D))/a]
      if tmin < t < tmax
        p = eval_ray(r, t)
        rec.t = t
        rec.p = p
        rec.normal = (p-s.center)/s.radius
        rec.material = s.material
        return true
      end
    end
  end
  return false
end

function hit(sc::Scene, tmin::Elem, tmax::Elem, r::Ray, rec::HitRecord)
  did_hit = false
  closest = tmax
  for i = 1:length(sc.spheres)
    if hit(sc.spheres[i], tmin, closest, r, rec)
      did_hit = true
      closest = rec.t
    end
  end
  return did_hit
end

function small_scene()::Scene
  nspheres = 3 + div(360, 15)
  spheres = Vector{Sphere}(undef, nspheres)

  spheres[1] = Sphere([0;-1000;0], 1000, Matte([.88;.96;.7]))
  spheres[2] = Sphere([1.5;1;0], 1, Dielectric(1.5))
  spheres[3] = Sphere([-1.5;1;0], 1, Metal([.8;.9;.8], 0))

  i = 4
  for deg = 0:15:(360-15)
    x = sin(Elem(deg) * π/180)
    z = cos(Elem(deg) * π/180)
    R0 = 3
    R1 = .33 + x*z/9
    spheres[i] = Sphere([R0*x;R1;R0*z], R1, Matte([x;.5+x*z/2;z]))
    i += 1
  end
  @assert i == length(spheres)+1

  return Scene(spheres)
end

function main()
  width, height = 600, 300
  wf, hf = Elem(width), Elem(height)
  nsamples = 10

  look_from = Elem[10;2.5;5]
  look_at = Elem[-4;0;-2]
  dist_to_focus = norm(look_from - look_at)
  aperture = Elem(.05)
  cam = camera_new(
    look_from, look_at, Elem[0;1;0], Elem(20), wf/hf, aperture, dist_to_focus,
  )
  sc = small_scene()
  buf = zeros(UInt8, 3, width, height)
  for i = 1:height
    for j = 1:width
      color = zeros(Elem, 3)
      for s = 1:nsamples
        x = (j + rand(Elem)) / wf
        y = (height-i+1 + rand(Elem)) / hf
        color = color + scene_color(sc, ray_at_xy(cam, x, y))
      end
      # Want NaN in case color is negative
      color = @fastmath sqrt.(color / Elem(nsamples))
      buf[:,j,i] = unsafe_trunc.(UInt8, 255*color)
    end
  end

  write_ppm("test.ppm", buf, width, height)
end

main()
