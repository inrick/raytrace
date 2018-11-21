#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h> // drand48 seems to require gnu extension
#include <unistd.h>

// vector begin
typedef struct {
  float x, y, z;
} v3;

const v3 v3_zero = (v3){0, 0, 0};
const v3 v3_one = (v3){1, 1, 1};

v3 v3_neg(v3 u) {
  return (v3){-u.x, -u.y, -u.z};
}

v3 v3_add(v3 u, v3 v) {
  return (v3){u.x+v.x, u.y+v.y, u.z+v.z};
}

v3 v3_kadd(float k, v3 u) {
  return (v3){k+u.x, k+u.y, k+u.z};
}

v3 v3_sub(v3 u, v3 v) {
  return (v3){u.x-v.x, u.y-v.y, u.z-v.z};
}

v3 v3_mul(v3 u, v3 v) {
  return (v3){u.x*v.x, u.y*v.y, u.z*v.z};
}

v3 v3_kmul(float k, v3 u) {
  return (v3){k*u.x, k*u.y, k*u.z};
}

v3 v3_div(v3 u, v3 v) {
  return (v3){u.x/v.x, u.y/v.y, u.z/v.z};
}

v3 v3_kdiv(v3 u, float k) {
  return (v3){u.x/k, u.y/k, u.z/k};
}
// L2 norm
float v3_norm(v3 u) {
  return sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
}

v3 v3_normalize(v3 u) {
  float norm = v3_norm(u);
  assert(fabsf(norm) > FLT_EPSILON); // rather not div by 0

  return v3_kmul(1.0/norm, u);
}

float v3_dot(v3 u, v3 v) {
  return u.x*v.x + u.y*v.y + u.z*v.z;
}

v3 v3_cross(v3 u, v3 v) {
  return (v3){
    .x = u.y*v.z - u.z*v.y,
    .y = u.z*v.x - u.x*v.z,
    .z = u.x*v.y - u.y*v.x,
  };
}

v3 v3_reflect(v3 u, v3 n) {
  return v3_sub(u, v3_kmul(2*v3_dot(u,n), n));
}
// vector end

// ray begin
typedef struct {
  v3 A, B; // origin, direction
} ray;

// eval ray r at param t
v3 ray_eval(ray *r, float t) {
  return v3_add(r->A, v3_kmul(t, r->B));
}
// ray end

typedef enum {
  MATTE,
  METAL,
  DIELECTRIC,
} material_t;

typedef struct {
  material_t type;
  union {
    struct {
      v3 albedo;
    } matte;
    struct {
      v3 albedo;
      float fuzz;
    } metal;
    struct {
      float ref_idx;
    } dielectric;
  };
} material;

typedef struct {
  float t;
  v3 p;
  v3 normal;
  material *mat;
} hit_record;

typedef struct {
  v3 center;
  float radius;
  material mat;
} sphere;

typedef struct {
  v3 llc; // lower left corner
  v3 horiz;
  v3 vert;
  v3 origin;
} camera;

camera camera_new(v3 lookfrom, v3 lookat, v3 vup, float vfov, float aspect) {
  float theta = vfov*M_PI/180;
  float half_height = tan(theta/2);
  float half_width = aspect * half_height;
  v3 w = v3_normalize(v3_sub(lookfrom, lookat));
  v3 u = v3_normalize(v3_cross(vup, w));
  v3 v = v3_cross(w, u);
  return (camera){
    .llc    = v3_sub(v3_sub(v3_sub(lookfrom, v3_kmul(half_width, u)), v3_kmul(half_height, v)), w),
    .horiz  = v3_kmul(2*half_width, u),
    .vert   = v3_kmul(2*half_height, v),
    .origin = lookfrom,
  };
}

ray camera_ray_at_xy(camera *c, float x, float y) {
  // llc + x*horiz + y*vert - origin
  v3 dir = v3_sub(
      v3_add(c->llc, v3_add(v3_kmul(x, c->horiz), v3_kmul(y, c->vert))),
      c->origin);
  return (ray){.A = c->origin, .B = dir};
}

void raytrace(void);

int main() {
  raytrace();
  return 0;
}

v3 random_in_unit_ball() {
  v3 u;
  float norm = 0;
  while (norm >= 1.0 || fabsf(norm) < FLT_EPSILON) {
    u = v3_sub(v3_kmul(2, (v3){drand48(), drand48(), drand48()}), v3_one);
    norm = v3_norm(u);
  }
  return u;
}

float schlick(float cosine, float ref_idx) {
  float r0 = (1-ref_idx)/(1+ref_idx);
  r0 = r0*r0;
  return r0 + (1-r0)*pow(1-cosine, 5);
}

bool refract(v3 u, v3 n, float ni_over_nt, v3 *refracted) {
  v3 unormed = v3_normalize(u);
  float dt = v3_dot(unormed, n);
  float D = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
  if (D > 0) {
    v3 ref = v3_sub(v3_kmul(ni_over_nt, v3_sub(unormed, v3_kmul(dt, n))), v3_kmul(sqrt(D), n));
    memcpy(refracted, &ref, sizeof(*refracted));
    return true;
  }
  return false;
}

// in: mat, r_in, rec
// out: attenuation, scattered
// out parameters are written if scatter returns true
bool scatter(material *mat, ray *r_in, hit_record *rec, v3 *attenuation, ray *scattered) {
  switch (mat->type) {
  case MATTE: {
    v3 target = v3_add(v3_add(rec->p, rec->normal), random_in_unit_ball());
    ray r = {.A = rec->p, .B = v3_sub(target, rec->p)};
    memcpy(attenuation, &mat->matte.albedo, sizeof(*attenuation));
    memcpy(scattered, &r, sizeof(*scattered));
    return true;
  }
  case METAL: {
    v3 reflected = v3_reflect(v3_normalize(r_in->B), rec->normal);
    ray r = {.A = rec->p, .B = v3_add(reflected, v3_kmul(mat->metal.fuzz, random_in_unit_ball()))};
    memcpy(attenuation, &mat->metal.albedo, sizeof(*attenuation));
    memcpy(scattered, &r, sizeof(*scattered));
    return v3_dot(scattered->B, rec->normal) > 0;
  }
  case DIELECTRIC: {
    v3 outward_normal;
    float ni_over_nt;
    float cosine;
    float ref_idx = mat->dielectric.ref_idx;
    if (v3_dot(r_in->B, rec->normal) > 0) {
      outward_normal = v3_neg(rec->normal);
      ni_over_nt = ref_idx;
      cosine = ref_idx * v3_dot(r_in->B, rec->normal) / v3_norm(r_in->B);
    } else {
      outward_normal = rec->normal;
      ni_over_nt = 1.0 / ref_idx;
      cosine = -v3_dot(r_in->B, rec->normal) / v3_norm(r_in->B);
    }
    v3 refracted;
    float reflect_prob;
    if (refract(r_in->B, outward_normal, ni_over_nt, &refracted)) {
      reflect_prob = schlick(cosine, ref_idx);
    } else {
      reflect_prob = 1.0;
    }
    ray r;
    if (drand48() < reflect_prob) {
      r = (ray){.A = rec->p, .B = v3_reflect(r_in->B, rec->normal)};
    } else {
      r = (ray){.A = rec->p, .B = refracted};
    }
    memcpy(attenuation, &v3_one, sizeof(*attenuation));
    memcpy(scattered, &r, sizeof(*scattered));
    return true;
  }
  default:
    assert(0);
  }
  return false;
}

// The out parameter hit_record will be written to if function returns true
bool hit_sphere(sphere *s, ray *r, float tmin, float tmax, hit_record *rec) {
  v3 oc = v3_sub(r->A, s->center);
  float a = v3_dot(r->B, r->B);
  float b = v3_dot(oc, r->B);
  float c = v3_dot(oc, oc) - s->radius*s->radius;
  float D = b*b - a*c; // NOTE: 4 cancels because b no longer mult by 2
  if (D > 0) {
    // try first root
    float t = (-b - sqrt(D))/a;
    if (tmin < t && t < tmax) {
      rec->t = t;
      rec->p = ray_eval(r, t);
      // (p-c)/r
      rec->normal = v3_kdiv(v3_sub(rec->p, s->center), s->radius);
      rec->mat = &s->mat;
      return true;
    }
    // try second root
    t = (-b + sqrt(D))/a;
    if (tmin < t && t < tmax) {
      rec->t = t;
      rec->p = ray_eval(r, t);
      // (p-c)/r
      rec->normal = v3_kdiv(v3_sub(rec->p, s->center), s->radius);
      rec->mat = &s->mat;
      return true;
    }
  }
  return false;
}

// The out parameter hit_record will be written to if function returns true
bool hit_sphere_arr(sphere spheres[], size_t nspheres, ray *r, float tmin, float tmax, hit_record *rec) {
  hit_record tmp;
  bool hit_obj = false;
  float closest = tmax;
  for (size_t i = 0; i < nspheres; i++) {
    if (hit_sphere(&spheres[i], r, tmin, closest, &tmp)) {
      hit_obj = true;
      closest = tmp.t;
      memcpy(rec, &tmp, sizeof tmp);
    }
  }
  return hit_obj;
}

v3 ray_color(ray *r, sphere spheres[], size_t nspheres, size_t depth) {
  hit_record rec;
  // apparently one clips slightly above 0 to avoid "shadow acne"
  if (hit_sphere_arr(spheres, nspheres, r, 0.001, FLT_MAX, &rec)) {
    ray scattered;
    v3 attenuation;
    if (depth < 50 && scatter(rec.mat, r, &rec, &attenuation, &scattered)) {
      return v3_mul(attenuation, ray_color(&scattered, spheres, nspheres, depth+1));
    }
    return v3_zero;
  }
  float t = 0.5*(v3_normalize(r->B).y + 1.0);
  return v3_add(
      v3_kmul(1.0-t, v3_one),
      v3_kmul(t, (v3){0.5, 0.7, 1.0}));
}

void raytrace(void) {
  size_t nx = 600;
  size_t ny = 300;
  size_t ns = 100;

  camera cam = camera_new(
    (v3){-2.0,0.7,1.6}, (v3){0,0,-1}, (v3){0,1,0}, 30, (float)nx / (float)ny
  );
  sphere spheres[] = {
    {.center = {0,0,-1},      .radius =  0.5,  .mat = {.type = MATTE, .matte.albedo = (v3){0.2,0.4,0.7}}},
    {.center = {0,-100.5,-1}, .radius =  100,  .mat = {.type = MATTE, .matte.albedo = (v3){0.8,0.8,0.0}}},
    {.center = {1,0,-1},      .radius =  0.5,  .mat = {.type = METAL, .metal = {.albedo = (v3){0.6,0.6,0.2}, .fuzz = 0.2}}},
    {.center = {-1,0,-1},     .radius =  0.5,  .mat = {.type = DIELECTRIC, .dielectric.ref_idx = 1.5}},
    {.center = {-1,0,-1},     .radius = -0.45, .mat = {.type = DIELECTRIC, .dielectric.ref_idx = 1.5}},
  };
  size_t nspheres = sizeof(spheres)/sizeof(spheres[0]);

  uint8_t buf[nx*ny*3];
  size_t bi = 0;
  printf("P6\n%zu %zu 255\n", nx, ny);
  fflush(stdout);
  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      v3 color = v3_zero;
      // anti-alias by averaging color around random nearby samples
      for (size_t s = 0; s < ns; s++) {
        float x = (float)(i+drand48()) / (float)nx;
        float y = (float)(j-1+drand48()) / (float)ny;
        ray r = camera_ray_at_xy(&cam, x, y);
        color = v3_add(color, ray_color(&r, spheres, nspheres, 0));
      }
      color = v3_kdiv(color, (float)ns);
      color = (v3){sqrt(color.x), sqrt(color.y), sqrt(color.z)};
      buf[bi+0] = (255.99 * color.x);
      buf[bi+1] = (255.99 * color.y);
      buf[bi+2] = (255.99 * color.z);
      bi += 3;
    }
  }
  write(STDOUT_FILENO, buf, sizeof buf);
}
