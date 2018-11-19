#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// vector begin
typedef struct {
  float x, y, z;
} v3;

const v3 v3_zero = (v3){0, 0, 0};
const v3 v3_one = (v3){1, 1, 1};

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

typedef struct {
  float t;
  v3 p;
  v3 normal;
} hit_record;

typedef struct {
  v3 center;
  float radius;
} sphere;

void raytrace1(void);
void raytrace2(void);
void ppm_gradient(void);

int main() {
  raytrace2();
  return 0;
}

// The out parameter hit_record will be written to if function returns true
bool hit_sphere2(sphere *s, ray *r, float tmin, float tmax, hit_record *rec) {
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
      return true;
    }
    // try second root
    t = (-b + sqrt(D))/a;
    if (tmin < t && t < tmax) {
      rec->t = t;
      rec->p = ray_eval(r, t);
      // (p-c)/r
      rec->normal = v3_kdiv(v3_sub(rec->p, s->center), s->radius);
      return true;
    }
  }
  return false;
}

// The out parameter hit_record will be written to if function returns true
bool hit_sphere_arr(sphere s[], size_t nspheres, ray *r, float tmin, float tmax, hit_record *rec) {
  hit_record tmp;
  bool hit_obj = false;
  float closest = tmax;
  for (size_t i = 0; i < nspheres; i++) {
    if (hit_sphere2(&s[i], r, tmin, closest, &tmp)) {
      hit_obj = true;
      closest = tmp.t;
      memcpy(rec, &tmp, sizeof tmp);
    }
  }
  return hit_obj;
}

v3 ray_color2(ray *r, sphere s[], size_t nspheres) {
  hit_record rec;
  if (hit_sphere_arr(s, nspheres, r, 0.0, FLT_MAX, &rec)) {
    return v3_kmul(0.5, v3_add(rec.normal, v3_one));
  }
  float t = 0.5*(v3_normalize(r->B).y + 1.0);
  return v3_add(
      v3_kmul(1.0-t, v3_one),
      v3_kmul(t, (v3){0.5, 0.7, 1.0}));
}

void raytrace2(void) {
  size_t nx = 600;
  size_t ny = 300;
  printf("P3\n%zu %zu\n255\n", nx, ny);
  v3 llc = (v3){-2.0, -1.0, -1.0};
  v3 horiz = (v3){4.0, 0.0, 0.0};
  v3 vert = (v3){0.0, 2.0, 0.0};
  sphere s[] = {
    (sphere){(v3){0,0,-1}, 0.5},
    (sphere){(v3){0,-100.5,-1}, 100},
  };
  size_t nspheres = 2;
  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      float a = (float)i / (float)nx;
      float b = (float)(j-1) / (float)ny;
      v3 dir = v3_add(llc, v3_add(v3_kmul(a, horiz), v3_kmul(b, vert)));
      ray r = (ray){v3_zero, dir};
      v3 col = ray_color2(&r, s, nspheres);
      int ir = (int)(255.99 * col.x);
      int ig = (int)(255.99 * col.y);
      int ib = (int)(255.99 * col.z);
      printf("%d %d %d\n", ir, ig, ib);
    }
  }
}

float hit_sphere1(v3 center, float radius, ray *r) {
  v3 oc = v3_sub(r->A, center);
  float a = v3_dot(r->B, r->B);
  float b = 2.0 * v3_dot(oc, r->B);
  float c = v3_dot(oc, oc) - radius*radius;
  float D = b*b - 4*a*c;
  return (D < 0) ? -1.0 : (-b - sqrt(D)) / (2.0*a);
}

v3 ray_color1(ray *r) {
  float t = hit_sphere1((v3){0,0,-1}, -0.5, r);
  if (t > 0.0) {
    v3 N = v3_normalize(v3_sub(ray_eval(r, t), (v3){0,0,-1}));
    return v3_kmul(0.5, v3_add(N, v3_one));
  }
  t = 0.5*(v3_normalize(r->B).y + 1.0);
  // linear blend (1-t)*(1,1,1) + t*(0.5,0.7,1)
  return v3_add(
      v3_kmul(1.0 - t, v3_one),
      v3_kmul(t, (v3){0.5, 0.7, 1.0}));
}

void raytrace1(void) {
  size_t nx = 600;
  size_t ny = 300;
  printf("P3\n%zu %zu\n255\n", nx, ny);

  // lower left corner and viewport dims
  v3 llc = (v3){-2.0, -1.0, -1.0};
  v3 horiz = (v3){4.0, 0.0, 0.0};
  v3 vert = (v3){0.0, 2.0, 0.0};

  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      float a = (float)i / (float)nx;
      float b = (float)(j-1) / (float)ny;
      // dir = llc + a*horiz + b*vert
      v3 dir = v3_add(llc, v3_add(v3_kmul(a, horiz), v3_kmul(b, vert)));
      ray r = (ray){v3_zero, dir};
      v3 col = ray_color1(&r);
      int ir = (int)(255.99 * col.x);
      int ig = (int)(255.99 * col.y);
      int ib = (int)(255.99 * col.z);
      printf("%d %d %d\n", ir, ig, ib);
    }
  }
}

void ppm_gradient(void) {
  size_t nx = 400;
  size_t ny = 200;
  printf("P3\n%zu %zu\n255\n", nx, ny);
  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      v3 rgb = (v3){(float)i / (float)nx, (float)(j-1) / (float)ny, 0.5};
      int ir = (int)(255.99 * rgb.x);
      int ig = (int)(255.99 * rgb.y);
      int ib = (int)(255.99 * rgb.z);
      printf("%d %d %d\n", ir, ig, ib);
    }
  }
}
