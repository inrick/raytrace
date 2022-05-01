typedef struct {
  v3 A, B; // origin, direction
} ray;

// eval ray r at param t
static v3 ray_eval(ray *r, float t) {
  return v3_add(r->A, v3_kmul(t, r->B));
}

typedef enum {
  MATTE,
  METAL,
  DIELECTRIC,
} material_t;

typedef struct {
  material_t type;
  union {
    struct { v3 albedo; }             matte;
    struct { v3 albedo; float fuzz; } metal;
    struct { float ref_idx; }         dielectric;
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
  size_t nspheres;
  sphere *spheres;
} scene;

typedef struct {
  v3 llc; // lower left corner
  v3 horiz, vert, origin, u, v, w;
  float lrad; // lens radius
} camera;

static camera camera_new(
    v3 lookfrom, v3 lookat, v3 vup, float vfov,
    float aspect, float aperture, float focus_dist
) {
  float theta, half_height, half_width;
  v3 u, v, w, llc;

  theta = vfov*M_PI/180;
  half_height = tan(theta/2);
  half_width = aspect * half_height;
  w = v3_normalize(v3_sub(lookfrom, lookat));
  u = v3_normalize(v3_cross(vup, w));
  v = v3_cross(w, u);
  llc = v3_sub(
      v3_sub(
          v3_sub(
              lookfrom,
              v3_kmul(half_width*focus_dist, u)),
          v3_kmul(half_height*focus_dist, v)),
      v3_kmul(focus_dist, w));
  return (camera){
    .llc    = llc,
    .horiz  = v3_kmul(2*half_width*focus_dist, u),
    .vert   = v3_kmul(2*half_height*focus_dist, v),
    .origin = lookfrom,
    .u      = u,
    .v      = v,
    .w      = w,
    .lrad   = aperture/2,
  };
}

static ray camera_ray_at_xy(camera *c, float x, float y) {
  v3 rd, offset, dir;

  rd = v3_kmul(c->lrad, random_in_unit_ball());
  offset = v3_add(v3_kmul(rd.x, c->u), v3_kmul(rd.y, c->v));
  dir = v3_sub(
      v3_sub(
          v3_add(c->llc, v3_add(v3_kmul(x, c->horiz), v3_kmul(y, c->vert))),
          c->origin),
      offset);
  return (ray){.A = v3_add(c->origin, offset), .B = dir};
}

