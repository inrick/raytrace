typedef struct {
  float x, y, z;
} v3;

const v3 v3_zero = (v3){0, 0, 0};
const v3 v3_one = (v3){1, 1, 1};

static v3 v3_neg(v3 u) {
  return (v3){-u.x, -u.y, -u.z};
}

static v3 v3_add(v3 u, v3 v) {
  return (v3){u.x+v.x, u.y+v.y, u.z+v.z};
}

static v3 v3_sub(v3 u, v3 v) {
  return (v3){u.x-v.x, u.y-v.y, u.z-v.z};
}

static v3 v3_mul(v3 u, v3 v) {
  return (v3){u.x*v.x, u.y*v.y, u.z*v.z};
}

static v3 v3_kmul(float k, v3 u) {
  return (v3){k*u.x, k*u.y, k*u.z};
}

static v3 v3_kdiv(v3 u, float k) {
  return (v3){u.x/k, u.y/k, u.z/k};
}

static v3 v3_sqrt(v3 u) {
  return (v3){sqrt(u.x), sqrt(u.y), sqrt(u.z)};
}

// L2 norm
static float v3_norm(v3 u) {
  return sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
}

static v3 v3_normalize(v3 u) {
  float norm = v3_norm(u);
  assert(fabsf(norm) > FLT_EPSILON); // rather not div by 0

  return v3_kmul(1.0/norm, u);
}

static float v3_dot(v3 u, v3 v) {
  return u.x*v.x + u.y*v.y + u.z*v.z;
}

static v3 v3_cross(v3 u, v3 v) {
  return (v3){
    .x = u.y*v.z - u.z*v.y,
    .y = u.z*v.x - u.x*v.z,
    .z = u.x*v.y - u.y*v.x,
  };
}

static v3 v3_reflect(v3 u, v3 n) {
  return v3_sub(u, v3_kmul(2*v3_dot(u,n), n));
}

static v3 random_in_unit_ball() {
  v3 u;
  float norm = 0;
  while (norm >= 1.0 || fabsf(norm) < FLT_EPSILON) {
    u = v3_sub(v3_kmul(2, (v3){my_rand(), my_rand(), my_rand()}), v3_one);
    norm = v3_norm(u);
  }
  return u;
}
