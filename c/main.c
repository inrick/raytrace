#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include "rand.c"
#include "vec.c"
#include "structs.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef void image_write_fn(FILE*, uint8_t*, size_t, size_t, size_t);

static image_write_fn ppm_write;
static image_write_fn png_write;
static image_write_fn jpg_write;

typedef struct options options;
static void raytrace(image_write_fn, FILE*, options*);

static char* argv0;

void usage(void) {
  fprintf(
    stderr,
    "Usage: %s [-n <number of samples>] [-t <number of threads>] [-o <output file>.<ppm/png/jpg>]\n"
    "\n"
    "If <output file> is '-' then a ppm image is written on stdout.\n"
    "\n"
    "Examples:\n"
    "\n",
    argv0);
  fprintf(stderr, "	%s -n 100 -t 4 -o out.jpg\n", argv0);
  fprintf(stderr, "	%s -o out.ppm\n", argv0);
  fprintf(stderr, "	%s -o - > out.ppm\n", argv0);
  exit(1);
}

struct options {
  size_t nsamples;
  size_t threads;
};

int main(int argc, char** argv) {
  argv0 = argv[0];

  int opt;
  FILE* fp = NULL;
  image_write_fn* iwrite = ppm_write;

  options opts = {
    .nsamples = 10,
    .threads = 4,
  };

  while ((opt = getopt(argc, argv, "o:n:t:")) != -1) {
    switch (opt) {
    case 'o': {
      if (strcmp(optarg, "-") == 0) {
        fp = stdout;
        break;
      }
      char* ext = strrchr(optarg, '.');
      if (ext == NULL) {
        fprintf(
          stderr,
          "missing file name extension, support ppm/png/jpg\n");
        usage();
      } else if (strcasecmp(ext, ".png") == 0) {
        iwrite = png_write;
      } else if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0) {
        iwrite = jpg_write;
      } else if (strcasecmp(ext, ".ppm") == 0) {
        iwrite = ppm_write;
      } else {
        fprintf(
          stderr,
          "unsupported file extension: %s (only support ppm/png/jpg)\n",
          ext);
        usage();
      }
      fp = fopen(optarg, "wb");
      if (fp == NULL) {
        perror("could not open output file");
        exit(1);
      }
    } break;
    case 'n': {
      char* endptr = NULL;
      long n = strtol(optarg, &endptr, 10);
      if (n <= 0 || strcmp("", endptr) != 0) {
        fprintf(
          stderr,
          "invalid number of samples (%s), must be a positive integer\n",
          optarg);
        usage();
      }
      opts.nsamples = (size_t) n;
    } break;
    case 't': {
      char* endptr = NULL;
      long n = strtol(optarg, &endptr, 10);
      if (n <= 0 || strcmp("", endptr) != 0) {
        fprintf(
          stderr,
          "invalid number of threads (%s), must be a positive integer\n",
          optarg);
        usage();
      }
      opts.threads = (size_t) n;
    } break;
    default:
      usage();
    }
  }

  if (fp == NULL) {
    usage();
  }

  raytrace(iwrite, fp, &opts);

  if (fp != stdout && 0 != fclose(fp)) {
    perror("could not close output file");
    exit(1);
  }

  return 0;
}

static float schlick(float cosine, float ref_idx) {
  float r0 = (1-ref_idx)/(1+ref_idx);
  r0 = r0*r0;
  return r0 + (1-r0)*pow(1-cosine, 5);
}

static bool refract(v3 u, v3 n, float ni_over_nt, v3 *refracted) {
  v3 unormed = v3_normalize(u);
  float dt = v3_dot(unormed, n);
  float D = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
  if (D > 0) {
    *refracted = v3_sub(
      v3_kmul(ni_over_nt, v3_sub(unormed, v3_kmul(dt, n))),
      v3_kmul(sqrt(D), n)
    );
    return true;
  }
  return false;
}

// in: r_in, rec
// out: attenuation, scattered
// out parameters are written if scatter returns true
static bool scatter(
  ray *r_in, hit_record *rec, v3 *attenuation, ray *scattered
) {
  material *mat = rec->mat;
  switch (mat->type) {
  case MATTE: {
    v3 target = v3_add(v3_add(rec->p, rec->normal), random_in_unit_ball());
    *attenuation = mat->matte.albedo;
    *scattered = (ray){.A = rec->p, .B = v3_sub(target, rec->p)};;
    return true;
  }
  case METAL: {
    v3 reflected = v3_reflect(v3_normalize(r_in->B), rec->normal);
    v3 dir = v3_add(reflected, v3_kmul(mat->metal.fuzz, random_in_unit_ball()));
    *attenuation = mat->metal.albedo;
    *scattered = (ray){.A = rec->p, .B = dir};;
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
    ray r;
    if (refract(r_in->B, outward_normal, ni_over_nt, &refracted)
        && my_rand() >= schlick(cosine, ref_idx)) {
      r = (ray){.A = rec->p, .B = refracted};
    } else {
      r = (ray){.A = rec->p, .B = v3_reflect(r_in->B, rec->normal)};
    }
    *attenuation = v3_one;
    *scattered = r;
    return true;
  }
  default:
    assert(0);
  }
  return false;
}

// The out parameter hit_record will be written to if function returns true
static
bool hit_sphere(sphere *s, ray *r, float tmin, float tmax, hit_record *rec) {
  v3 oc = v3_sub(r->A, s->center);
  float a = v3_dot(r->B, r->B);
  float b = v3_dot(oc, r->B);
  float c = v3_dot(oc, oc) - s->radius*s->radius;
  float D = b*b - a*c; // NOTE: 4 cancels because b no longer mult by 2
  if (D > 0) {
    float tt[] = {(-b - sqrt(D))/a, (-b + sqrt(D))/a};
    for (int i = 0; i < 2; i++) {
      float t = tt[i];
      if (tmin < t && t < tmax) {
        rec->t = t;
        rec->p = ray_eval(r, t);
        // (p-c)/r
        rec->normal = v3_kdiv(v3_sub(rec->p, s->center), s->radius);
        rec->mat = &s->mat;
        return true;
      }
    }
  }
  return false;
}

// The out parameter hit_record will be written to if function returns true
static bool hit_scene(
  scene *sc, ray *r, float tmin, float tmax, hit_record *rec
) {
  hit_record tmp;
  bool hit_obj = false;
  float closest = tmax;
  size_t N = sc->nspheres;
  sphere *spheres = sc->spheres;

  for (size_t i = 0; i < N; i++) {
    if (hit_sphere(&spheres[i], r, tmin, closest, &tmp)) {
      hit_obj = true;
      closest = tmp.t;
      *rec = tmp;
    }
  }
  return hit_obj;
}

static v3 ray_color(ray *r0, scene *sc) {
  hit_record rec;
  v3 color = v3_one;
  ray r = *r0;
  for (size_t depth = 0; depth < 50; depth++) {
    // apparently one clips slightly above 0 to avoid "shadow acne"
    if (hit_scene(sc, &r, 0.001, FLT_MAX, &rec)) {
      ray scattered;
      v3 attenuation;
      if (scatter(&r, &rec, &attenuation, &scattered)) {
        r = scattered;
        color = v3_mul(attenuation, color);
      } else {
        color = v3_zero;
      }
    } else {
      float t = 0.5*(v3_normalize(r.B).y + 1.0);
      color = v3_mul(
        color, v3_add(
          v3_kmul(1.0-t, v3_one),
          v3_kmul(t, (v3){0.75, 0.95, 1.0})));
      break;
    }
  }
  return color;
}

/*
static scene random_scene() {
  size_t nspheres = 500; // some extra room, should calculate this properly
  sphere *spheres = calloc(nspheres, sizeof(*spheres));
  assert(spheres);
  spheres[0] = (sphere){
    .center = {0,-1000,0},
    .radius = 1000,
    .mat = {.type = MATTE, .matte.albedo = (v3){0.7,0.7,0.7}},
  };
  size_t i = 1;
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = my_rand();
      v3 center = (v3){a+0.9*my_rand(), 0.2, b+0.9*my_rand()};
      if (choose_mat < 0.8) { // matte
        spheres[i++] = (sphere){
          .center = center,
          .radius = 0.2,
          .mat = {.type = MATTE, .matte.albedo = {
            my_rand()*my_rand(), my_rand()*my_rand(), my_rand()*my_rand()
          }},
        };
      } else if (choose_mat < 0.95) { // metal
        spheres[i++] = (sphere){
          .center = center,
          .radius = 0.2,
          .mat = {.type = METAL, .metal = {
            .albedo = {
              0.5*(1+my_rand()), 0.5*(1+my_rand()), 0.5*(1+my_rand())
            },
            .fuzz = 0.5*my_rand(),
          }},
        };
      } else { // dielectric
        spheres[i++] = (sphere){
          .center = center,
          .radius = 0.2,
          .mat = {.type = DIELECTRIC, .dielectric.ref_idx = 1.5},
        };
      }
    }
  }
  spheres[i++] = (sphere){
    .center = {0,1,0},
    .radius = 1,
    .mat = {.type = DIELECTRIC, .dielectric.ref_idx = 1.5},
  };
  spheres[i++] = (sphere){
    .center = {-4,1,0},
    .radius = 1,
    .mat = {.type = MATTE, .matte.albedo = {0.2,0.4,0.7}},
  };
  spheres[i++] = (sphere){
    .center = {4,1,0},
    .radius = 1,
    .mat = {.type = METAL, .metal = {.albedo = {0.6, 0.6, 0.5}, .fuzz = 0}},
  };
  return (scene){.nspheres = i, .spheres = spheres};
}
*/

/*
static scene trivial_scene() {
  size_t nspheres = 2;
  sphere *spheres = calloc(nspheres, sizeof(*spheres));
  assert(spheres);
  spheres[0] = (sphere){
    .center = {0,-1000,0},
    .radius = 1000,
    .mat = {.type = MATTE, .matte.albedo = (v3){0.88,0.96,0.7}},
  };
  spheres[1] = (sphere){
    .center = {0,1,0},
    .radius = 1,
    .mat = {.type = METAL, .metal = {.albedo = {0.8,0.9,0.8}, .fuzz = 0.4}},
  };
  return (scene){.nspheres = nspheres, .spheres = spheres};
}
*/

// Call with NULL to get the number of spheres needed for the scene. Then
// allocate a buffer with room for the scene, create a scene object and call
// the function again.
static size_t small_scene(scene* sc) {
  if (sc == NULL) {
    return 3+360/15;
  }
  size_t nspheres = sc->nspheres;
  sphere* spheres = sc->spheres;
  assert(spheres);

  spheres[0] = (sphere){
    .center = {0,-1000,0},
    .radius = 1000,
    .mat = {.type = MATTE, .matte.albedo = (v3){0.88,0.96,0.7}},
  };
  spheres[1] = (sphere){
    .center = {1.5,1,0},
    .radius = 1,
    .mat = {.type = DIELECTRIC, .dielectric.ref_idx = 1.5},
  };
  spheres[2] = (sphere){
    .center = {-1.5,1,0},
    .radius = 1,
    .mat = {.type = METAL, .metal = {.albedo = {0.8,0.9,0.8}, .fuzz = 0.0}},
  };

  size_t i = 3;
  for (int deg = 0; deg < 360; deg += 15) {
    float x = sin(deg*M_PI/180.0);
    float z = cos(deg*M_PI/180.0);
    float R0 = 3;
    float R1 = 0.33+x*z/9;
    spheres[i++] = (sphere){
      .center = {R0*x,R1,R0*z},
      .radius = R1,
      .mat = {.type = MATTE, .matte.albedo = {x,0.5+x*z/2,z}},
    };
  }

  assert(nspheres >= i); // in case calculation is off
  return i;
}

static void
ppm_write(FILE* fp, uint8_t* buf, size_t size, size_t w, size_t h) {
  fprintf(fp, "P6\n%zu %zu 255\n", w, h);
  fflush(fp);
  fwrite(buf, size, 1, fp);
}

static void
stb_write_fn(void* context, void* data, int size) {
  fwrite(data, 1, size, (FILE*) context);
}

static void
png_write(FILE* fp, uint8_t* buf, size_t size, size_t w, size_t h) {
  int comp = 3; // components
  int stride_in_bytes = comp*w;
  assert(size == comp*w*h);
  stbi_write_png_to_func(stb_write_fn, fp, w, h, comp, buf, stride_in_bytes);
}

static void
jpg_write(FILE* fp, uint8_t* buf, size_t size, size_t w, size_t h) {
  int comp = 3; // components
  assert(size == comp*w*h);
  stbi_write_jpg_to_func(stb_write_fn, fp, w, h, comp, buf, 90);
}

typedef struct {
  uint8_t* buf;
  size_t len;
  camera* cam;
  scene* sc;
  size_t nsamples, nx, ny;
  float ymin, ymax;
} render_arg;

static void* render(void* arg0) {
  render_arg* arg = (render_arg*) arg0;

  uint8_t* buf = arg->buf;
  camera* cam = arg->cam;
  scene* sc = arg->sc;
  size_t nsamples = arg->nsamples;
  size_t nx = arg->nx, ny = arg->ny;
  float ymin = arg->ymin, ymax = arg->ymax;

  assert(arg->len == 3*nx*ny);

  float yheight = ymax - ymin;
  size_t bi = 0;
  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      v3 color = v3_zero;
      // anti-alias by averaging color around random nearby samples
      for (size_t s = 0; s < nsamples; s++) {
        float x = (float)(i+0+my_rand()) / (float)nx;
        float y = ymin + yheight*(float)(j-1+my_rand()) / (float)ny;
        ray r = camera_ray_at_xy(cam, x, y);
        color = v3_add(color, ray_color(&r, sc));
      }
      color = v3_kdiv(color, (float)nsamples);
      color = v3_sqrt(color);
      if (isnanf(color.x)) color.x = 0;
      if (isnanf(color.y)) color.y = 0;
      if (isnanf(color.z)) color.z = 0;
      buf[bi+0] = (255.0f * color.x);
      buf[bi+1] = (255.0f * color.y);
      buf[bi+2] = (255.0f * color.z);
      bi += 3;
    }
  }

  return (void*)0;
}

static void raytrace(image_write_fn iwrite, FILE* fp, options* opts) {
  size_t nx = 600, ny = 300;

  //v3 lookfrom = {11,1.8,5};
  //v3 lookat = {0,0,-1};
  v3 lookfrom = {10,2.5,5};
  v3 lookat = {-4,0,-2};
  float dist_to_focus = v3_norm(v3_sub(lookfrom, lookat));
  float aperture = 0.05;
  camera cam = camera_new(
    lookfrom, lookat, (v3){0,1,0}, 20,
    (float)nx / (float)ny, aperture, dist_to_focus
  );

  size_t nspheres = small_scene(NULL);
  sphere spheres[nspheres];
  scene sc = (scene){.spheres = spheres, .nspheres = nspheres};
  assert(nspheres == small_scene(&sc));

  size_t buflen = 3*nx*ny;
  uint8_t buf[buflen];
  size_t bufpos = 0;
  size_t ny_pos = 0;
  size_t threads = opts->threads;
  render_arg args[threads];
  for (size_t i = 0; i < threads; i++) {
    size_t ny_remaining = ny-ny_pos;
    size_t ny_th = ny_remaining/(threads - i);
    size_t len_th = 3*nx*ny_th;
    float ymax = (float)(ny_remaining)/(float)ny;
    float ymin = (float)(ny_remaining-ny_th)/(float)ny;
    args[i] = (render_arg){
      .buf = &buf[bufpos],
      .len = len_th,
      .cam = &cam,
      .sc = &sc,
      .nsamples = opts->nsamples,
      .nx = nx,
      .ny = ny_th,
      .ymin = ymin,
      .ymax = ymax,
    };
    ny_pos += ny_th;
    bufpos += len_th;
  }

  pthread_t thread_ids[threads];
  for (size_t i = 0; i < threads; i++) {
    pthread_t thread_id;
    if (0 != pthread_create(&thread_id, NULL, &render, (void*)&args[i])) {
      perror("pthread_create error");
      exit(1);
    }
    thread_ids[i] = thread_id;
  }

  for (size_t i = 0; i < threads; i++) {
    void* retval;
    pthread_join(thread_ids[i], &retval);
    assert((intptr_t)retval == 0);
  }

  iwrite(fp, buf, sizeof buf, nx, ny);
}
