#include <stdio.h>

void ppm_gradient(void);

int main() {
  ppm_gradient();
  return 0;
}

void ppm_gradient() {
  size_t nx = 200;
  size_t ny = 100;
  printf("P3\n%zu %zu\n255\n", nx, ny);
  for (size_t j = ny; j > 0; j--) {
    for (size_t i = 0; i < nx; i++) {
      float r = (float)i / (float)nx;
      float g = (float)(j-1) / (float)ny;
      float b = 0.5;
      int ir = (int)(255.99 * r);
      int ig = (int)(255.99 * g);
      int ib = (int)(255.99 * b);
      printf("%d %d %d\n", ir, ig, ib);
    }
  }
}
