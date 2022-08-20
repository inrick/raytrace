// Taken from Eskil Steenberg's talk "How I program C":
// https://www.youtube.com/watch?v=443UNeGrFoM#t=2h09m55s
static uint32_t f_randi(uint32_t index) {
  index = (index << 13) ^ index;
  return ((index * (index * index * 15731 + 789221) + 1376312589) & 0x7fffffff);
}

static double my_rand(void) {
  static __thread uint32_t rand_seed = 123456789;

  uint32_t result = f_randi(rand_seed);
  // Offset next seed by 1 to avoid a fixed point. The final image still looks
  // good. Without the offset it looks awful.
  rand_seed = result + 1;
  return (double)(result) / (double)(0x7fffffff);
}
