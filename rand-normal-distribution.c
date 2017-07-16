#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* generate a random value weighted within the normal (gaussian) distribution */
static double gauss(void)
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}
/* aggregate 100k cycles and display */
int main(void) {
  int i = 0;
  srandom(time(NULL));
  while (i++ < 100000)
      printf("%f\n",gauss());
  return 0;
}
