#define _XOPEN_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>

#define bli_fmin(a, b) ((a) < (b) ? (a) : (b))

static double gtod_ref_time_sec = 0.0;
static double* cs = NULL;

double clock_helper()
{
  double the_time, norm_sec;
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)ts.tv_sec;

  norm_sec = (double)ts.tv_sec - gtod_ref_time_sec;

  the_time = norm_sec + ts.tv_nsec * 1.0e-9;

  return the_time;
}

double cclock(void)
{
  return clock_helper();
}

double clock_min_diff(double time_min, double time_start)
{
  double time_min_prev;
  double time_diff;

  // Save the old value.
  time_min_prev = time_min;

  time_diff = cclock() - time_start;

  time_min = bli_fmin(time_min, time_diff);

  // Assume that anything:
  // - under or equal to zero,
  // - under a nanosecond
  // is actually garbled due to the clocks being taken too closely together.
  if (time_min <= 0.0)
    time_min = time_min_prev;
  else if (time_min < 1.0e-9)
    time_min = time_min_prev;

  return time_min;
}

double cache_scrub()
{
  double dtime, dtime_save = 1e10;
  if (cs == NULL) {
    cs = (double*)malloc(LAMP_L3_CACHE_SIZE * sizeof(double));
    srand48((unsigned)time((time_t*)NULL));
    for (int i = 0; i < LAMP_L3_CACHE_SIZE; i++)
      cs[i] = drand48();
  }
  dtime = cclock();
  for (int i = 0; i < LAMP_L3_CACHE_SIZE; i++)
    cs[i] += 1e-3;
  dtime_save = clock_min_diff(dtime_save, dtime);
  return dtime_save;
  return 0.0;
}
