#include "myTime.h"
#include "sys/resource.h"
#include "sys/time.h"
#include "unistd.h"
#include <omp.h>

/*
double clock_time()
{
  double mytime;
  struct rusage rusage;
  if(getrusage(RUSAGE_SELF,&rusage) !=0)
    {return (double) 0.0;}
  mytime = ((rusage.ru_utime.tv_sec + rusage.ru_stime.tv_sec)+
            1.0e-6*(rusage.ru_utime.tv_usec + rusage.ru_stime.tv_usec));
  return mytime;

}
*/

double clock_time() {
  double clock = omp_get_wtime();
  return clock;
}
