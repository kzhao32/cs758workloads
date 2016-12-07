//SB Related
#ifndef __SIM__HH_
#define __SIM__HH_

//ROI functions
void begin_roi();
void end_roi();


#ifdef __x86_64__
__attribute__ ((noinline))  void begin_roi() {
}
__attribute__ ((noinline))  void end_roi()   {
}

#else
__attribute__ ((noinline))  void begin_roi() {
  __asm__ __volatile__("add x0, x0, 1"); \
}
__attribute__ ((noinline))  void end_roi()   {
   __asm__ __volatile__("add x0, x0, 2"); \
}
#endif

#endif
