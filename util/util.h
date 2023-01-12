
#ifndef _UTIL_H
#define _UTIL_H

#include "legion_hook.h"

constexpr int MAIN_TASK_ID = 0;

extern void main_task(const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*);

#endif
