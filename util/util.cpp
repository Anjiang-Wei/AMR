
#include <vector>
#include "legion_hook.h"

void main_task(const Task* task, const std::vector<PhysicalRegion>& rgns, Context ctx, Runtime* rt) {
    int argc    = Legion::Runtime::get_input_args().argc;
    char** argv = Legion::Runtime::get_input_args().argv;

    printf("HelloWorld from Legion main_task!\n TODO: Change util/util.cpp/main_task\n\n");
}

