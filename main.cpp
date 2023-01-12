#include <stdio.h>
#include "util.h"


int main(int argc, char* argv[]) {

    printf("Hello World from the real main test!\n");

    Legion::Runtime::set_top_level_task_id(MAIN_TASK_ID);

    {
        Legion::TaskVariantRegistrar registrar(MAIN_TASK_ID, "main_task");
        registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<main_task>(registrar, "main_task");
    }

    return Legion::Runtime::start(argc, argv);

}
