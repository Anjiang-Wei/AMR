#include <stdio.h>
#include "amrh.h"

struct Type {
  double x, y, z;
};



int main(int argc, char *argv[]) {

  const int dim = 3;
  double corner[dim] = {-10, -10, -10};
  int size[dim] = {20, 20, 20};

  int halo = 5;

  Grid<Type, dim> base(corner, size, halo);

  double coor[3];
  base.getCoor({1, 1, 2}, coor);

  for(int i = 0; i < dim; i++)
    printf("coor %f\n", coor[i]);
}
