#include <stdio.h>
#include "amrh.h"

#define DTYPE double // FIELD DATA TYPE
#define NDIM 3 // NUMBER OF DIMENSIONS
#define NLEVEL 7 // NUMBER OF LEVELS
#define PSIZE 32 // PATCH SIZE

// COMPILE-TIME POWER OPERATION
constexpr unsigned int ipow(unsigned int num, unsigned int pow)
{
    return (pow >= sizeof(unsigned int)*8) ? 0 :
        pow == 0 ? 1 : num * ipow(num, pow-1);
}

template <unsigned int D, typename T>
struct Patch {

  T *data = new T[ipow(PSIZE, D)];

};

template <unsigned int D, typename T>
struct Base {

private:
  unsigned int dim[D];
  unsigned int nbasepatch = 1;
  Patch<D, T> *patchlist;
  

public:
  Base(unsigned int dim[]) {
    for(auto d = 0; d < D; d++) {
      this->dim[d] = dim[d];
      nbasepatch *= dim[d];
    }
    printf("A Base with dims ");
    for(int d = 0; d < D; d++)
      printf("%d ", this->dim[d]);
    printf("is created with %d base patches.\n", nbasepatch);

    patchlist = new Patch<D,T>[nbasepatch];
  };
};


int main(int argc, char *argv[]) {

  unsigned int dim[NDIM] = {12, 13, 14};
  Base<NDIM, DTYPE> base(dim);

}
