#include <stdio.h>

// COMPILE-TIME POWER OPERATION
constexpr unsigned ipow(unsigned num, unsigned pow)
{
    return (pow >= sizeof(unsigned ) * 8) ? 0 :
        pow == 0 ? 1 : num * ipow(num, pow - 1);
}

template <unsigned D, unsigned S, typename T>
struct Patch {

  const unsigned ndata = ipow(S, D);

  T const *data = new T[ndata];


};

template <unsigned D, unsigned S, typename T>
struct Grid {

private:
  unsigned dim[D];
  unsigned numpatch = 1;
  const unsigned numlevel;
  Patch<D, S, T> *patchlist;
  

public:
  Grid(const unsigned dim[], const unsigned numlevel) : numlevel(numlevel) {
    for(auto d = 0; d < D; d++) {
      this->dim[d] = dim[d];
      numpatch *= dim[d];
    }

    // REPORT BASE GRID
    printf("\nA %d-dimensional ( ", D);
    for(auto d = 0; d < D; d++)
      printf("%d ", this->dim[d]);
    printf(") grid of %d patches with size %d is created.\n", numpatch, S);
    printf("The base grid is ");
    for(auto d = 0; d < D; d++)
      printf("%d ", this->dim[d] * S);
    printf(".\n\n");

    patchlist = new Patch<D, S, T>[numpatch];
  };
};

#define DTYPE double // FIELD DATA TYPE
#define NDIM 2 // NUMBER OF DIMENSIONS
#define PSIZE 32 // PATCH SIZE

int main(int argc, char *argv[]) {

  auto numlevel = 5;

  unsigned int dim[NDIM] = {5, 6};

  Grid<NDIM, PSIZE, DTYPE> base(dim, numlevel);

}
