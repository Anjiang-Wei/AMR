#include <stdio.h>

// COMPILE-TIME POWER OPERATION
constexpr unsigned ipow(unsigned num, unsigned pow) {
    return (pow >= sizeof(unsigned ) * 8) ? 0 :
        pow == 0 ? 1 : num * ipow(num, pow - 1);
}

// PATCH DATA STRUCTURE
template <unsigned D, unsigned S, typename T>
struct Patch {

  T const *data = new T[ipow(S, D)];

};

// GRID DATA STRUCTURE
template <unsigned D, unsigned S, typename T>
struct Grid {

private:
  unsigned dim[D];
  unsigned numpatch = 1;
  const unsigned numlevel;
  Patch<D, S, T> **patchlist;// = new *Patch<D, S, T>[numlevel];
  
public:
  Grid(const unsigned int dim[], const unsigned numlevel) : numlevel(numlevel) {

    for(auto d = 0; d < D; d++) {
      this->dim[d] = dim[d];
      numpatch *= dim[d];
    }
    // patchlist = new *Patch<D, S, T>[numlevel];

    // REPORT BASE GRID
    {
      printf("\nA %d-dimensional ", D);
      printf(" grid of patches with size %d is created.\n\n", S);
      printf("Number of levels: %d\n\n", numlevel);
      printf("Number of patches:\n");
      for(int level = 0; level < numlevel; level++) {
        unsigned long patches = 1;
        printf("Level %d: ", level);
        for(auto d = 0; d < D; d++) {
          printf("%d ", this->dim[d] * ipow(2, level));
          patches *= this->dim[d] * ipow(2, level);
        }
        printf("| %ld\n", patches);
      }
      printf("\n");
      printf("Number of data points:\n");
      for(int level = 0; level < numlevel; level++) {
        unsigned long points = 1;
        printf("Level %d: ", level);
        for(auto d = 0; d < D; d++) {
          printf("%d ", this->dim[d] * ipow(2, level) * S);
          points *= this->dim[d] * ipow(2, level) * S;
        }
        printf("| %ld ( %.2e GB)\n", points, points * sizeof(T) / 1.e9);
      }

    }
  };
};

#define DTYPE double // FIELD DATA TYPE
#define NDIM 2 // NUMBER OF DIMENSIONS
#define PSIZE 32 // PATCH SIZE

int main(int argc, char *argv[]) {

  printf("Start program!\n");
  printf("\n");
  printf("sizeof(unsigned) = %ld\n", sizeof(unsigned));
  printf("sizeof(long unsigned) = %ld\n", sizeof(long unsigned));
  printf("sizeof(DTYPE) = %ld\n", sizeof(DTYPE));
  printf("sizeof(Patch) = %ld\n", sizeof(Patch<NDIM,PSIZE,DTYPE>));
  printf("sizeof(*Patch) = %ld\n", sizeof(Patch<NDIM,PSIZE,DTYPE>));
  printf("sizeof(Grid) = %ld\n", sizeof(Grid<NDIM,PSIZE,DTYPE>));

  auto numlevel = 7;

  const unsigned dim[NDIM] = {6, 7};

  Grid<NDIM, PSIZE, DTYPE> base(dim, numlevel);

}
