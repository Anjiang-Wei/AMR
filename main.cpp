#include <stdio.h>
#include <cstring>
#include <cmath>

constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double R1 = 1.0;
constexpr double R2 = 2.0;
constexpr double R3 = 3.0;
constexpr double d1 = 0.1;
constexpr double d2 = 0.3;
constexpr double d3 = 0.1;

// COMPILE-TIME POWER OPERATION
constexpr unsigned ipow(unsigned num, unsigned pow) {
    return (pow >= sizeof(unsigned ) * 8) ? 0 :
        pow == 0 ? 1 : num * ipow(num, pow - 1);
}

// PATCH DATA STRUCTURE
template <unsigned D, unsigned S, typename T>
struct Patch {

  T *data = new T[ipow(S, D)];
  const unsigned long ip;
  const unsigned long jp;
  const unsigned long idx;
  
  Patch (unsigned long ip, unsigned long jp, unsigned long idx) : ip(ip), jp(jp), idx(idx) {
    memset(data, 0, ipow(S, D) * sizeof(T));
    for(auto i = 0; i < S; i++)
      for(auto j = 0; j < S; j++) {
        data[i * S + j].i = ip * S + i;
        data[i * S + j].j = jp * S + j;
      }
  }
  inline bool refineReq() const {
    for (auto idx = 0; idx < ipow(S , D); idx++)
      if (data[idx].refine_req)
        return true;
    return false;
  }

  ~Patch() {delete[] data;}
};

// GRID DATA STRUCTURE
template <unsigned D, unsigned S, typename T>
struct Grid {

  unsigned dim[D];
  const unsigned int numlevel;
  unsigned *numpatch = new unsigned[numlevel];
  Patch<D, S, T> ***patchlist;
  
  // PATCH ALLOCATOR
  void allocPatch(unsigned level, unsigned long index)  {
    unsigned long ip = index / (dim[1] * ipow(2, level));
    unsigned long jp = index % (dim[1] * ipow(2, level));
    if ( this->patchlist[level][index] || ip < 0 || ip >= dim[0] * ipow(2, level) || jp < 0 || jp >= dim[1] * ipow(2, level) )
        return;
    this->patchlist[level][index] = new Patch<D, S, T>(ip, jp, index);
    Patch<D,S,T>& p = *this->patchlist[level][index];
    printf("patch at level %u with global index %lu (%lu %lu) is allocated\n", level, index, p.ip, p.jp);
  }

  // PATCH DEALLOCATOR
  void freePatch(unsigned level, unsigned long index) {
    if (this->patchlist[level][index])
      delete this->patchlist[level][index];
    this->patchlist[level][index] = nullptr;
  }

  // ITERATE OVER ALL PATHES ON GIVEN LEVEL
  template<typename F>
  void forEachPatchInLevel( int level, F fcn) {
    for (auto p = 0; p < this->numpatch[level]; p++)
      if (patchlist[level][p])
        fcn(*this->patchlist[level][p]);
  }
  // ITERATE OVER ALL DATA ON GIVEN LEVEL
  template<typename F>
  void forEachPointInLevel( int level, F fcn) {
    for (auto p = 0; p < this->numpatch[level]; p++) {
      if (this->patchlist[level][p])
        for (auto idx = 0; idx < ipow(S, D); idx++)
          fcn(*this->patchlist[level][p], idx); 
    }
  }
  

  // CONSTRUCTOR FOR THE GRID
  Grid(const unsigned dim[], const unsigned numlevel) : numlevel(numlevel) {

    patchlist = new Patch<D, S, T>**[numlevel];

    // FIND NUMBER OF POTENTIAL PATCHES IN EACH LEVEL
    for (auto level = 0; level < numlevel; level++) {
      numpatch[level] = 1;
      for(auto d = 0; d < D; d++) {
        this->dim[d] = dim[d];
        numpatch[level] *= dim[d] * ipow(2, level);
      }
      // ALLOCATE POINTERS IN EACH LEVEL
      patchlist[level] = new Patch<D, S, T>*[numpatch[level]];
      for(auto i = 0; i < numpatch[level]; i++)
        this->patchlist[level][i] = nullptr;
    } 
    // ALLOCATE ALL PATCHES ON LEVEL 0
    for(auto i = 0; i < numpatch[0]; i++)
      this->allocPatch(0, i);

    // REPORT BASE GRID
    {
      printf("\nA %d-dimensional ", D);
      printf(" grid of patches with size %d is created.\n\n", S);
      printf("Number of levels: %d\n\n", numlevel);
      printf("Number of patches:\n");
      for(auto level = 0; level < numlevel; level++) {
        unsigned long patches = 1;
        printf("Level %d: ", level);
        for(auto d = 0; d < D; d++) {
          printf("%u ", this->dim[d] * ipow(2, level));
          patches *= this->dim[d] * ipow(2, level);
        }
        printf("| %ld\n", patches);
      }
      printf("\n");
      printf("Number of data points:\n");
      for(auto level = 0; level < numlevel; level++) {
        unsigned long points = 1;
        printf("Level %d: ", level);
        for(auto d = 0; d < D; d++) {
          printf("%u ", this->dim[d] * ipow(2, level) * S);
          points *= this->dim[d] * ipow(2, level) * S;
        }
        printf("| %ld ( %.2e GB)\n", points, points * sizeof(T) / 1.e9);
      }

    }
  }
};

// FIELD DATA TYPE
struct FieldSpace {
  double x;
  double y;
  double val;
  bool refine_req;
  unsigned long i;
  unsigned long j;
};

inline double refProfile(const double x , const double y) {
    const double r  = sqrt(x*x + y*y);
    const double a1 = (r-R1) / (2.0*d1);
    const double a2 = (r-R2) / (2.0*d2);
    const double a3 = (r-R3) / (2.0*d3);
    return exp(-a1*a1) + exp(-a2*a2) + exp(-a3*a3);
}

inline double smoothIndicator(const double x, const double y) {
    const double r  = sqrt(x * x + y * y);
    const double rd1 = r - R1;
    const double rd2 = r - R2;
    const double rd3 = r - R3;
    const double a1 = rd1 / (2.0 * d1);
    const double a2 = rd2 / (2.0 * d2);
    const double a3 = rd3 / (2.0 * d3);
    return fabs(
        exp(-a1*a1) * (rd1*rd1 - 2.0*d1*d1) / (4.0 * d1*d1*d1*d1)
       +exp(-a2*a2) * (rd2*rd2 - 2.0*d2*d2) / (4.0 * d2*d2*d2*d2)
       +exp(-a3*a3) * (rd3*rd3 - 2.0*d3*d3) / (4.0 * d3*d3*d3*d3)
    );
}

#define NDIM 2 // NUMBER OF DIMENSIONS
#define PSIZE 32 // PATCH SIZE

#define LAMBDA_PATCH [&](Patch<NDIM, PSIZE,FieldSpace>& p)
#define LAMBDA_POINT [&](Patch<NDIM, PSIZE,FieldSpace>& p, unsigned long idx)

int main(int argc, char *argv[]) {

  printf("Start program!\n");
  printf("\n");
  printf("sizeof(unsigned) = %ld\n", sizeof(unsigned));
  printf("sizeof(long unsigned) = %ld\n", sizeof(long unsigned));
  printf("sizeof(FieldSpace) = %ld\n", sizeof(FieldSpace));
  printf("sizeof(Patch) = %ld\n", sizeof(Patch<NDIM,PSIZE,FieldSpace>));
  printf("sizeof(*Patch) = %ld\n", sizeof(Patch<NDIM,PSIZE,FieldSpace>));
  printf("sizeof(Grid) = %ld\n", sizeof(Grid<NDIM,PSIZE,FieldSpace>));

  constexpr int numlevel = 4;

  const double threshold[numlevel] = {10.0, 20.0, 25.0, 30.0};
  const unsigned dim[NDIM] = {12, 12};

  Grid<NDIM, PSIZE, FieldSpace> grid(dim, numlevel);

  for(int l = 0; l < numlevel; l++) {
    grid.forEachPointInLevel(l, LAMBDA_POINT {
       unsigned long Nx = dim[0] * ipow(2, l) * PSIZE;
       unsigned long Ny = dim[1] * ipow(2, l) * PSIZE;
       p.data[idx].x = Lx * ( (p.data[idx].i + 0.5) / Nx - 0.5);
       p.data[idx].y = Ly * ( (p.data[idx].j + 0.5) / Ny - 0.5);

       double indicator = smoothIndicator(p.data[idx].x, p.data[idx].y);
       double reference = refProfile(p.data[idx].x, p.data[idx].y);

       p.data[idx].val = reference;
       p.data[idx].refine_req = indicator > threshold[l];

       //printf("Patch(%lu, %lu) Point(%lu, %lu) (%.3e, %.3e)\n", p.ip, p.jp, p.data[idx].i, p.data[idx].j, p.data[idx].x, p.data[idx].y);
    });
    if(l < numlevel - 1)
      grid.forEachPatchInLevel(l, LAMBDA_PATCH {
        auto patch_idx = [&](unsigned long& ip, unsigned long& jp)->unsigned long {
            return ip * dim[1] * ipow(2, l + 1) + jp;
        };
        if(p.refineReq()) {
          // (0,5) (1,5) | (2,5) (3,5) | (4,5) (5,5)
          // (0,4) (1,4) | (2,4) (3,4) | (4,4) (5,4)
          // ---------------------------------------
          // (0,3) (1,3) | (2,3) (3,3) | (4,3) (5,3)
          // (0,2) (1,2) | (2,2) (3,2) | (4,2) (5,2)
          // ---------------------------------------
          // (0,1) (1,1) | (2,1) (3,1) | (4,1) (5,1)
          // (0,0) (1,0) | (2,0) (3,0) | (4,0) (5,0)
          printf("patch %lu %lu (%lu) in level %d needs to be refined\n", p.ip, p.jp, p.idx, l);
          unsigned long ip_0 = (p.ip - 1) * 2;
          unsigned long jp_0 = (p.jp - 1) * 2;
          for (auto ipr = ip_0; ipr < ip_0 + 6; ipr++)
	    for (auto jpr=jp_0; jpr<jp_0 + 6; jpr++)
              grid.allocPatch(l + 1, patch_idx(ipr, jpr));
        }
     });
  }

  // Dump patches
  constexpr unsigned long idx_anchor_1 = 0;
  constexpr unsigned long idx_anchor_2 = PSIZE * PSIZE - 1;
  FILE * fp;
  fp = fopen("output_patch_layouts.dat", "w");
  fprintf(fp, "%10s, %23s, %23s, %23s, %23s\n", "Level", "x-coord anchor 1", "y-coord anchor 1", "x-coord anchor 2", "y-coord anchor 2");
  for (int l = 0; l < numlevel; l++) {
    grid.forEachPatchInLevel(l, LAMBDA_PATCH {
        const double dx = Lx / (dim[0] * ipow(2, l) * PSIZE);
        const double dy = Ly / (dim[1] * ipow(2, l) * PSIZE);
        const double x1 = p.data[idx_anchor_1].x - 0.5*dx;
        const double y1 = p.data[idx_anchor_1].y - 0.5*dy;
        const double x2 = p.data[idx_anchor_2].x + 0.5*dx;
        const double y2 = p.data[idx_anchor_2].y + 0.5*dy;
        fprintf(fp, "%10d, %23.16e, %23.16e, %23.16e, %23.16e\n", l, x1, y1, x2, y2);
    });
  }
  fclose(fp);
}
