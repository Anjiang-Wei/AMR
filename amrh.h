

template <typename T, int D>
struct Grid {

  T *data;
  size_t data_size;
  double corner[D];
  int halo;

  Grid(const double (&corner)[D], const int (&size)[D], const int &halo) : halo(halo) {

    data_size = 1;
    for(int dim = 0; dim < D; dim++)
      data_size = data_size * size[dim];

    data = new T[data_size];

    for(int dim = 0; dim < D; dim++)
      this->corner[dim] = corner[dim];

    printf("New Grid of (");
    for(int dim = 0; dim < D; dim++)
      printf(" %d ", size[dim]);
    printf(") %ld %.2e GB\n", data_size, data_size * sizeof(T) / 1.e9);
  }

  void getCoor(const int (&index)[D], double (&coor)[D]) {

    for(int dim = 0; dim < D; dim++)
      coor[dim] = corner[dim] + index[dim];
  }

  
};



