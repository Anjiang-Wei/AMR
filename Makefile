# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -std=c++11 -fopenmp
OPTFLAGS = -O3 

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v"

LD_FLAGS = -ccbin=mpicxx -Xcompiler -fopenmp 

TARGETS = AMR-H
OBJECTS = main.o 

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $< -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@

AMR-H: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
