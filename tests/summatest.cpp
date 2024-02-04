#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif


// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
  typedef SpDCCols < int64_t, NT > DCCols;
  typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(argc < 2){
    if(myrank == 0)
    {
      cout << "Usage: ./<Binary> <MatrixA> " << endl;
    }
    MPI_Finalize();
    return -1;
  }
  else {
    double vm_usage, resident_set;
    string Aname(argv[1]);
    string Bname(argv[2]);
    if(myrank == 0){
      fprintf(stderr, "Data: %s\n", argv[1]);
      fprintf(stderr, "Data: %s\n", argv[2]);
    }
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

    double t0, t1, t2;

    SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);
    SpParMat<int64_t, double, SpDCCols < int64_t, double >> MB(fullWorld);

    //// Read matrix market files
    //t0 = MPI_Wtime();
    //M.ParallelReadMM(Aname, true, maximum<double>());
    //t1 = MPI_Wtime();
    //if(myrank == 0) fprintf(stderr, "Time taken to read file: %lf\n", t1-t0);
    //t0 = MPI_Wtime();
    //FullyDistVec<int64_t, int64_t> p( M.getcommgrid() );
    //FullyDistVec<int64_t, int64_t> q( M.getcommgrid() );
    //p.iota(M.getnrow(), 0);
    //q.iota(M.getncol(), 0);
    //p.RandPerm();
    //q.RandPerm();
    //(M)(p,q,true);// in-place permute to save memory
    //t1 = MPI_Wtime();
    //if(myrank == 0) fprintf(stderr, "Time taken to permuatate input: %lf\n", t1-t0);

    // Read labelled triple files
    t0 = MPI_Wtime();
    M.ReadGeneralizedTuples(Aname, maximum<double>());
    fprintf(stderr, "first file read completed");
    MB.ReadGeneralizedTuples(Bname, maximum<double>());
    fprintf(stderr, "second file read completed");
    t1 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "Time taken to read file: %lf\n", t1-t0);

    typedef PlusTimesSRing<double, double> PTFF;

    // Run 2D multiplication to compare against
    SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(M);
    SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(MB);
    SpParMat<int64_t, double, SpDCCols < int64_t, double >> C2D =
        Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
        (A2D, B2D);
    t2 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "Time taken to multiply file: %lf\n", t2-t1);
    if(myrank == 0) fprintf(stderr, "2D Multiplication done \n");

//    // Increase number of layers 1 -> 4 -> 16
//    for(int layers = 1; layers <= 16; layers = layers * 4){
//      // Create two copies of input matrix which would be used in multiplication
//      SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2(M);
//      SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2(M);
//
//      // Convert 2D matrices to 3D
//      SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2, layers, true, false);
//      SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B2, layers, false, false);
//      if(myrank == 0) fprintf(stderr, "Running 3D with %d layers\n", layers);
//
//      SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> C3D =
//          Mult_AnXBn_SUMMA3D<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//          (A3D, B3D);
//
//      SpParMat<int64_t, double, SpDCCols < int64_t, double >> C3D2D = C3D.Convert2D();
//      if(C2D == C3D2D){
//        if(myrank == 0) fprintf(stderr, "Correct!\n");
//      }
//      else{
//        if(myrank == 0) fprintf(stderr, "Not correct!\n");
//      }
//    }

  }
  MPI_Finalize();
  return 0;
}
