#include <cusp/csr_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
// #include <cusp/krylov/cg.h>
// #include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <iostream>
int main(int argc, char *argv[])
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;
    // create an empty sparse matrix structure
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    // construct 2d poisson matrix
    IndexType N = 265;
    cusp::gallery::poisson5pt(A, N, N);
    std::cout << "Generated matrix (poisson5pt) "
              << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n";
    std::cout << "\nSolving with smoothed aggregation preconditioner and jacobi smoother" << std::endl;
    cusp::print(A);
    //cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
    // print AMG statistics
    // M.print();
    // // allocate storage for solution (x) and right hand side (b)
    // cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    // cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
    // // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
    // cusp::monitor<ValueType> monitor(b, 1000, 1e-10);
    // // solve
    // cusp::krylov::cg(A, x, b, monitor, M);
    // // report status
    // monitor.print();
    return 0;
}
