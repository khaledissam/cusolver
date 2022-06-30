#include <iostream>
#include <cuda.h>
#include <cusolverDn.h>
using namespace std;

int main()
{
    cusolverDnHandle_t handle;
    cusolverStatus_t status;

    // create handle
    status = cusolverDnCreate(&handle);

    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to create cusolver handle\n";
	return -1;
    } else {
	cerr<<"succeeded to create cusolverhandle\n";
    }

    // allocate A and B on host
    int m = 24000;
    int n = 24000;

    double* h_A = new double[m*n];
    double* h_B = new double[n];

    for(int j=0; j<n; j++) {
	for(int i=0; i<m; i++) {
	    if(i!=j) h_A[j*m+i] = -1;
	    else h_A[j*m+i] = 4;
	}
	h_B[j] = -n+5;
    }

    // allocate A and B on device
    double* d_A, *d_B;
    cudaMalloc((void**) &d_A, m*n*sizeof(double));
    cudaMalloc((void**) &d_B, n*sizeof(double));

    // copy A and B to GPU
    cudaMemcpy(d_A, h_A, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*sizeof(double), cudaMemcpyHostToDevice);

    // get workspace size
    int lda = m, Lwork=0;
    status = cusolverDnDgetrf_bufferSize(handle,m,n,d_A,lda,&Lwork);
    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to get workspace size\n";
	return -1;
    } else {
	cerr<<"workspace size = "<<Lwork<<"\n";
    }

    // set up workspace
    double* d_workspace;
    int* d_ipiv, *d_info;
    cudaMalloc((void**)&d_workspace, Lwork*sizeof(double));
    cudaMalloc((void**)&d_ipiv, m*sizeof(int));
    cudaMalloc((void**)&d_info, sizeof(int));

    // LU decomposition
    status = cusolverDnDgetrf(handle,m,n,d_A,lda,d_workspace,d_ipiv,d_info);
    int h_info = 0;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to LU, info = "<<h_info<<"\n";
	return -1;
    } else {
	cerr<<"done LU, info = "<<h_info<<"\n";
    }

    // solve
    int ldb = n, nrhs = 1;
    cublasOperation_t trans = CUBLAS_OP_N;
    status = cusolverDnDgetrs(handle,trans,n,nrhs,d_A,lda,d_ipiv,d_B,ldb,d_info);
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to solve, info = "<<h_info<<"\n";
	return -1;
    } else {
	cerr<<"solved, info = "<<h_info<<"\n";
    }

    // check results
    cudaMemcpy(h_B, d_B, n*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=1; i<n; i++) {
	if(abs(h_B[i] - h_B[0])>1E-7) {
	    cerr<<"wrong answer, B["<<i<<"] = "<<h_B[i]<<" , B[0] = "<<h_B[0]<<"\n";
	    return -1;
	}
    }
    cerr<<"B[0] = "<<h_B[0]<<"\n";

    // clear
    delete [] h_A;
    delete [] h_B;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_workspace);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cusolverDnDestroy(handle);
    
    return 0;
}
