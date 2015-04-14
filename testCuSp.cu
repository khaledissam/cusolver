#include <iostream>
#include <cuda.h>
#include <cusolverSp.h>

using namespace std;

int main()
{
    cusolverSpHandle_t handle;
    cusolverStatus_t status;
    cusparseStatus_t status2;

    int n = 10000;

    // create handle
    status = cusolverSpCreate(&handle);

    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to create cusolver handle\n";
	return -1;
    } else {
	cerr<<"succeeded to create cusolverhandle\n";
    }

    // create matrix descriptor
    cusparseMatDescr_t descr;
    status2 = cusparseCreateMatDescr(&descr);
    if(status2 != CUSPARSE_STATUS_SUCCESS) {
	cerr<<"failed to create matrix descriptor\n";
	return -1;
    } else {
	cerr<<"succeeded to create matrix descriptor\n";
    }

    // allocate A and b on host
    int nnz = 3*n-2;
    double* h_csrValA = new double[nnz];
    int* h_csrRowPtrA = new int[n+1];
    int* h_csrColIndA = new int[nnz];
    double* h_b = new double[n];
    double* h_x = new double[n];

    h_csrValA[0] = 4;
    h_csrValA[1] = -1;
    h_csrRowPtrA[0] = 0;
    h_csrRowPtrA[1] = 2;
    h_csrColIndA[0] = 0;
    h_csrColIndA[1] = 1;
    h_b[0] = 3;
    for(int i=1; i<n-1; i++) {
	h_csrRowPtrA[i] = 2+3*i-3;
	h_csrValA[2+3*i-3] = -1;
	h_csrValA[2+3*i-2] = 4;
	h_csrValA[2+3*i-1] = -1;
	h_csrColIndA[2+3*i-3] = i-1;
	h_csrColIndA[2+3*i-2] = i;
	h_csrColIndA[2+3*i-1] = i+1;
	h_b[i] = 2;
    }
    h_csrValA[nnz-2] = -1;
    h_csrValA[nnz-1] = 4;
    h_csrColIndA[nnz-2] = n-2;
    h_csrColIndA[nnz-1] = n-1;
    h_csrRowPtrA[n-1] = nnz-2;
    h_csrRowPtrA[n] = nnz;
    h_b[n-1] = 3;

    // allocate A and b on device
    double* d_csrValA, *d_b, *d_x;
    int* d_csrRowPtrA, *d_csrColIndA;
    cudaMalloc((void**)&d_csrValA, nnz*sizeof(double));
    cudaMalloc((void**)&d_b, n*sizeof(double));
    cudaMalloc((void**)&d_x, n*sizeof(double));
    cudaMalloc((void**)&d_csrRowPtrA, (n+1)*sizeof(int));
    cudaMalloc((void**)&d_csrColIndA, nnz*sizeof(int));

    // copy to device
    cudaMemcpy(d_csrValA, h_csrValA, nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, h_csrColIndA, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);
    

    // solve
    cout<<"start solving...\n";
    double tol = 1e-16;
    int reorder = 1;
    int singularity = 0;
    status = cusolverSpDcsrlsvqr(handle,n,nnz,descr,d_csrValA,d_csrRowPtrA,d_csrColIndA,d_b,tol,
    				 reorder,d_x,&singularity);
    // status = cusolverSpDcsrlsvqrHost(handle,n,nnz,descr,h_csrValA,h_csrRowPtrA,h_csrColIndA,h_b,tol,
    // 				     reorder,h_x,&singularity);
    cout<<"end solving...\n";
    cudaMemcpy(h_x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);
    cout<<"singularity = "<<singularity<<"\n";
    if(status != CUSOLVER_STATUS_SUCCESS) {
	cerr<<"failed to solve\n";
	return -1;
    } else {
	cerr<<"succeeded to solve\n";
    }

    cout<<"x[0] = "<<h_x[0]<<"\n";
    cout<<"x[n-1] = "<<h_x[n-1]<<"\n";

    // clear
    delete [] h_csrValA;
    delete [] h_csrRowPtrA;
    delete [] h_csrColIndA;
    delete [] h_b;
    delete [] h_x;
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_b);
    cudaFree(d_x);
    cusolverSpDestroy(handle);
    
    return 0;
}
