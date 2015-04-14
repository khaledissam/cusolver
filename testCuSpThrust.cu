#include <iostream>
#include <cuda.h>
#include <cusolverSp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main()
{
    cusolverSpHandle_t handle;
    cusolverStatus_t status;
    cusparseStatus_t status2;

    int n = 500000;

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
    thrust::host_vector<double> h_csrValA(nnz,0);
    thrust::host_vector<int> h_csrRowPtrA(n+1,0);
    thrust::host_vector<int> h_csrColIndA(nnz,0);
    thrust::host_vector<double> h_b(n,0);
    thrust::host_vector<double> h_x(n,0);

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
    thrust::device_vector<double> d_csrValA(h_csrValA);
    thrust::device_vector<int> d_csrRowPtrA(h_csrRowPtrA);
    thrust::device_vector<int> d_csrColIndA(h_csrColIndA);
    thrust::device_vector<double> d_b(h_b);
    thrust::device_vector<double> d_x(h_x);

    // raw pointers
    double* ptrValA = thrust::raw_pointer_cast(&d_csrValA[0]);
    int* ptrRowA = thrust::raw_pointer_cast(&d_csrRowPtrA[0]);
    int* ptrColA = thrust::raw_pointer_cast(&d_csrColIndA[0]);
    double* ptrb = thrust::raw_pointer_cast(&d_b[0]);
    double* ptrx = thrust::raw_pointer_cast(&d_x[0]);

    // solve
    cout<<"start solving...\n";
    double tol = 1e-16;
    int reorder = 1;
    int singularity = 0;
    status = cusolverSpDcsrlsvqr(handle,n,nnz,descr,ptrValA,ptrRowA,ptrColA,ptrb,tol,
    				 reorder,ptrx,&singularity);
    // status = cusolverSpDcsrlsvqrHost(handle,n,nnz,descr,h_csrValA,h_csrRowPtrA,h_csrColIndA,h_b,tol,
    // 				     reorder,h_x,&singularity);
    cout<<"end solving...\n";
    thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
    
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
    cusolverSpDestroy(handle);
    
    return 0;
}
