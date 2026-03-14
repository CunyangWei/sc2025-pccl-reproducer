#ifndef ALL_TO_ALLV_H
#define ALL_TO_ALLV_H

#include <mpi.h>
#include <nccl.h>

// NCCL P2P spread-out all-to-allv implementation  
// Uses NCCL point-to-point send/recv operations with variable message sizes
// Note: sendcounts, recvcounts, send_displs, recv_displs are in bytes, not elements
void ncclSpreadOutAllToAllvGPU(void* output, 
                               const void* input, 
                               const int* send_bytes,
                               const int* recv_bytes,
                               const int* send_byte_displs,
                               const int* recv_byte_displs,
                               int rank,
                               int size,
                               ncclComm_t comm);

void ncclAllToAllvGPU(void* output, 
                    const void* input, 
                    const int* send_bytes,
                    const int* recv_bytes,
                    const int* send_byte_displs,
                    const int* recv_byte_displs,
                    int rank,
                    int size,
                    ncclComm_t comm);
 
void AllToAllvGPU_pairwise_sendrecv(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_pairwise_sendrecv_datatype(void* output,
    const void* input,
    const int* sendcounts,
    const int* recvcounts,
    const int* send_displs,
    const int* recv_displs,
    int ,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_pairwise_exchange(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_pairwise_scattered(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_openmpi_pairwise(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_openmpi_basic_linear(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_openmpi_basic_inplace(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_openmpi_inter(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);

void AllToAllvGPU_openmpi_persistent(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm);
#endif // ALL_TO_ALLV_H