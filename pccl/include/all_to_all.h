#ifndef ALL_TO_ALL_H
#define ALL_TO_ALL_H

#include <mpi.h>
#include <nccl.h>

void spreadOutAllToAllGPU(void* output, 
                          const void* input, 
                          int total_elems, 
                          void* send_buf,
                          void* recv_buf,
                          MPI_Comm comm = MPI_COMM_WORLD);

void pairwiseExchangeAllToAllGPU(void* output, 
                                 const void* input, 
                                 int total_elems, 
                                 void* send_buf,
                                 void* recv_buf,
                                 MPI_Comm comm = MPI_COMM_WORLD);

void ringAllToAllGPU(void* output, 
                     const void* input, 
                     int total_elems, 
                     void* send_buf,
                     void* recv_buf,
                     MPI_Comm comm = MPI_COMM_WORLD);

void bruckAllToAllGPU(void* output, 
                      const void* input, 
                      int total_elems, 
                      void* send_buf,
                      void* recv_buf,
                      MPI_Comm comm = MPI_COMM_WORLD);

void ncclAllToAllGPU(void* output, 
                     const void* input, 
                     int total_elems, 
                     ncclComm_t comm,
                     cudaStream_t stream = 0);

// Helper function to create NCCL communicator for process groups
ncclComm_t createNCCLComm(int rank, int size, ncclUniqueId id);

#endif // ALL_TO_ALL_H