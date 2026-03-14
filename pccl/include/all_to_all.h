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

void bruckAllToAllGPU(void* output, 
                      const void* input, 
                      int total_elems, 
                      void* send_buf,
                      void* recv_buf,
                      MPI_Comm comm = MPI_COMM_WORLD);

void ncclAllToAllGPU(void* output, 
                     const void* input, 
                     int total_elems, 
                     int rank,
                     int size,
                     ncclComm_t comm);

void radixRBruckAllToAllGPU(void* output,
                            const void* input, 
                            int total_elems,
                            void* send_buf,
                            void* recv_buf,
                            MPI_Comm comm,
                            int r);

void uniformModifiedRadixRBruckAllToAllGPU(void* output,
                                            const void* input,
                                            int total_elems,
                                            void* send_buf,
                                            void* recv_buf,
                                            MPI_Comm comm,
                                            int r);

// NCCL P2P versions of the algorithms
void ncclSpreadOutAllToAllGPU(void* output, 
                              const void* input, 
                              int total_elems, 
                              void* send_buf,
                              void* recv_buf,
                              int rank,
                              int size,
                              ncclComm_t comm);

void ncclPairwiseExchangeAllToAllGPU(void* output, 
                                     const void* input, 
                                     int total_elems, 
                                     void* send_buf,
                                     void* recv_buf,
                                     int rank,
                                     int size,
                                     ncclComm_t comm);

void ncclBruckAllToAllGPU(void* output, 
                          const void* input, 
                          int total_elems, 
                          void* send_buf,
                          void* recv_buf,
                          int rank,
                          int size,
                          ncclComm_t comm);

void ncclRadixRBruckAllToAllGPU(void* output,
                                const void* input, 
                                int total_elems,
                                void* send_buf,
                                void* recv_buf,
                                int rank,
                                int size,
                                ncclComm_t comm,
                                int r);

void ncclUniformModifiedRadixRBruckAllToAllGPU(void* output,
                                                const void* input,
                                                int total_elems,
                                                void* send_buf,
                                                void* recv_buf,
                                                int rank,
                                                int size,
                                                ncclComm_t comm,
                                                int r);

// Helper function to create NCCL communicator for process groups
ncclComm_t createNCCLComm(int rank, int size, ncclUniqueId id);

#endif // ALL_TO_ALL_H