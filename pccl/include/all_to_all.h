#ifndef ALL_TO_ALL_H
#define ALL_TO_ALL_H

#include <mpi.h>

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

#endif // ALL_TO_ALL_H