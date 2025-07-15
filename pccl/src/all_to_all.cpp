#include <cassert>
#include <cmath>

#include "all_to_all.h"
#include "common.h"

// Performs a spread-out all-to-all on GPU tensors.
// Each process sends block_size elements to every other process.
// - output: CUDA device pointer where the final result will be stored.
// - input: CUDA device pointer to the local data of size total_elems.
// - total_elems: total number of elements in input/output (P * block_size).
// - send_buf: temporary buffer same size as input
// - recv_buf: temporary buffer same size as input
// - comm: MPI communicator (default MPI_COMM_WORLD).
void spreadOutAllToAllGPU(void* output, 
                          const void* input, 
                          int total_elems, 
                          void* send_buf,
                          void* recv_buf,
                          MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Create CUDA event for synchronization
    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    // Copy input to send buffer
    CUDA_CHECK(cudaMemcpyAsync(send_buf, 
                             input, 
                             total_elems, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Initialize output buffer with local data
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + rank * block_size, 
                             static_cast<const char*>(input) + rank * block_size, 
                             block_size, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Use non-blocking communication for better overlap
    MPI_Request requests[2 * (size - 1)];
    int req_idx = 0;

    CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

    // Perform all-to-all exchanges
    for (int step = 1; step < size; step++) {
        
        int send_partner = (rank + step) % size;
        int recv_partner = (rank - step + size) % size;

        // Send data
        int send_offset = send_partner * block_size;
        void *send_ptr = (char *)send_buf + send_offset;

        // Synchronize CUDA stream before MPI communication
        // CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        // CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

        MPI_Isend(send_ptr, block_size, MPI_BYTE, send_partner, step,
                  comm, &requests[req_idx++]);

        // Receive data
        int recv_offset = recv_partner * block_size;
        void *recv_ptr = (char *)output + recv_offset;

        MPI_Irecv(recv_ptr, block_size, MPI_BYTE, recv_partner, step,
                  comm, &requests[req_idx++]);
        
    }
    MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}

// Performs a pairwise exchange all-to-all on GPU tensors.
// Processes are paired and exchange data simultaneously to reduce congestion.
void pairwiseExchangeAllToAllGPU(void* output, 
                                 const void* input, 
                                 int total_elems, 
                                 void* send_buf,
                                 void* recv_buf,
                                 MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Create CUDA event for synchronization
    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    // Copy input to output initially
    CUDA_CHECK(cudaMemcpyAsync(output, 
                             input, 
                             total_elems, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Pairwise exchange in log(P) phases
    int nsteps = (int)ceil(log2(size));

    for (int step = 0; step < nsteps; step++)
    {
        int partner = rank ^ (1 << step);

        if (partner >= size)
            continue;

        int send_count = 0;
        int recv_count = 0;
        
        char *send_ptr = (char *)send_buf;
        for (int i = 0; i < size; i++)
        {
            if ((i & (1 << step)) != (rank & (1 << step)))
            {
                CUDA_CHECK(cudaMemcpyAsync(send_ptr, (char *)output + i * block_size, block_size, cudaMemcpyDeviceToDevice, stream));
                send_ptr += block_size;
                send_count += block_size;
            }
        }

        for (int i = 0; i < size; i++)
        {
            if ((i & (1 << step)) != (rank & (1 << step)))
            {
                recv_count += block_size;
            }
        }

        // Synchronize CUDA stream before MPI communication
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
        
        // Exchange blocks with partner
        MPI_Sendrecv(send_buf, send_count, MPI_CHAR, partner, 0,
            recv_buf, recv_count, MPI_CHAR, partner, 0,
            comm, MPI_STATUS_IGNORE);
        
        // Copy received data to correct position
        char *recv_ptr = (char *)recv_buf;
        for (int i = 0; i < size; i++)
        {
            if ((i & (1 << step)) != (rank & (1 << step)))
            {
                CUDA_CHECK(cudaMemcpyAsync((char *)output + i * block_size, recv_ptr, block_size, cudaMemcpyDeviceToDevice, stream));
                recv_ptr += block_size;
            }
        }
    }
    MPI_Barrier(comm);
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}

// Performs a ring-based all-to-all on GPU tensors.
// Data flows around a logical ring topology.
void ringAllToAllGPU(void* output, 
                     const void* input, 
                     int total_elems, 
                     void* send_buf,
                     void* recv_buf,
                     MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    // Ring algorithm: P-1 steps, each process sends one block around the ring
    MPI_Request requests[size];
    for (int i = 0; i < size; i++)
    {
        MPI_Isend(input + i * block_size, block_size, MPI_BYTE, i, 0, comm, &requests[i]);
        MPI_Irecv(output + i * block_size, block_size, MPI_BYTE, i, 0, comm, &requests[i]);
    }
    MPI_Waitall(size, requests, MPI_STATUSES_IGNORE);
    MPI_Barrier(comm);
}

// Performs a Bruck all-to-all on GPU tensors.
// Uses a recursive decomposition approach with log(P) communication steps.
void bruckAllToAllGPU(void* output, 
                      const void* input, 
                      int total_elems, 
                      void* send_buf,
                      void* recv_buf,
                      MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Create CUDA event for synchronization
    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    void *R       = send_buf;
    void *T_send  = recv_buf;
    void *T_recv;
    CUDA_CHECK(cudaMalloc(&T_recv, ((size + 1) / 2) * block_size));

    const void *S = input;
    void *O       = output;

    for (int i = 0; i < size; ++i) {
        int src = (rank + i) % size;
        CUDA_CHECK(cudaMemcpyAsync(R + i * block_size,
               S + src * block_size,
               block_size,
               cudaMemcpyDeviceToDevice,
               stream));
    }

    int *SB = (int *)malloc(sizeof(int) * ((size + 1) / 2));

    for (int k = 1; k < size; k <<= 1)
    {
        int NB = 0;
        for (int i = k; i < size; ++i)
            if (i & k)
                SB[NB++] = i;

        const int send_bytes = NB * block_size;

        for (int idx = 0; idx < NB; ++idx)
            CUDA_CHECK(cudaMemcpyAsync(T_send + idx * block_size,
                   R + SB[idx] * block_size,
                   block_size,
                   cudaMemcpyDeviceToDevice,
                   stream));

        const int sendproc = (rank + k) % size;
        const int recvproc = (rank - k + size) % size;

        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

        // CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Sendrecv(T_send,  send_bytes, MPI_BYTE, sendproc, 0,
                     T_recv,  send_bytes, MPI_BYTE, recvproc, 0,
                     comm, MPI_STATUS_IGNORE);

        for (int idx = 0; idx < NB; ++idx)
            CUDA_CHECK(cudaMemcpyAsync(R + SB[idx] * block_size,
                   T_recv + idx * block_size,
                   block_size,
                   cudaMemcpyDeviceToDevice,
                   stream));
    }

    free(SB);
    cudaFree(T_recv);

    for (int i = 0; i < size; ++i) {
        int src = (rank - i + size) % size;
        CUDA_CHECK(cudaMemcpyAsync(O + i * block_size,
               R + src * block_size,
               block_size,
               cudaMemcpyDeviceToDevice,
               stream));
    }

    MPI_Barrier(comm);
    
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}

// Performs a direct NCCL all-to-all on GPU tensors using NCCL grouped operations.
// This is the most efficient implementation for intra-node communication.
void ncclAllToAllGPU(void* output, 
                     const void* input, 
                     int total_elems, 
                     ncclComm_t comm,
                     cudaStream_t stream) {
    
    int rank, nranks;
    NCCL_CHECK(ncclCommUserRank(comm, &rank));
    NCCL_CHECK(ncclCommCount(comm, &nranks));

    assert(total_elems % nranks == 0 && "Input tensor size must be divisible by number of ranks");
    int block_size = total_elems / nranks;

    // Use NCCL grouped operations for optimal performance
    NCCL_CHECK(ncclGroupStart());
    
    for (int r = 0; r < nranks; r++) {
        // Send block r to rank r
        const char* send_ptr = static_cast<const char*>(input) + r * block_size;
        NCCL_CHECK(ncclSend(send_ptr, block_size, ncclInt8, r, comm, stream));
        
        // Receive block from rank r
        char* recv_ptr = static_cast<char*>(output) + r * block_size;
        NCCL_CHECK(ncclRecv(recv_ptr, block_size, ncclInt8, r, comm, stream));
    }
    
    NCCL_CHECK(ncclGroupEnd());
}