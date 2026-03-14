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

int myPow(int x, unsigned int p) {
    if (p == 0) return 1;
    if (p == 1) return x;
    
    int tmp = myPow(x, p/2);
    if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

void convert10tob(int w, int N, int b, int* result) {
    // Initialize result array to 0
    for (int i = 0; i < w; i++) {
        result[i] = 0;
    }
    
    int i = 0;
    while(N && i < w) {
        result[i++] = (N % b);
        N /= b;
    }
}

// Performs a radix-r Bruck all-to-all on GPU tensors.
// Uses radix-r representation for more flexible communication patterns.
// - output: CUDA device pointer where the final result will be stored.
// - input: CUDA device pointer to the local data of size total_elems.
// - total_elems: total number of elements in input/output (P * block_size).
// - send_buf: temporary buffer same size as input
// - recv_buf: temporary buffer same size as input
// - comm: MPI communicator (default MPI_COMM_WORLD).
// - r: radix parameter for the algorithm (must be >= 2).
void radixRBruckAllToAllGPU(void *output,
                            const void *input, 
                            int total_elems,
                            void *send_buf,
                            void *recv_buf,
                            MPI_Comm comm,
                            int r)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    assert(total_elems % nprocs == 0 && "total_elems must be divisible by number of processes");
    assert(r >= 2 && "radix parameter r must be >= 2");

    int unit_size = total_elems / nprocs;
    int w = ceil(log(nprocs) / log(r)); 
    int nlpow = myPow(r, w-1);
    int d = (myPow(r, w) - nprocs) / nlpow; 

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Create CUDA event for synchronization
    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    // Convert rank to base r representation
    int* rank_r_reps = (int*) malloc(nprocs * w * sizeof(int));
    for (int i = 0; i < nprocs; i++) {
        convert10tob(w, i, r, &rank_r_reps[i*w]);
    }

    // Copy own data to output
    CUDA_CHECK(cudaMemcpyAsync((char*)output + rank*unit_size, (const char*)input + rank*unit_size, unit_size, cudaMemcpyDeviceToDevice, stream));

    // Create local index array after rotation
    int* rotate_array = (int*)malloc(nprocs * sizeof(int));
    for (int i = 0; i < nprocs; i++)
        rotate_array[i] = (2*rank-i+nprocs)%nprocs;

    char* stemp_buffer;
    CUDA_CHECK(cudaMalloc(&stemp_buffer, nlpow * unit_size)); 
    char* rtemp_buffer;
    CUDA_CHECK(cudaMalloc(&rtemp_buffer, nlpow * unit_size)); 

    int* sent_blocks;
    sent_blocks = (int*)malloc(nlpow * sizeof(int));
    int di = 0;
    int ci = 0;

    // Copy input to send buffer for manipulation
    CUDA_CHECK(cudaMemcpyAsync(send_buf, input, total_elems, cudaMemcpyDeviceToDevice, stream));

    // Communication steps = (r - 1)w - d
    for (int x = 0; x < w; x++) {
        int ze = (x == w - 1)? r - d: r;
        for (int z = 1; z < ze; z++) {

            // Get the sent data-blocks
            di = 0;
            ci = 0;
            for (int i = 0; i < nprocs; i++) {
                if (rank_r_reps[i*w + x] == z){
                    int sbs =(i + rank) % nprocs;
                    sent_blocks[di++] = sbs;
                    CUDA_CHECK(cudaMemcpyAsync(&stemp_buffer[unit_size*ci++], 
                            &((char*)send_buf)[rotate_array[sbs]*unit_size], unit_size, cudaMemcpyDeviceToDevice, stream));
                }
            }

            int distance = z * myPow(r, x);
            int recv_proc = (rank + distance) % nprocs; 
            int send_proc = (rank - distance + nprocs) % nprocs; 
            long long comm_size = di * unit_size;

            // Synchronize CUDA stream before MPI communication
            CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

            MPI_Sendrecv(stemp_buffer, comm_size, MPI_CHAR, send_proc, 0, 
            rtemp_buffer, comm_size, MPI_CHAR, recv_proc, 0, 
            comm, MPI_STATUS_IGNORE);

            for (int i = 0; i < di; i++) {
                long long offset = rotate_array[sent_blocks[i]] * unit_size;
                CUDA_CHECK(cudaMemcpyAsync((char*)output + (sent_blocks[i]*unit_size), 
                rtemp_buffer + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync((char*)send_buf + offset, 
                rtemp_buffer + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    free(rank_r_reps);
    free(rotate_array);
    CUDA_CHECK(cudaFree(stemp_buffer));
    CUDA_CHECK(cudaFree(rtemp_buffer));
    free(sent_blocks);
    
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}


void uniformModifiedRadixRBruckAllToAllGPU(void *output,
                                            const void *input,
                                            int total_elems,
                                            void *send_buf,
                                            void *recv_buf,
                                            MPI_Comm comm,
                                            int r)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    assert(total_elems % nprocs == 0 && "total_elems must be divisible by number of processes");
    assert(r >= 2 && "radix parameter r must be >= 2");

    int unit_size = total_elems / nprocs;
    int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
    int nlpow = myPow(r, w-1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Create CUDA event for synchronization
    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    // Initial rotation step - reorder data
    for (int i = 0; i < nprocs; i++) {
        int index = (2*rank-i+nprocs)%nprocs;
        CUDA_CHECK(cudaMemcpyAsync((char*)output + (index*unit_size), (const char*)input + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
    }

    int* sent_blocks = (int*)malloc(nlpow * sizeof(int));
    int di = 0;
    int ci = 0;

    char* temp_buffer;
    CUDA_CHECK(cudaMalloc(&temp_buffer, nlpow * unit_size)); // temporary buffer
    char* sendbuf;
    CUDA_CHECK(cudaMalloc(&sendbuf, nlpow * unit_size)); // received data buffer

    int spoint = 1, distance = 1, next_distance = r;
    for (int x = 0; x < w; x++) {
        for (int z = 1; z < r; z++) {

            // get the sent data-blocks
            // copy blocks which need to be sent at this step
            spoint = z * distance;
            if (spoint > nprocs - 1) {break;}
                di = 0; ci = 0;
            for (int i = spoint; i < nprocs; i += next_distance) {
                for (int j = i; j < (i+distance); j++) {
                    if (j > nprocs - 1 ) { break; }
                    int id = (j + rank) % nprocs;
                    sent_blocks[di++] = id;
                    CUDA_CHECK(cudaMemcpyAsync(&temp_buffer[unit_size*ci++], &((char*)output)[id*unit_size], unit_size, cudaMemcpyDeviceToDevice, stream));
                }
            }

            // send and receive
            int recv_proc = (rank + spoint) % nprocs; // receive data from rank - 2^step process
            int send_proc = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process
            long long comm_size = di * unit_size;

            // Synchronize CUDA stream before MPI communication
            CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

            MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, 
            sendbuf, comm_size, MPI_CHAR, recv_proc, 0, 
            comm, MPI_STATUS_IGNORE);

            // replace with received data
            for (int i = 0; i < di; i++) {
                long long offset = sent_blocks[i] * unit_size;
                CUDA_CHECK(cudaMemcpyAsync((char*)output + offset, sendbuf + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        distance *= r;
        next_distance *= r;
    }

    CUDA_CHECK(cudaFree(temp_buffer));
    CUDA_CHECK(cudaFree(sendbuf));
    free(sent_blocks);

    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}


// Performs a direct NCCL all-to-all on GPU tensors using NCCL grouped operations.
// This is the most efficient implementation for intra-node communication.
void ncclAllToAllGPU(void* output, 
    const void* input, 
    int total_elems, 
    int rank,
    int size,
    ncclComm_t comm) {

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of ranks");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Use NCCL grouped operations for optimal performance
    NCCL_CHECK(ncclGroupStart());

    for (int r = 0; r < size; r++) {
        // Send block r to rank r
        const char* send_ptr = static_cast<const char*>(input) + r * block_size;
        NCCL_CHECK(ncclSend(send_ptr, block_size, ncclInt8, r, comm, stream));

        // Receive block from rank r
        char* recv_ptr = static_cast<char*>(output) + r * block_size;
        NCCL_CHECK(ncclRecv(recv_ptr, block_size, ncclInt8, r, comm, stream));
    }

    NCCL_CHECK(ncclGroupEnd());
}

// NCCL P2P spread-out all-to-all implementation
// Uses NCCL point-to-point send/recv operations instead of MPI
void ncclSpreadOutAllToAllGPU(void* output, 
                              const void* input, 
                              int total_elems, 
                              void* send_buf,
                              void* recv_buf,
                              int rank,
                              int size,
                              ncclComm_t comm) {
    
    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Copy input to send buffer
    // CUDA_CHECK(cudaMemcpyAsync(send_buf, 
    //                          input, 
    //                          total_elems, 
    //                          cudaMemcpyDeviceToDevice, 
    //                          stream));

    // Initialize output buffer with local data
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + rank * block_size, 
                             static_cast<const char*>(input) + rank * block_size, 
                             block_size, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Perform all-to-all exchanges using NCCL P2P
    for (int step = 1; step < size; step++) {
        
        int send_partner = (rank + step) % size;
        int recv_partner = (rank - step + size) % size;

        // Use NCCL grouped operations for this step
        NCCL_CHECK(ncclGroupStart());

        // Send data
        int send_offset = send_partner * block_size;
        const char *send_ptr = static_cast<const char*>(input) + send_offset;
        NCCL_CHECK(ncclSend(send_ptr, block_size, ncclInt8, send_partner, comm, stream));

        // Receive data  
        int recv_offset = recv_partner * block_size;
        char *recv_ptr = static_cast<char*>(output) + recv_offset;
        NCCL_CHECK(ncclRecv(recv_ptr, block_size, ncclInt8, recv_partner, comm, stream));
        
        NCCL_CHECK(ncclGroupEnd());
    }
}

// NCCL P2P pairwise exchange all-to-all implementation
// Uses NCCL point-to-point operations with pairwise exchange pattern
void ncclPairwiseExchangeAllToAllGPU(void* output, 
                                     const void* input, 
                                     int total_elems, 
                                     void* send_buf,
                                     void* recv_buf,
                                     int rank,
                                     int size,
                                     ncclComm_t comm) {
    
    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
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
        
        // Pack data to be sent to partner
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

        // Exchange blocks with partner using NCCL
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(send_buf, send_count, ncclInt8, partner, comm, stream));
        NCCL_CHECK(ncclRecv(recv_buf, recv_count, ncclInt8, partner, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        
        // Unpack received data to correct positions
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
}

// NCCL P2P Bruck all-to-all implementation
// Uses NCCL point-to-point operations with Bruck recursive decomposition
void ncclBruckAllToAllGPU(void* output, 
                          const void* input, 
                          int total_elems, 
                          void* send_buf,
                          void* recv_buf,
                          int rank,
                          int size,
                          ncclComm_t comm) {
    
    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    
    void *R       = send_buf;
    void *T_send  = recv_buf;
    void *T_recv;
    CUDA_CHECK(cudaMalloc(&T_recv, ((size + 1) / 2) * block_size));

    const void *S = input;
    void *O       = output;

    // Initial rotation - copy data from S to R in rotated order
    for (int i = 0; i < size; ++i) {
        int src = (rank + i) % size;
        CUDA_CHECK(cudaMemcpyAsync((char*)R + i * block_size,
               (const char*)S + src * block_size,
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

        // Pack data to be sent
        for (int idx = 0; idx < NB; ++idx)
            CUDA_CHECK(cudaMemcpyAsync((char*)T_send + idx * block_size,
                   (char*)R + SB[idx] * block_size,
                   block_size,
                   cudaMemcpyDeviceToDevice,
                   stream));

        const int sendproc = (rank + k) % size;
        const int recvproc = (rank - k + size) % size;

        // Exchange data using NCCL P2P
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(T_send, send_bytes, ncclInt8, sendproc, comm, stream));
        NCCL_CHECK(ncclRecv(T_recv, send_bytes, ncclInt8, recvproc, comm, stream));
        NCCL_CHECK(ncclGroupEnd());

        // Unpack received data
        for (int idx = 0; idx < NB; ++idx)
            CUDA_CHECK(cudaMemcpyAsync((char*)R + SB[idx] * block_size,
                   (char*)T_recv + idx * block_size,
                   block_size,
                   cudaMemcpyDeviceToDevice,
                   stream));
    }

    free(SB);
    cudaFree(T_recv);

    // Final rotation - copy from R to O in reverse rotated order
    for (int i = 0; i < size; ++i) {
        int src = (rank - i + size) % size;
        CUDA_CHECK(cudaMemcpyAsync((char*)O + i * block_size,
               (char*)R + src * block_size,
               block_size,
               cudaMemcpyDeviceToDevice,
               stream));
    }
}

// NCCL P2P radix-r Bruck all-to-all implementation
// Uses NCCL point-to-point operations with radix-r Bruck algorithm
void ncclRadixRBruckAllToAllGPU(void *output,
                                const void *input, 
                                int total_elems,
                                void *send_buf,
                                void *recv_buf,
                                int rank,
                                int size,
                                ncclComm_t comm,
                                int r)
{
    assert(total_elems % size == 0 && "total_elems must be divisible by number of processes");
    assert(r >= 2 && "radix parameter r must be >= 2");

    int unit_size = total_elems / size;
    int w = ceil(log(size) / log(r)); 
    int nlpow = myPow(r, w-1);
    int d = (myPow(r, w) - size) / nlpow; 

    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Convert rank to base r representation
    int* rank_r_reps = (int*) malloc(size * w * sizeof(int));
    for (int i = 0; i < size; i++) {
        convert10tob(w, i, r, &rank_r_reps[i*w]);
    }

    // Copy own data to output
    CUDA_CHECK(cudaMemcpyAsync((char*)output + rank*unit_size, (const char*)input + rank*unit_size, unit_size, cudaMemcpyDeviceToDevice, stream));

    // Create local index array after rotation
    int* rotate_array = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        rotate_array[i] = (2*rank-i+size)%size;

    char* stemp_buffer;
    CUDA_CHECK(cudaMalloc(&stemp_buffer, nlpow * unit_size)); 
    char* rtemp_buffer;
    CUDA_CHECK(cudaMalloc(&rtemp_buffer, nlpow * unit_size)); 

    int* sent_blocks;
    sent_blocks = (int*)malloc(nlpow * sizeof(int));
    int di = 0;
    int ci = 0;

    // Copy input to send buffer for manipulation
    CUDA_CHECK(cudaMemcpyAsync(send_buf, input, total_elems, cudaMemcpyDeviceToDevice, stream));

    // Communication steps = (r - 1)w - d
    for (int x = 0; x < w; x++) {
        int ze = (x == w - 1)? r - d: r;
        for (int z = 1; z < ze; z++) {

            // Get the sent data-blocks
            di = 0;
            ci = 0;
            for (int i = 0; i < size; i++) {
                if (rank_r_reps[i*w + x] == z){
                    int sbs =(i + rank) % size;
                    sent_blocks[di++] = sbs;
                    CUDA_CHECK(cudaMemcpyAsync(&stemp_buffer[unit_size*ci++], 
                            &((char*)send_buf)[rotate_array[sbs]*unit_size], unit_size, cudaMemcpyDeviceToDevice, stream));
                }
            }

            int distance = z * myPow(r, x);
            int recv_proc = (rank + distance) % size; 
            int send_proc = (rank - distance + size) % size; 
            long long comm_size = di * unit_size;

            // Exchange data using NCCL P2P
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(stemp_buffer, comm_size, ncclInt8, send_proc, comm, stream));
            NCCL_CHECK(ncclRecv(rtemp_buffer, comm_size, ncclInt8, recv_proc, comm, stream));
            NCCL_CHECK(ncclGroupEnd());

            for (int i = 0; i < di; i++) {
                long long offset = rotate_array[sent_blocks[i]] * unit_size;
                CUDA_CHECK(cudaMemcpyAsync((char*)output + (sent_blocks[i]*unit_size), 
                rtemp_buffer + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync((char*)send_buf + offset, 
                rtemp_buffer + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    free(rank_r_reps);
    free(rotate_array);
    CUDA_CHECK(cudaFree(stemp_buffer));
    CUDA_CHECK(cudaFree(rtemp_buffer));
    free(sent_blocks);
}

// NCCL P2P uniform modified radix-r Bruck all-to-all implementation
// Uses NCCL point-to-point operations with uniform modified radix-r Bruck algorithm
void ncclUniformModifiedRadixRBruckAllToAllGPU(void *output,
                                                const void *input,
                                                int total_elems,
                                                void *send_buf,
                                                void *recv_buf,
                                                int rank,
                                                int size,
                                                ncclComm_t comm,
                                                int r)
{
    assert(total_elems % size == 0 && "total_elems must be divisible by number of processes");
    assert(r >= 2 && "radix parameter r must be >= 2");

    int unit_size = total_elems / size;
    int w = ceil(log(size) / log(r)); // calculate the number of digits when using r-representation
    int nlpow = myPow(r, w-1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Initial rotation step - reorder data
    for (int i = 0; i < size; i++) {
        int index = (2*rank-i+size)%size;
        CUDA_CHECK(cudaMemcpyAsync((char*)output + (index*unit_size), (const char*)input + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
    }

    int* sent_blocks = (int*)malloc(nlpow * sizeof(int));
    int di = 0;
    int ci = 0;

    char* temp_buffer;
    CUDA_CHECK(cudaMalloc(&temp_buffer, nlpow * unit_size)); // temporary buffer
    char* sendbuf;
    CUDA_CHECK(cudaMalloc(&sendbuf, nlpow * unit_size)); // received data buffer

    int spoint = 1, distance = 1, next_distance = r;
    for (int x = 0; x < w; x++) {
        for (int z = 1; z < r; z++) {

            // get the sent data-blocks
            // copy blocks which need to be sent at this step
            spoint = z * distance;
            if (spoint > size - 1) {break;}
                di = 0; ci = 0;
            for (int i = spoint; i < size; i += next_distance) {
                for (int j = i; j < (i+distance); j++) {
                    if (j > size - 1 ) { break; }
                    int id = (j + rank) % size;
                    sent_blocks[di++] = id;
                    CUDA_CHECK(cudaMemcpyAsync(&temp_buffer[unit_size*ci++], &((char*)output)[id*unit_size], unit_size, cudaMemcpyDeviceToDevice, stream));
                }
            }

            // send and receive using NCCL P2P
            int recv_proc = (rank + spoint) % size; // receive data from rank - 2^step process
            int send_proc = (rank - spoint + size) % size; // send data from rank + 2^k process
            long long comm_size = di * unit_size;

            // Exchange data using NCCL P2P
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(temp_buffer, comm_size, ncclInt8, send_proc, comm, stream));
            NCCL_CHECK(ncclRecv(sendbuf, comm_size, ncclInt8, recv_proc, comm, stream));
            NCCL_CHECK(ncclGroupEnd());

            // replace with received data
            for (int i = 0; i < di; i++) {
                long long offset = sent_blocks[i] * unit_size;
                CUDA_CHECK(cudaMemcpyAsync((char*)output + offset, sendbuf + (i*unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        distance *= r;
        next_distance *= r;
    }

    CUDA_CHECK(cudaFree(temp_buffer));
    CUDA_CHECK(cudaFree(sendbuf));
    free(sent_blocks);
}