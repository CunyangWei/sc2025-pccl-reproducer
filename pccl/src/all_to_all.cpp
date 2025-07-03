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

    // Perform all-to-all exchanges
    for (int step = 1; step < size; step++) {
        int partner = (rank + step) % size;
        
        // Calculate send and receive offsets
        int send_offset = partner * block_size;
        int recv_offset = partner * block_size;
        
        // Synchronize CUDA stream before MPI communication
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
        
        // Exchange data with partner
        MPI_Sendrecv(static_cast<char*>(send_buf) + send_offset, 
                     block_size, MPI_BYTE, partner, 0,
                     static_cast<char*>(recv_buf) + recv_offset, 
                     block_size, MPI_BYTE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
        
        // Copy received data to correct position in output
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + recv_offset, 
                                 static_cast<char*>(recv_buf) + recv_offset, 
                                 block_size, 
                                 cudaMemcpyDeviceToDevice, 
                                 stream));
    }
    
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
    for (int phase = 0; phase < size - 1; phase++) {
        int partner;
        if (phase % 2 == 0) {
            // Even phase: pair with next process
            if (rank % 2 == 0 && rank + 1 < size) {
                partner = rank + 1;
            } else if (rank % 2 == 1) {
                partner = rank - 1;
            } else {
                continue; // Odd rank at end, no partner
            }
        } else {
            // Odd phase: pair with previous process
            if (rank % 2 == 1 && rank + 1 < size) {
                partner = rank + 1;
            } else if (rank % 2 == 0 && rank > 0) {
                partner = rank - 1;
            } else {
                continue; // Even rank at start, no partner
            }
        }
        
        // Determine which blocks to exchange
        int send_block = (rank + partner + phase) % size;
        int recv_block = (partner + rank + phase) % size;
        
        int send_offset = send_block * block_size;
        int recv_offset = recv_block * block_size;
        
        // Synchronize CUDA stream before MPI communication
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
        
        // Exchange blocks with partner
        MPI_Sendrecv(static_cast<char*>(output) + send_offset, 
                     block_size, MPI_BYTE, partner, phase,
                     static_cast<char*>(recv_buf), 
                     block_size, MPI_BYTE, partner, phase,
                     comm, MPI_STATUS_IGNORE);
        
        // Copy received data to correct position
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + recv_offset, 
                                 recv_buf, 
                                 block_size, 
                                 cudaMemcpyDeviceToDevice, 
                                 stream));
    }
    
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

    // Initialize output with local data
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + rank * block_size, 
                             static_cast<const char*>(input) + rank * block_size, 
                             block_size, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Ring algorithm: P-1 steps, each process sends one block around the ring
    for (int step = 0; step < size - 1; step++) {
        int send_to = (rank + 1) % size;
        int recv_from = (rank - 1 + size) % size;
        
        // Calculate which block to send/receive in this step
        int send_block = (rank - step + size) % size;
        int recv_block = (recv_from - step + size) % size;
        
        int send_offset = send_block * block_size;
        int recv_offset = recv_block * block_size;
        
        // Synchronize CUDA stream before MPI communication
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
        
        // Send to next, receive from previous in ring
        MPI_Sendrecv(static_cast<char*>(send_buf) + send_offset, 
                     block_size, MPI_BYTE, send_to, step,
                     static_cast<char*>(recv_buf), 
                     block_size, MPI_BYTE, recv_from, step,
                     comm, MPI_STATUS_IGNORE);
        
        // Copy received data to output and update send buffer for next iteration
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + recv_offset, 
                                 recv_buf, 
                                 block_size, 
                                 cudaMemcpyDeviceToDevice, 
                                 stream));
        
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(send_buf) + recv_offset, 
                                 recv_buf, 
                                 block_size, 
                                 cudaMemcpyDeviceToDevice, 
                                 stream));
    }
    
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

    // Copy input to send buffer
    CUDA_CHECK(cudaMemcpyAsync(send_buf, 
                             input, 
                             total_elems, 
                             cudaMemcpyDeviceToDevice, 
                             stream));

    // Bruck algorithm: log(P) rotation and exchange steps
    int num_steps = static_cast<int>(std::ceil(std::log2(size)));
    
    for (int step = 0; step < num_steps; step++) {
        int distance = 1 << step; // 2^step
        int partner = rank ^ distance; // XOR for partner calculation
        
        if (partner < size) {
            // Calculate blocks to exchange in this step
            int send_count = 0;
            int recv_count = 0;
            
            // Determine which blocks to send/receive based on binary representation
            for (int i = 0; i < size; i++) {
                if ((i ^ rank) >= distance && (i ^ rank) < (distance << 1)) {
                    if (step == 0 || ((i ^ rank) & ((1 << step) - 1)) == 0) {
                        send_count++;
                    }
                }
                if ((i ^ partner) >= distance && (i ^ partner) < (distance << 1)) {
                    if (step == 0 || ((i ^ partner) & ((1 << step) - 1)) == 0) {
                        recv_count++;
                    }
                }
            }
            
            // For simplicity, exchange blocks in chunks
            int exchange_size = (size >> (step + 1)) * block_size;
            if (exchange_size == 0) exchange_size = block_size;
            
            int send_offset = ((rank ^ distance) % size) * block_size;
            int recv_offset = send_offset;
            
            // Synchronize CUDA stream before MPI communication
            CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
            
            // Exchange data with partner
            MPI_Sendrecv(static_cast<char*>(send_buf) + send_offset, 
                         exchange_size, MPI_BYTE, partner, step,
                         static_cast<char*>(recv_buf), 
                         exchange_size, MPI_BYTE, partner, step,
                         comm, MPI_STATUS_IGNORE);
            
            // Update send buffer with received data for next iteration
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(send_buf) + recv_offset, 
                                     recv_buf, 
                                     exchange_size, 
                                     cudaMemcpyDeviceToDevice, 
                                     stream));
        }
    }
    
    // Final copy to output buffer in correct order
    for (int i = 0; i < size; i++) {
        int src_offset = ((rank + i) % size) * block_size;
        int dst_offset = i * block_size;
        
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output) + dst_offset, 
                                 static_cast<char*>(send_buf) + src_offset, 
                                 block_size, 
                                 cudaMemcpyDeviceToDevice, 
                                 stream));
    }
    
    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}