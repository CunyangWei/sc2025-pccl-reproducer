#include <cassert>
#include <cmath>

#include "all_to_allv.h"
#include "common.h"

// NCCL P2P spread-out all-to-allv implementation  
// Uses NCCL point-to-point send/recv operations with variable message sizes
// Each process can send different amounts of data to other processes
// - output: CUDA device pointer where the final result will be stored.
// - input: CUDA device pointer to the local data.
// - send_bytes: array of integers specifying number of bytes to send to each process.
// - recv_bytes: array of integers specifying number of bytes to receive from each process.
// - send_byte_displs: array of displacements (offsets) in bytes in input buffer for data to be sent.
// - recv_byte_displs: array of displacements (offsets) in bytes in output buffer for received data.
// - rank: rank of current process.
// - size: total number of processes.
// - comm: NCCL communicator.
void ncclSpreadOutAllToAllvGPU(void* output, 
                               const void* input, 
                               const int* send_bytes,
                               const int* recv_bytes,
                               const int* send_byte_displs,
                               const int* recv_byte_displs,
                               int rank,
                               int size,
                               ncclComm_t comm) {
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // // Validate input parameters
    // if (output == nullptr || input == nullptr) {
    //     throw std::runtime_error("NULL pointer passed to alltoallv");
    // }
    // if (send_bytes == nullptr || recv_bytes == nullptr || 
    //     send_byte_displs == nullptr || recv_byte_displs == nullptr) {
    //     throw std::runtime_error("NULL array pointer passed to alltoallv");
    // }
    // if (rank < 0 || rank >= size) {
    //     throw std::runtime_error("Invalid rank in alltoallv");
    // }
    
    // Initialize output buffer with local data first (self communication)
    if (send_bytes[rank] > 0) {
        // Validate pointers before copying
        // if (output == nullptr || input == nullptr) {
        //     throw std::runtime_error("NULL pointer passed to alltoallv");
        // }
        
        // Validate displacement bounds
        // if (send_byte_displs[rank] < 0 || recv_byte_displs[rank] < 0) {
        //     throw std::runtime_error("Negative displacement in alltoallv");
        // }
        
        // Perform self copy
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(output) + recv_byte_displs[rank], 
            static_cast<const char*>(input) + send_byte_displs[rank], 
            send_bytes[rank], 
            cudaMemcpyDeviceToDevice, 
            stream));
    }
    
    // Perform all-to-allv exchanges using NCCL P2P with spread-out pattern
    // Use the same communication pattern as ncclSpreadOutAllToAllGPU
    for (int step = 1; step < size; step++) {
        
        int send_partner = (rank + step) % size;
        int recv_partner = (rank - step + size) % size;
        
        // Use NCCL grouped operations for this step
        NCCL_CHECK(ncclGroupStart());

        // Send data to send_partner
        if (send_bytes[send_partner] > 0) {
            const char *send_ptr = static_cast<const char*>(input) + send_byte_displs[send_partner];
            NCCL_CHECK(ncclSend(send_ptr, send_bytes[send_partner], ncclInt8, send_partner, comm, stream));
        }

        // Receive data from recv_partner  
        if (recv_bytes[recv_partner] > 0) {
            char *recv_ptr = static_cast<char*>(output) + recv_byte_displs[recv_partner];
            NCCL_CHECK(ncclRecv(recv_ptr, recv_bytes[recv_partner], ncclInt8, recv_partner, comm, stream));
        }
        
        NCCL_CHECK(ncclGroupEnd());
    }
}

void ncclAllToAllvGPU(void* output, 
                    const void* input, 
                    const int* send_bytes,
                    const int* recv_bytes,
                    const int* send_byte_displs,
                    const int* recv_byte_displs,
                    int rank,
                    int size,
                    ncclComm_t comm) {

    auto stream = at::cuda::getCurrentCUDAStream();
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    // Validate input parameters
    // if (output == nullptr || input == nullptr) {
    //     throw std::runtime_error("NULL pointer passed to alltoallv");
    // }
    // if (send_bytes == nullptr || recv_bytes == nullptr || 
    //     send_byte_displs == nullptr || recv_byte_displs == nullptr) {
    //     throw std::runtime_error("NULL array pointer passed to alltoallv");
    // }
    // if (rank < 0 || rank >= size) {
    //     throw std::runtime_error("Invalid rank in alltoallv");
    // }

    // Use NCCL grouped operations for optimal performance
    NCCL_CHECK(ncclGroupStart());

    // Process all ranks in order
    for (int r = 0; r < size; r++) {
        // Send data to rank r if we have data to send
        if (send_bytes[r] > 0) {
            const char* send_ptr = static_cast<const char*>(input) + send_byte_displs[r];
            NCCL_CHECK(ncclSend(send_ptr, send_bytes[r], ncclInt8, r, comm, stream));
        }

        // Receive data from rank r if we expect to receive data
        if (recv_bytes[r] > 0) {
            char* recv_ptr = static_cast<char*>(output) + recv_byte_displs[r];
            NCCL_CHECK(ncclRecv(recv_ptr, recv_bytes[r], ncclInt8, r, comm, stream));
        }
    }

    NCCL_CHECK(ncclGroupEnd());

}


// Define default throttle parameters
#ifndef ALLTOALL_THROTTLE
#define ALLTOALL_THROTTLE 32
#endif

/**
 * Pairwise SendRecv algorithm implementation
 * Suitable for general intra-communicator scenarios
 * Uses pairwise sendrecv operations where each process exchanges data with others in order
 */
void AllToAllvGPU_pairwise_sendrecv(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm) 
{
    int i, j;
    MPI_Status status;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    
    // First copy self-to-self data (rank to rank)
    if (send_bytes[rank] > 0) {
        memcpy(recvbuf + recv_byte_displs[rank],
               sendbuf + send_byte_displs[rank],
               send_bytes[rank]);
    }

    // Pairwise exchange pattern
    // Each process exchanges data with all other processes in a fixed order
    for (i = 1; i < size; i++) {
        // Calculate communication partners
        int sendto = (rank + i) % size;
        int recvfrom = (rank - i + size) % size;
        
        void* sendaddr = NULL;
        void* recvaddr = NULL;
        int sendcount = 0;
        int recvcount = 0;
        
        // Set send parameters
        if (send_bytes[sendto] > 0) {
            sendaddr = sendbuf + send_byte_displs[sendto];
            sendcount = send_bytes[sendto];
        }

        // Set receive parameters
        if (recv_bytes[recvfrom] > 0) {
            recvaddr = recvbuf + recv_byte_displs[recvfrom];
            recvcount = recv_bytes[recvfrom];
        }

        // Perform sendrecv operation
        if (sendcount > 0 || recvcount > 0) {
            MPI_Sendrecv(sendaddr, sendcount, MPI_BYTE, sendto, 0,
                        recvaddr, recvcount, MPI_BYTE, recvfrom, 0,
                        comm, &status);
        }
    }
}

void AllToAllvGPU_pairwise_sendrecv_datatype(void* output, 
    const void* input, 
    const int* sendcounts,
    const int* recvcounts,
    const int* send_displs,
    const int* recv_displs,
    int dtype_size,
    int rank,
    int size,
    MPI_Comm comm) 
{
    int i, j;
    MPI_Status status;
    
    if (dtype_size == 4) {
        // Use float pointer to avoid manual byte offset calculation
        float* sendbuf = (float*)input;
        float* recvbuf = (float*)output;
        
        // First copy self-to-self data (rank to rank)
        if (sendcounts[rank] > 0) {
            memcpy(recvbuf + recv_displs[rank],
                   sendbuf + send_displs[rank],
                   sendcounts[rank] * sizeof(float));
        }

        // Pairwise exchange pattern
        for (i = 1; i < size; i++) {
            int sendto = (rank + i) % size;
            int recvfrom = (rank - i + size) % size;

            // Perform sendrecv operation
            if (sendcounts[sendto] > 0 || recvcounts[recvfrom] > 0) {
                MPI_Sendrecv(sendbuf + send_displs[sendto], sendcounts[sendto], MPI_FLOAT, sendto, 0,
                            recvbuf + recv_displs[recvfrom], recvcounts[recvfrom], MPI_FLOAT, recvfrom, 0,
                            comm, &status);
            }
        }
    } else if (dtype_size == 2) {
        // Use int16_t pointer to handle float16
        int16_t* sendbuf = (int16_t*)input;
        int16_t* recvbuf = (int16_t*)output;
        
        // First copy self-to-self data
        if (sendcounts[rank] > 0) {
            memcpy(recvbuf + recv_displs[rank],
                   sendbuf + send_displs[rank],
                   sendcounts[rank] * sizeof(int16_t));
        }

        // Pairwise exchange pattern, using BYTE transfer
        for (i = 1; i < size; i++) {
            int sendto = (rank + i) % size;
            int recvfrom = (rank - i + size) % size;
            
            if (sendcounts[sendto] > 0 || recvcounts[recvfrom] > 0) {
                MPI_Sendrecv(sendbuf + send_displs[sendto], sendcounts[sendto] * 2, MPI_BYTE, sendto, 0,
                            recvbuf + recv_displs[recvfrom], recvcounts[recvfrom] * 2, MPI_BYTE, recvfrom, 0,
                            comm, &status);
            }
        }
    } else {
        // dtype_size == 1 or other cases, use char pointer
        char* sendbuf = (char*)input;
        char* recvbuf = (char*)output;
        
        // First copy self-to-self data
        if (sendcounts[rank] > 0) {
            memcpy(recvbuf + recv_displs[rank],
                   sendbuf + send_displs[rank],
                   sendcounts[rank]);
        }
        
        // Pairwise exchange pattern
        for (i = 1; i < size; i++) {
            int sendto = (rank + i) % size;
            int recvfrom = (rank - i + size) % size;
            
            if (sendcounts[sendto] > 0 || recvcounts[recvfrom] > 0) {
                MPI_Sendrecv(sendbuf + send_displs[sendto], sendcounts[sendto], MPI_CHAR, sendto, 0,
                            recvbuf + recv_displs[recvfrom], recvcounts[recvfrom], MPI_CHAR, recvfrom, 0,
                            comm, &status);
            }
        }
    }
}

/**
 * Pairwise Exchange algorithm implementation
 * Optimized pairwise algorithm using ring communication pattern
 * Suitable for scenarios requiring load balancing
 */
void AllToAllvGPU_pairwise_exchange(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i;
    MPI_Status status;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    
    // Use pairwise exchange pattern
    // At each step, all processes exchange data with a specific partner
    for (i = 0; i < size; i++) {
        // Calculate source and destination for the current step
        int src = (rank - i + size) % size;
        int dst = (rank + i) % size;
        
        // Handle self-to-self case
        if (src == rank && dst == rank) {
            if (send_bytes[rank] > 0) {
                memcpy(recvbuf + recv_byte_displs[rank],
                       sendbuf + send_byte_displs[rank],
                       send_bytes[rank]);
            }
            continue;
        }
        
        void* sendaddr = NULL;
        void* recvaddr = NULL;
        int sendcount = 0;
        int recvcount = 0;
        
        // Set send parameters
        if (send_bytes[dst] > 0) {
            sendaddr = sendbuf + send_byte_displs[dst];
            sendcount = send_bytes[dst];
        }

        // Set receive parameters
        if (recv_bytes[src] > 0) {
            recvaddr = recvbuf + recv_byte_displs[src];
            recvcount = recv_bytes[src];
        }

        // Perform sendrecv
        if (sendcount > 0 || recvcount > 0) {
            MPI_Sendrecv(sendaddr, sendcount, MPI_BYTE, dst, i,
                        recvaddr, recvcount, MPI_BYTE, src, i,
                        comm, &status);
        }
    }
}

/**
 * Scattered algorithm implementation (Tony Ladd optimized version)
 * Uses throttling to avoid network congestion, processes communication in batches
 * Suitable for large-scale parallelism and large data transfers
 */
void AllToAllvGPU_pairwise_scattered(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i, ii, ss, dst;
    int bblock;
    int req_cnt;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    MPI_Request* reqarray;
    MPI_Status* starray;
    
    // Set throttle parameters
    bblock = ALLTOALL_THROTTLE;
    if (bblock <= 0 || bblock > size) {
        bblock = size;
    }
    
    // Allocate request and status arrays
    reqarray = (MPI_Request*)malloc(2 * bblock * sizeof(MPI_Request));
    starray = (MPI_Status*)malloc(2 * bblock * sizeof(MPI_Status));
    
    if (!reqarray || !starray) {
        // Memory allocation failed, fall back to simple algorithm
        if (reqarray) free(reqarray);
        if (starray) free(starray);
        AllToAllvGPU_pairwise_sendrecv(output, input, send_bytes, recv_bytes,
                                       send_byte_displs, recv_byte_displs,
                                       rank, size, comm);
        return;
    }
    
    // First handle self-to-self data
    if (send_bytes[rank] > 0) {
        memcpy(recvbuf + recv_byte_displs[rank],
               sendbuf + send_byte_displs[rank],
               send_bytes[rank]);
    }

    // Process communication in batches
    for (ii = 0; ii < size; ii += bblock) {
        req_cnt = 0;
        ss = (size - ii < bblock) ? (size - ii) : bblock;
        
        // Initiate ss non-blocking receives
        for (i = 0; i < ss; i++) {
            dst = (rank + i + ii) % size;

            // Skip self
            if (dst == rank) continue;
            
            if (recv_bytes[dst] > 0) {
                MPI_Irecv(recvbuf + recv_byte_displs[dst],
                         recv_bytes[dst],
                         MPI_BYTE,
                         dst,
                         0,
                         comm,
                         &reqarray[req_cnt]);
                req_cnt++;
            }
        }
        
        // Initiate ss non-blocking sends
        for (i = 0; i < ss; i++) {
            dst = (rank - i - ii + size) % size;

            // Skip self
            if (dst == rank) continue;
            
            if (send_bytes[dst] > 0) {
                MPI_Isend(sendbuf + send_byte_displs[dst],
                         send_bytes[dst],
                         MPI_BYTE,
                         dst,
                         0,
                         comm,
                         &reqarray[req_cnt]);
                req_cnt++;
            }
        }
        
        // Wait for all communications to complete
        if (req_cnt > 0) {
            MPI_Waitall(req_cnt, reqarray, starray);
            
            // Check error status
            for (i = 0; i < req_cnt; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    fprintf(stderr, "Rank %d: Communication error in batch %d\n", 
                            rank, ii/bblock);
                }
            }
        }
    }
    
    // Free resources
    free(reqarray);
    free(starray);
}


void AllToAllvGPU_openmpi_pairwise(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int step, sendto, recvfrom;
    MPI_Request req;
    MPI_Status status;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    void* psnd;
    void* prcv;
    
    /* Perform pairwise exchange step by step */
    for (step = 0; step < size; step++) {
        req = MPI_REQUEST_NULL;
        
        /* Determine send and receive targets for this step */
        sendto = (rank + step) % size;
        recvfrom = (rank + size - step) % size;
        
        /* Determine send and receive positions */
        psnd = sendbuf + send_byte_displs[sendto];
        prcv = recvbuf + recv_byte_displs[recvfrom];
        
        /* Initiate non-blocking receive first */
        if (recv_bytes[recvfrom] > 0) {
            MPI_Irecv(prcv, recv_bytes[recvfrom], MPI_BYTE, 
                     recvfrom, 0, comm, &req);
        }
        
        /* Send data */
        if (send_bytes[sendto] > 0) {
            MPI_Send(psnd, send_bytes[sendto], MPI_BYTE, 
                    sendto, 0, comm);
        }
        
        /* Wait for receive to complete */
        if (req != MPI_REQUEST_NULL) {
            MPI_Wait(&req, &status);
        }
    }
}

/**
 * Open MPI Basic Linear algorithm implementation
 * Issues all non-blocking communications at once, then waits for all to complete
 */
void AllToAllvGPU_openmpi_basic_linear(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i, nreqs;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    MPI_Request* reqs;
    MPI_Status* stats;
    
    /* First handle self-to-self data */
    if (send_bytes[rank] > 0) {
        memcpy(recvbuf + recv_byte_displs[rank],
               sendbuf + send_byte_displs[rank],
               send_bytes[rank]);
    }
    
    /* If there is only one process, return immediately */
    if (size == 1) {
        return;
    }
    
    /* Allocate request arrays */
    reqs = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
    stats = (MPI_Status*)malloc(2 * size * sizeof(MPI_Status));
    
    if (!reqs || !stats) {
        /* Memory allocation failed, fall back to pairwise algorithm */
        if (reqs) free(reqs);
        if (stats) free(stats);
        AllToAllvGPU_openmpi_pairwise(output, input, send_bytes, recv_bytes,
                                      send_byte_displs, recv_byte_displs,
                                      rank, size, comm);
        return;
    }
    
    nreqs = 0;
    
    /* Initiate all receive requests first */
    for (i = 0; i < size; ++i) {
        if (i == rank) {
            continue;
        }
        
        if (recv_bytes[i] > 0) {
            MPI_Irecv(recvbuf + recv_byte_displs[i],
                     recv_bytes[i], MPI_BYTE,
                     i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Then initiate all send requests */
    for (i = 0; i < size; ++i) {
        if (i == rank) {
            continue;
        }
        
        if (send_bytes[i] > 0) {
            MPI_Isend(sendbuf + send_byte_displs[i],
                     send_bytes[i], MPI_BYTE,
                     i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Wait for all requests to complete */
    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs, stats);
    }
    
    /* Free resources */
    free(reqs);
    free(stats);
}

/**
 * Open MPI Basic Inplace algorithm implementation
 * Optimized algorithm for MPI_IN_PLACE scenarios, uses a ring algorithm variant
 * Note: assumes input == output indicates an in-place operation
 */
void AllToAllvGPU_openmpi_basic_inplace(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i, left, right;
    char* buffer = (char*)output;  /* In-place operation, input and output use the same buffer */
    char* tmp_buffer;
    int max_size = 0;
    MPI_Request req;
    MPI_Status status;
    
    /* If not an in-place operation, call the standard algorithm */
    if (input != output) {
        AllToAllvGPU_openmpi_pairwise(output, input, send_bytes, recv_bytes,
                                      send_byte_displs, recv_byte_displs,
                                      rank, size, comm);
        return;
    }
    
    /* Find the maximum message size */
    for (i = 0; i < size; ++i) {
        if (i == rank) continue;
        if (recv_bytes[i] > max_size) {
            max_size = recv_bytes[i];
        }
    }
    
    /* Handle simple cases */
    if (size == 1 || max_size == 0) {
        return;
    }
    
    /* Allocate temporary buffer */
    tmp_buffer = (char*)malloc(max_size);
    if (!tmp_buffer) {
        /* Memory allocation failed, cannot execute in-place algorithm */
        return;
    }
    
    /* Use ring algorithm variant, only requires size/2 rounds of communication */
    for (i = 1; i <= (size >> 1); ++i) {
        right = (rank + i) % size;
        left = (rank + size - i) % size;
        
        req = MPI_REQUEST_NULL;
        
        /* Handle communication with right neighbor */
        if (recv_bytes[right] > 0) {
            /* Save data to be sent to right neighbor into temporary buffer */
            memcpy(tmp_buffer, buffer + recv_byte_displs[right], recv_bytes[right]);
            
            /* Receive data from right neighbor */
            MPI_Irecv(buffer + recv_byte_displs[right],
                     recv_bytes[right], MPI_BYTE,
                     right, 0, comm, &req);
        }
        
        /* Handle communication with left neighbor (if left and right differ) */
        if ((left != right) && (recv_bytes[left] > 0)) {
            /* Send data to left neighbor */
            MPI_Send(buffer + recv_byte_displs[left],
                    recv_bytes[left], MPI_BYTE,
                    left, 0, comm);
            
            /* Wait for receive from right neighbor to complete */
            if (req != MPI_REQUEST_NULL) {
                MPI_Wait(&req, &status);
            }
            
            /* Receive data from left neighbor */
            MPI_Irecv(buffer + recv_byte_displs[left],
                     recv_bytes[left], MPI_BYTE,
                     left, 0, comm, &req);
        }
        
        /* Send temporary buffer data to right neighbor */
        if (recv_bytes[right] > 0) {
            MPI_Send(tmp_buffer, recv_bytes[right], MPI_BYTE,
                    right, 0, comm);
        }
        
        /* Wait for all receives to complete */
        if (req != MPI_REQUEST_NULL) {
            MPI_Wait(&req, &status);
        }
    }
    
    /* Free temporary buffer */
    free(tmp_buffer);
}

/**
 * Open MPI Inter-communicator algorithm implementation
 * Simplified version for inter-group communication
 */
void AllToAllvGPU_openmpi_inter(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i, nreqs;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    MPI_Request* reqs;
    MPI_Status* stats;
    
    /* Allocate request arrays */
    reqs = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
    stats = (MPI_Status*)malloc(2 * size * sizeof(MPI_Status));
    
    if (!reqs || !stats) {
        if (reqs) free(reqs);
        if (stats) free(stats);
        return;
    }
    
    nreqs = 0;
    
    /* Initiate all receives first */
    for (i = 0; i < size; ++i) {
        if (recv_bytes[i] > 0) {
            MPI_Irecv(recvbuf + recv_byte_displs[i],
                     recv_bytes[i], MPI_BYTE,
                     i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Then initiate all sends */
    for (i = 0; i < size; ++i) {
        if (send_bytes[i] > 0) {
            MPI_Isend(sendbuf + send_byte_displs[i],
                     send_bytes[i], MPI_BYTE,
                     i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Wait for all communications to complete */
    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs, stats);
    }
    
    /* Free resources */
    free(reqs);
    free(stats);
}

/**
 * Open MPI optimized persistent request version
 * Uses persistent communication requests to reduce overhead
 */
void AllToAllvGPU_openmpi_persistent(void* output, 
    const void* input, 
    const int* send_bytes,
    const int* recv_bytes,
    const int* send_byte_displs,
    const int* recv_byte_displs,
    int rank,
    int size,
    MPI_Comm comm)
{
    int i, nreqs;
    char* sendbuf = (char*)input;
    char* recvbuf = (char*)output;
    MPI_Request* reqs;
    MPI_Status* stats;
    
    /* Handle self-to-self data */
    if (send_bytes[rank] > 0) {
        memcpy(recvbuf + recv_byte_displs[rank],
               sendbuf + send_byte_displs[rank],
               send_bytes[rank]);
    }
    
    if (size == 1) {
        return;
    }
    
    /* Allocate request arrays */
    reqs = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
    stats = (MPI_Status*)malloc(2 * size * sizeof(MPI_Status));
    
    if (!reqs || !stats) {
        if (reqs) free(reqs);
        if (stats) free(stats);
        AllToAllvGPU_openmpi_pairwise(output, input, send_bytes, recv_bytes,
                                      send_byte_displs, recv_byte_displs,
                                      rank, size, comm);
        return;
    }
    
    nreqs = 0;
    
    /* Create persistent receive requests */
    for (i = 0; i < size; ++i) {
        if (i == rank) continue;
        
        if (recv_bytes[i] > 0) {
            MPI_Recv_init(recvbuf + recv_byte_displs[i],
                         recv_bytes[i], MPI_BYTE,
                         i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Create persistent send requests */
    for (i = 0; i < size; ++i) {
        if (i == rank) continue;
        
        if (send_bytes[i] > 0) {
            MPI_Send_init(sendbuf + send_byte_displs[i],
                         send_bytes[i], MPI_BYTE,
                         i, 0, comm, &reqs[nreqs]);
            nreqs++;
        }
    }
    
    /* Start all persistent requests */
    MPI_Startall(nreqs, reqs);
    
    /* Wait for all requests to complete */
    MPI_Waitall(nreqs, reqs, stats);
    
    /* Free persistent requests */
    for (i = 0; i < nreqs; ++i) {
        MPI_Request_free(&reqs[i]);
    }
    
    /* Free resources */
    free(reqs);
    free(stats);
}