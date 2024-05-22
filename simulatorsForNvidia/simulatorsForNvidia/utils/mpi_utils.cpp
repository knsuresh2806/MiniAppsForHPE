#include <cstdlib>
#include <cstring>

#include "mpi_utils.h"

// Rank inside the node (to avoid using more ranks than GPUs)
int
mpi_utils::get_mpi_node_rank(MPI_Comm comm)
{
    int npes, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    int lname;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &lname);
    char* allnames = (char*)malloc(npes * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    int count = 0;
    // Count how many ranks before this one have the same hostname
    for (int i = 0; i < rank; i++) {
        if (strcmp(hostname, allnames + i * MPI_MAX_PROCESSOR_NAME) == 0)
            count++;
    }
    free(allnames);
    return count;
}
