#ifndef _EMSL_MPI_UTILS_
#define _EMSL_MPI_UTILS_

#include <mpi.h>

namespace mpi_utils {

// Rank inside the node
int get_mpi_node_rank(MPI_Comm comm);

}

#endif
