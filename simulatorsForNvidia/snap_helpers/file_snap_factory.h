
#ifndef _FILE_SNAP_FACTORY
#define _FILE_SNAP_FACTORY

#include <mpi.h>
#include "file_const.h"
#include "std_const.h"

//Forward declarations
class file_snap;

class file_snap_factory
{

public:
    /// Creates a new file_snap instance based on the input parameters
    static file_snap* create(MPI_Comm comm, char* scratch, u64 nval, realtype mem_perc, int nt, int snap_type);

private:
    file_snap_factory() {}
};

#endif
