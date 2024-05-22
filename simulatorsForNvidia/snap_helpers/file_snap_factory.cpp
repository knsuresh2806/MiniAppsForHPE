/*************************************************************************
 *
 * Author:     Mike Townsley
 * Change Log:
 *
 *************************************************************************/

#include <mpi.h>
#include "file_snap_factory.h"
#include "file_snap_orig.h"
#include "file_snap.h"
#include "file_const.h"
#include "emsl_error.h"
#include <sys/stat.h>
#include <cstdio>
#include <cerrno>

file_snap*
file_snap_factory::create(MPI_Comm comm, char* scratch, u64 nval, realtype mem_perc, int nt, int snap_type)
{
    int mype, npe;
    int scratch_len = 0;
    char test_char = 'a';

    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npe);

    // In the case where the scratch path doesn't exist, simply try to write
    // to the /scr device directly.  This is to support the eventual move away
    // from named /scr folders
#ifdef _INTEL
    struct stat info;
    if (-1 == stat(scratch, &info)) {
        // Attempt to change scratch to /scr if scratch supplied is unwriteable
        // This change must be passed back up the call trace so that the
        // job_status object is also looking at the correct directory
        while (test_char != '\0') {
            test_char = scratch[scratch_len];
            if (test_char != '\0')
                scratch_len++;
        }
        if (scratch_len > 3) {
            fprintf(stderr, "snaps: Can't find directory %s,", scratch);
            scratch[0] = '/';
            scratch[1] = 's';
            scratch[2] = 'c';
            scratch[3] = 'r';
            scratch[4] = '\0';
            fprintf(stderr, "attempting to write snaps to %s\n", scratch);
        }
        if (-1 == stat(scratch, &info))
            EMSL_ERROR("Unable to determine file system info for %s: %s", scratch, strerror(errno));
    }
#endif

    if (!mype)
        fprintf(stderr, "snaps: Using standard (original) snapshot logic\n");
    return (new file_snap_orig(comm, scratch, nval, mem_perc, nt));

    EMSL_ERROR("snap_type value of %d is not recognized", snap_type);
}
