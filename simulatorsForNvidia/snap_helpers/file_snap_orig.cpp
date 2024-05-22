/*************************************************************************
 * File object for reading and writing plane and volume snapshots  
 *
 * Author:     Samuel Brown
 * Change Log:
 *
 *************************************************************************/

#include <mpi.h>
#include "file_snap_orig.h"
#include "file_async.h"
#include "Array.h"
#include "timer.h"
#include "sys_info_linux.h"
#include "emsl_error.h"
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include <new>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdlib.h>

file_snap_orig::file_snap_orig(MPI_Comm comm_, char* scratch, long nval, realtype mem_perc, int nt)
    : fd(NULL), time_index(0), strategy(nt), nval(nval), tempBuff(NULL)
{
    comm = comm_;

    fname.resize(strlen(scratch) + 40);
    sprintf(&fname[0], "%s/snap_scratch.%d", scratch, getpid());

    if (mem_perc > 0) {
        init_memory_snaps(nval, nt, mem_perc);
    }
}

file_snap_orig::~file_snap_orig()
{
    remove(fname.c_str());

    if (fd)
        delete fd;
    if (tempBuff)
        free(tempBuff);
    for (int i = 0; i < memBuff.size(); ++i) {
        delete[](memBuff[i]);
    } //end i
}

void
file_snap_orig::write_step(realtype* data)
{
    EMSL_VERIFY(time_index < strategy.size());

    if (MEMORY == strategy[time_index].dest) {
        memcpy(strategy[time_index].buff, data, nval * sizeof(realtype));
    }

    // Strategy is to wait on the last write first before
    // copying this next buffer to our tempBuffer and issuing
    // another async write
    if (DISK == strategy[time_index].dest) {
        fd->wait();
        EMSL_VERIFY(tempBuff != NULL);
        EMSL_VERIFY(data != NULL);
        memcpy(tempBuff, data, nval * sizeof(realtype));
        fd->seek_beg(strategy[time_index].offset);
        fd->write((void*)tempBuff, nvalPad * sizeof(realtype));
    }

    ++time_index;
}

void
file_snap_orig::read_step(realtype* data)
{
    --time_index;

    EMSL_VERIFY(time_index >= 0);

    if (MEMORY == strategy[time_index].dest) {
        memcpy(data, strategy[time_index].buff, nval * sizeof(realtype));
    }

    // Strategy HERE is to wait on the last read OR write, then
    // return the contents of the temp buffer (which if this is
    // the first read, the contents are already the last write).
    // Then issue the next async read so we're ahead of the
    // game.  If there are no more DISK timesteps in the strategy
    // then we should never be called again for read.
    if (DISK == strategy[time_index].dest) {
        fd->wait();
        memcpy(data, tempBuff, nval * sizeof(realtype));
        for (int i = time_index - 1; i >= 0; i--) {
            if (DISK == strategy[i].dest) {
                fd->seek_beg(strategy[i].offset);
                fd->read((void*)tempBuff, nvalPad * sizeof(realtype));
                break;
            }
        }
    }
}

void
file_snap_orig::write_step(realtype* data1, realtype* data2, int data2Size)
{
    EMSL_VERIFY(time_index < strategy.size());

    EMSL_VERIFY(MEMORY == strategy[time_index].dest);

    memcpy(&(strategy[time_index].buff[0]), data1, ((nval - data2Size) * (sizeof(realtype))));
    memcpy(&(strategy[time_index].buff[nval - data2Size]), data2, (data2Size * (sizeof(realtype))));

    ++time_index;
}

void
file_snap_orig::write_step(realtype* data1, realtype* data2)
{
    EMSL_VERIFY(time_index < strategy.size());

    EMSL_VERIFY(MEMORY == strategy[time_index].dest);

    memcpy(&(strategy[time_index].buff[0]), data1, (nval / 2) * sizeof(realtype));
    memcpy(&(strategy[time_index].buff[nval / 2]), data2, (nval / 2) * sizeof(realtype));

    ++time_index;
}

void
file_snap_orig::write_step(std::vector<realtype*>& data_in, std::vector<int>& size_in)
{
    EMSL_VERIFY(time_index < strategy.size());

    if (nval > 0) {
        EMSL_VERIFY(MEMORY == strategy[time_index].dest);
    }

    for (size_t data_number = 0, buff_index = 0; data_number < data_in.size(); ++data_number) {
        realtype* domain = data_in[data_number];
        int size = size_in[data_number];
        memcpy(&(strategy[time_index].buff[buff_index]), domain, (size * sizeof(realtype)));
        buff_index += size;
    }

    ++time_index;
}

void
file_snap_orig::read_step(realtype* data1, realtype* data2, int data2Size)
{
    --time_index;

    EMSL_VERIFY(time_index >= 0);

    EMSL_VERIFY(MEMORY == strategy[time_index].dest);

    memcpy(data1, &(strategy[time_index].buff[0]), ((nval - data2Size) * (sizeof(realtype))));
    memcpy(data2, &(strategy[time_index].buff[nval - data2Size]), (data2Size * (sizeof(realtype))));
}

void
file_snap_orig::read_step(realtype* data1, realtype* data2)
{
    --time_index;

    EMSL_VERIFY(time_index >= 0);

    EMSL_VERIFY(MEMORY == strategy[time_index].dest);

    memcpy(data1, &(strategy[time_index].buff[0]), (nval / 2) * sizeof(realtype));
    memcpy(data2, &(strategy[time_index].buff[nval / 2]), (nval / 2) * sizeof(realtype));
}

void
file_snap_orig::read_step(std::vector<realtype*>& data_in, std::vector<int>& size_in)
{
    --time_index;
    EMSL_VERIFY(time_index >= 0);

    if (nval > 0) {
        EMSL_VERIFY(MEMORY == strategy[time_index].dest);
    }

    for (size_t data_number = 0, buff_index = 0; data_number < data_in.size(); ++data_number) {
        realtype* domain = data_in[data_number];
        int size = size_in[data_number];
        memcpy(domain, &(strategy[time_index].buff[buff_index]), (size * sizeof(realtype)));
        buff_index += size;
    }
}

void
file_snap_orig::print_strategy(FILE* fd) const
{
    std::vector<char> strat;
    int diskCount = 0;
    double ratio = 0;
    int mype;

    MPI_Comm_rank(comm, &mype);

    for (int i = 0; i < strategy.size(); i++) {
        if (MEMORY == strategy[i].dest) {
            strat.push_back('.');
        } else if (DISK == strategy[i].dest) {
            strat.push_back('#');
            ++diskCount;
        } else {
            strat.push_back('?');
        }
    } //end for i

    strat.push_back('\0');
    fprintf(fd, "%03d: disk=%3d strat='%s'\n", mype, diskCount, &strat[0]);
}

void
file_snap_orig::init_memory_snaps(long nval, int nt, double mem_perc)
{
    sys_info_linux sinfo;

    int npes, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    //Total available memory on each node
    double hostTotalMB = sinfo.get_host_memory_mb();

    //Amount of memory already allocated on this rank
    double allocatedMB = sinfo.get_process_memory_mb();

    //Size of a single snapshot for each rank
    double snapSizeMB = ((double)nval) * ((double)(sizeof(realtype))) / (1024.0 * 1024.0);

    //Get the host name for this rank
    char host_name[HOST_NAME_MAX];
    std::memset(host_name, '\0', HOST_NAME_MAX);
    gethostname(host_name, HOST_NAME_MAX);

    std::pair<char[HOST_NAME_MAX], Array<double, 2>> myData;
    std::vector<std::pair<char[HOST_NAME_MAX], Array<double, 2>>> allData(npes);

    strcpy(myData.first, host_name);
    myData.second[0] = allocatedMB;
    myData.second[1] = snapSizeMB;

    //Gather the data for each rank
    MPI_Allgather(&myData, sizeof(std::pair<char[HOST_NAME_MAX], Array<double, 2>>), MPI_BYTE, &(allData[0]),
                  sizeof(std::pair<char[HOST_NAME_MAX], Array<double, 2>>), MPI_BYTE, comm);

    double hostAllocatedMB = 0.0;
    double hostSnapSizeMB = 0.0;
    int hostRank = 0; //Relative rank on my node
    for (int i = 0; i < npes; ++i) {
        if (!strcmp(allData[i].first, host_name)) {
            hostAllocatedMB += allData[i].second[0];
            hostSnapSizeMB += allData[i].second[1];
            if (i < rank) {
                ++hostRank;
            }
        }
    } //end i

    allData.clear();

    char hostName[HOST_NAME_MAX + 1];
    int err = gethostname(hostName, (HOST_NAME_MAX + 1));
    EMSL_VERIFY(err == 0);

    // Utilize mem_perc% of this host's available memory for snaps ensuring each rank
    // on the local host gets the same number of snaps then sanity check that it
    // is >=0 and <= the number of timesteps.
    const double hostAvailableMemoryMB = (mem_perc * hostTotalMB) - hostAllocatedMB;

    int memSnaps = nt;
    if (hostSnapSizeMB > 0.0) {
        memSnaps = (int)(hostAvailableMemoryMB / hostSnapSizeMB);
    }
    memSnaps = std::max(std::min(nt, memSnaps), 0);

    const double hostScratchAvailMB = 0.0;

    const double hostScratchRequiredMB = ((double)(nt - memSnaps)) * hostSnapSizeMB;

    if ((memSnaps < nt) && (hostSnapSizeMB > 0.0)) {
        EMSL_ERROR("Host %s requested %lf MB of local scratch in %s but only %lf MB is available", hostName,
                   hostScratchRequiredMB, fname.c_str(), hostScratchAvailMB);
    }

    // If this rank has interior (nval > 0) then setup the snapshot strategy for writing
    // either to disk or to memory
    if (nval > 0) {
        const size_t blockSize = sysconf(_SC_PAGESIZE);
        nvalPad = ((nval + blockSize - 1) / blockSize) * blockSize;
        int diskSnaps = nt - memSnaps;
        // this->nav is one field size and is assigned in constructor
        // passed nval can be the total size of nultiple fields
        //        this->nval = nval;

        // If we can't fit everything in memory, then create the file_async instance
        if (memSnaps < nt) {
            EMSL_VERIFY(false); //Should not come here
            int err = posix_memalign((void**)&tempBuff, blockSize, nvalPad * sizeof(realtype));
            EMSL_VERIFY(err == 0);
            remove(fname.c_str());
            fd = new file_async((char*)(fname.c_str()), O_RDWR | O_CREAT | O_DIRECT);
            fd->mark_temporary();
        }

        // If we can fit something in memory, allocate buffers to store them
        if (memSnaps > 0) {
            EMSL_VERIFY(memBuff.empty());
            memBuff.resize(memSnaps);
            for (int i = 0; i < memSnaps; ++i) {
                memBuff[i] = new (std::nothrow) realtype[nval];
                EMSL_VERIFY(memBuff[i] != NULL);
            } //end i
        }

        // Vary the storage strategy to space out the I/O across the ranks on this machine
        // by using hostRank to offset the strategies to every rank doesn't read and write
        // at the exact same time
        double ratio = (double)memSnaps / (double)nt;
        int memCounter = 0, diskCounter = 0;
        for (int idx = 0; idx < nt; ++idx) {
            int i = (idx + hostRank) % nt;
            if (floor(ratio * (idx - 1)) == floor(ratio * idx)) {
                EMSL_VERIFY(false); //Should not come here
                strategy[i].dest = DISK;
                strategy[i].buff = NULL;
                strategy[i].offset = diskCounter * nvalPad * sizeof(realtype);
                ++diskCounter;
            } else {
                strategy[i].dest = MEMORY;
                strategy[i].buff = memBuff[memCounter];
                strategy[i].offset = 0;
                ++memCounter;
            }
        } //end for idx
    }     //end if has interior

    // Elect the root rank on each node to contribute to the output and describe the memory usage
    if (hostRank == 0) {
        if (hostSnapSizeMB > 0) {
            std::string diskReport(1024, '\0');
            fprintf(stderr, "snaps: %s ranks are keeping %d of %d snaps in memory%s\n", hostName, memSnaps, nt,
                    diskReport.c_str());
        } else {
            fprintf(stderr, "snaps: %s has no snaps due to having no ranks in the interior\n", hostName);
        }
    } //end if root hostRank
}
