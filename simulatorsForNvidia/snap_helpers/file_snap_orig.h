/*************************************************************************
 * File object for reading and writing plane and volume snapshots  
 *
 * Author:     Samuel Brown
 * Created:    Dec. 1, 2010	  
 * Change Log:
 *
 *************************************************************************/

#ifndef _FILE_SNAP_ORIG
#define _FILE_SNAP_ORIG

#include <mpi.h>
#include "file_snap.h"
#include <vector>
#include <cstdio>
#include <string>

//Forward declarations
class file_async;

/// This is the original implementation of file_snap that uses no shared memory and allows
/// all ranks to manage their own memory independently.  This implementation is good to keep
/// around for the times when an OS configuration doesn't have enough space allocated
/// to shared memory and therefore file_snap_shmem isn't a good alternative
class file_snap_orig : public file_snap
{

public:
    file_snap_orig(MPI_Comm comm_, char* scratch, long nval, realtype mem_perc, int nt);

    virtual ~file_snap_orig();

    /// Write a timestep to later be retrieved by read_step()
    virtual void write_step(realtype* data);

    /// Read a step that was written by write_step().  Data is read back in
    /// in first-in-last-out order
    virtual void read_step(realtype* data);

    virtual void write_step(realtype* data1, realtype* data2, int data2Size);
    virtual void write_step(realtype* data1, realtype* data2);
    virtual void write_step(std::vector<realtype*>& data_in, std::vector<int>& size_in);
    virtual void read_step(realtype* data1, realtype* data2, int data2Size);
    virtual void read_step(realtype* data1, realtype* data2);
    virtual void read_step(std::vector<realtype*>& data_in, std::vector<int>& size_in);

    //general function to write and read snap shots

private:
    void init_memory_snaps(long nval, int nt, double mem_perc);

    void print_strategy(FILE* fd) const;

    enum Destination
    {
        DISK,
        MEMORY
    };

    struct StrategyType
    {
        Destination dest;
        size_t offset;
        realtype* buff;
    };

    MPI_Comm comm;
    file_async* fd;
    std::string fname;
    std::vector<StrategyType> strategy;
    realtype* tempBuff;
    std::vector<realtype*> memBuff;

    long nval, nvalPad;
    int time_index;
};

#endif
