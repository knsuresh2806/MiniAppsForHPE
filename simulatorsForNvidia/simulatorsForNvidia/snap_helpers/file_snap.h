/*************************************************************************
 * File object for reading and writing plane and volume snapshots  
 *
 * Author:     Samuel Brown
 * Created:    Dec. 1, 2010
 * Change Log:
 *
 *************************************************************************/

#ifndef _FILE_SNAP
#define _FILE_SNAP

#include <cstdio>
#include <vector>
#include "cart_volume.h"

class file_snap
{

public:
    virtual ~file_snap() {}

    /// Write a timestep to later be retrieved by read_step()
    virtual void write_step(realtype* data) = 0;

    /// Read a step that was written by write_step().  Data is read back in
    /// in first-in-last-out order
    virtual void read_step(realtype* data) = 0;
    virtual void write_step(realtype* data1, realtype* data2, int data2Size) = 0;
    virtual void write_step(realtype* data1, realtype* data2) = 0;
    virtual void read_step(realtype* data1, realtype* data2, int data2Size) = 0;
    virtual void read_step(realtype* data1, realtype* data2) = 0;
    virtual void write_step(std::vector<realtype*>& data_in, std::vector<int>& size_in) = 0;
    virtual void read_step(std::vector<realtype*>& data_in, std::vector<int>& size_in) = 0;

    /// Report on the results of a run
    virtual void print_results(FILE* fd) {}
};

#endif
