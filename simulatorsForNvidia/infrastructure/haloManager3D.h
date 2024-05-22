/*
   Halo exchange manager for 3-D domains. 
Author: Rahul S. Sampath
*/

#ifndef _HALO_MANAGER_3D
#define _HALO_MANAGER_3D

#include <mpi.h>
#include <vector>
#include "mpi_const.h"
#include "grid_const.h"
#include "std_const.h"
#include <stdio.h>

//Forward declarations
template <class T>
class cart_volume;
class decompositionManager3D;

class haloManager3D
{
public:
    // We do not want copy operations because of the duplicated comm
    haloManager3D& operator=(const haloManager3D& other) = delete;

    //Destructor
    virtual ~haloManager3D() {}

    virtual void start_update() = 0;

    virtual void finish_update() = 0;

    virtual void print(FILE* fd) = 0;

protected:
    MPI_Comm comm;
};

#endif
