/*************************************************************************
 * Linux sys_info definition
 *
 * Author:     Mike Townsley
 * Created:    July 2011
 * Change Log:
 *
 *************************************************************************/

#ifndef _SYS_INFO
#define _SYS_INFO

#include <string>

class sys_info
{

public:
    virtual double get_host_memory_mb() = 0;

    virtual double get_process_memory_mb() = 0;

    virtual double get_disk_avail_mb(std::string& path) = 0;

    virtual double get_shmem_avail_mb() { return get_host_memory_mb(); }
};

#endif
