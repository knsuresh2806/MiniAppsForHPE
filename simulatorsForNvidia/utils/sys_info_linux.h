/*************************************************************************
 * Linux sys_info_linux implementation
 *
 * Author:     Mike Townsley
 * Created:    July 2011
 * Change Log:
 *
 *************************************************************************/

#ifndef _SYS_INFO_LINUX
#define _SYS_INFO_LINUX

#include "sys_info.h"
#include <sys/statvfs.h>
#include <linux/major.h>
#include <libgen.h>
#include "emsl_error.h"
#include <cstdio>
#include <cerrno>
#include <cstring>

#define MEM_GRP_FLG__ALLOCATED_IN_SYSTEM_MEMORY 1
#define MEM_GRP_FLG__FOUND_PSS_ALLOCATION 2

// This is the linux implementation of the sys_info interface, which
// answers simple questions about memory and disk availability
class sys_info_linux : public sys_info
{

public:
    virtual double get_host_memory_mb()
    {
        double hostTotalMemoryKB;
        FILE* memInfo = fopen("/proc/meminfo", "r");
        fscanf(memInfo, "MemTotal: %lf kB", &hostTotalMemoryKB);
        fclose(memInfo);

        return hostTotalMemoryKB / (double)1024;
    }

    virtual double get_process_memory_mb()
    {
        FILE* smapsFile;

        if (!(smapsFile = fopen("/proc/self/smaps", "r")))
            EMSL_ERROR("Failed to open /proc/self/smaps!\n");

        double vmmSize = 0.0, rssSize = 0.0, pssSize = 0.0;
        double vmmTotal = 0.0, rssTotal = 0.0, pssTotal = 0.0;
        unsigned char memoryGroupFlags = 0;
        while (!feof(smapsFile)) {
            char smapsLine[1024];
            fscanf(smapsFile, "%[^\n]\n", smapsLine);

            unsigned long startAddress, endAddress, addressOffset;
            char memoryPermissions[4];
            int deviceMajorNumber, deviceMinorNumber;
            unsigned long inode;

            if (sscanf(smapsLine, "%lx-%lx %4c %lx %x:%x %ld", &startAddress, &endAddress, memoryPermissions,
                       &addressOffset, &deviceMajorNumber, &deviceMinorNumber, &inode) == 7) {
                if ((memoryGroupFlags & MEM_GRP_FLG__ALLOCATED_IN_SYSTEM_MEMORY) &&
                    (!(memoryGroupFlags & MEM_GRP_FLG__FOUND_PSS_ALLOCATION) || pssSize > 0.0)) {
                    vmmTotal += vmmSize;
                    rssTotal += rssSize;
                    pssTotal += pssSize;
                }

                if (UNNAMED_MAJOR != deviceMajorNumber && MEM_MAJOR != deviceMajorNumber) {
                    memoryGroupFlags = 0;
                } else {
                    memoryGroupFlags = MEM_GRP_FLG__ALLOCATED_IN_SYSTEM_MEMORY;
                }
                vmmSize = rssSize = pssSize = 0;
            } else if (sscanf(smapsLine, "Size: %lf kB", &vmmSize) == 1) {
            } else if (sscanf(smapsLine, "Rss: %lf kB", &rssSize) == 1) {
            } else if (sscanf(smapsLine, "Pss: %lf kB", &pssSize) == 1) {
                memoryGroupFlags |= MEM_GRP_FLG__FOUND_PSS_ALLOCATION;
            }
        }

        fclose(smapsFile);

        // All values are in Kilobytes (KB) so divide by 1024 to get Megabytes (MB)
        return vmmTotal / (double)1024;
    }

    virtual double get_disk_avail_mb(std::string& path)
    {
        struct statvfs info;
        std::string copy(path);
        char* dname = dirname((char*)copy.c_str());
        if (-1 == statvfs(dname, &info))
            EMSL_ERROR("Unable to determine file system info for %s: %s", path.c_str(), strerror(errno));

        return info.f_bsize * info.f_bavail / (double)(1024 * 1024);
    }

    virtual double get_shmem_avail_mb()
    {
        struct statvfs info;
        if (-1 == statvfs("/dev/shm", &info))
            EMSL_ERROR("Unable to determine file system info for /dev/shm: %s", strerror(errno));

        return info.f_bsize * info.f_bavail / (double)(1024 * 1024);
    }
};

#endif
