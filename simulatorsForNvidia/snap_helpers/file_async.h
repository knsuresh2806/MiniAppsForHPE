
#ifndef _FILE_ASYNC
#define _FILE_ASYNC

#include <aio.h>
#include <sys/types.h>
#include <string>

/*
#define _XOPEN_SOURCE 600
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
*/

#define aionbufs 8

class file_async
{

public:
    file_async(char* name, int flags);

    ~file_async();

    void seek_beg(off_t offset, int bufp = 0);

    void write(void* buf, size_t nbytes, int bufp = 0);

    void read(void* buf, size_t nbytes, int bufp = 0);

    void wait(int bufp = 0);

    void mark_temporary();

private:
    std::string name;
    struct aiocb aioreq[aionbufs];
    int fd;
};

#endif
