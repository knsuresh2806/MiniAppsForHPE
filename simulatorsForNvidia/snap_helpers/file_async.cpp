
#include "file_async.h"
#include "emsl_error.h"
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <fcntl.h>

//const int aioflags = O_CREAT | O_RDWR | O_DIRECT;
//const int aioflagsp1 = O_CREAT | O_RDWR;
//const int aioflagsp2 = O_RDWR | O_DIRECT;

file_async::file_async(char* name, int flags) : name(name)
{
    fd = open(name, flags, S_IRWXU);
    if (fd < 0)
        EMSL_ERROR("open call for %s failed with %s", name, strerror(errno));

    memset((void*)aioreq, 0, aionbufs * sizeof(struct aiocb));
}

file_async::~file_async()
{
    if (-1 == close(fd))
        EMSL_ERROR("close() for %s failed with %s", name.c_str(), strerror(errno));
}

void
file_async::seek_beg(off_t offset, int bufp)
{
    aioreq[bufp].aio_fildes = fd;
    aioreq[bufp].aio_offset = offset;
}

void
file_async::write(void* buf, size_t nbytes, int bufp)
{
#ifndef DO_NOT_USE_AIO
    aioreq[bufp].aio_fildes = fd;
    aioreq[bufp].aio_buf = buf;
    aioreq[bufp].aio_nbytes = nbytes;
    if (-1 == aio_write(&aioreq[bufp]))
        EMSL_ERROR("aio_write with nbytes=%ld for %s failed with %s ", nbytes, name.c_str(), strerror(errno));
#else
    int nwrote;
    char* cbuf = (char*)buf;
    off_t offset = aioreq[bufp].aio_offset;
    nwrote = pwrite(fd, cbuf, nbytes, offset);
    if (nwrote == -1) {
        fprintf(stderr, "AIO_WRITE can't write %ld bytes code %d\n", nbytes, errno);
        EMSL_ERROR("aio_write with nbytes=%ld for %s failed with %s ", nbytes, name.c_str(), strerror(errno));
    }
#endif
}

void
file_async::read(void* buf, size_t nbytes, int bufp)
{
#ifndef DO_NOT_USE_AIO
    aioreq[bufp].aio_fildes = fd;
    aioreq[bufp].aio_buf = buf;
    aioreq[bufp].aio_nbytes = nbytes;
    if (-1 == aio_read(&aioreq[bufp]))
        EMSL_ERROR("aio_read call for %s failed with %s", name.c_str(), strerror(errno));
#else
    ssize_t no;
    off_t offset = aioreq[bufp].aio_offset;
    if (-1 == pread(fd, buf, nbytes, offset))
        EMSL_ERROR("aio_read call for %s failed with %s", name.c_str(), strerror(errno));
#endif
}

void
file_async::wait(int bufp)
{
#ifdef DO_NOT_USE_AIO
    return;
#else
    if (NULL == aioreq[bufp].aio_buf)
        return;

    // aio_error returns 0 indicating the operation completed successfully,
    // EINPROGRESS to say that the operation is still happening, or an
    // error code which is interpreted like errno.
    int ierr, icnt = 0;
    while ((ierr = aio_error(&aioreq[bufp])) != 0) {
        switch (ierr) {
            case EINPROGRESS:
                usleep(1000);             // Sleep nicely for 1/1000 of a second
                if (++icnt > 1000 * 1000) // Only do this for 1000 total seconds
                    EMSL_ERROR("file_async::wait exceeds 1000 seconds on %s", name.c_str());
                break;
            default:
                EMSL_ERROR("file_async::wait encountered an error writing to %s: %s", name.c_str(), strerror(ierr));
                break;
        }
    }

    size_t nbytes = aio_return(&aioreq[bufp]);
    if (nbytes != aioreq[bufp].aio_nbytes)
        EMSL_ERROR("I/O call returned %ld bytes for %s, should have returned %ld", nbytes, name.c_str(),
                   aioreq[bufp].aio_nbytes);

    memset((void*)&aioreq[bufp], 0, sizeof(struct aiocb));
#endif
}

void
file_async::mark_temporary()
{
    // Unlinking the file on disk will cause it to be deleted once the
    // process ends (normally or abnormally)
    unlink(name.c_str());
}
