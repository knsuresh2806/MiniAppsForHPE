
#ifndef _EMSL_ERROR
#define _EMSL_ERROR

#include <stdexcept>
#include <cstdio>
#include <cstdlib>

// Simple macro for doing an sprintf into a std::runtime_error exception.
// Exceptions can be used to return errors that should be reported and
// the application shutdown
#define EMSL_ERROR(...)                                                                                                \
    {                                                                                                                  \
        char a[4096], b[4096];                                                                                         \
        sprintf(a, __VA_ARGS__);                                                                                       \
        sprintf(b, "%s at line %d of %s.\n", a, __LINE__, __FILE__);                                                   \
        fprintf(stderr, "%s", b);                                                                                      \
        throw std::runtime_error(b);                                                                                   \
    }

#define EMSL_VERIFY(expr)                                                                                              \
    {                                                                                                                  \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "FAILED VERIFY at line %d of %s.\n", __LINE__, __FILE__);                                  \
            fflush(stderr);                                                                                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    }

#endif
