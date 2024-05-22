
/*************************************************************************
 *
 * Author:  Rahul S. Sampath / Samuel Brown
 *
 *************************************************************************/

#ifndef _STD_CONST
#define _STD_CONST

#include <string>

#ifdef EMSL_TARGET_DOUBLE
typedef double realtype;
#define MPI_REALTYPE MPI_DOUBLE
#define SQRT(x) (sqrt(x))
#define SIN(x) (sin(x))
#define COS(x) (cos(x))
#define TAN(x) (tan(x))
#define ASIN(x) (asin(x))
#define ACOS(x) (acos(x))
#define ATAN(x) (atan(x))
#define ATAN2(x, y) (atan2(x, y))
#else
typedef float realtype;
#define MPI_REALTYPE MPI_FLOAT
#define SQRT(x) (sqrtf(x))
#define SIN(x) (sinf(x))
#define COS(x) (cosf(x))
#define TAN(x) (tanf(x))
#define ASIN(x) (asinf(x))
#define ACOS(x) (acosf(x))
#define ATAN(x) (atanf(x))
#define ATAN2(x, y) (atan2f(x, y))
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define _STR_LEN 4096

#ifdef __GNUG__
#define _RESTRICT __restrict__
#else
#define _RESTRICT restrict
#endif

#define _RESTRICT_READ_FLOAT1_PTR realtype const* const _RESTRICT

#define _RESTRICT_WRITE_FLOAT1_PTR realtype* const _RESTRICT

#define _RESTRICT_READ_FLOAT2_PTR realtype const* const _RESTRICT* const _RESTRICT

#define _RESTRICT_WRITE_FLOAT2_PTR realtype* const _RESTRICT* const _RESTRICT

#define _RESTRICT_READ_FLOAT3_PTR realtype const* const _RESTRICT* const _RESTRICT* const _RESTRICT

#define _RESTRICT_WRITE_FLOAT3_PTR realtype* const _RESTRICT* const _RESTRICT* const _RESTRICT

#define _RESTRICT_READ_FLOAT4_PTR realtype const* const _RESTRICT* const _RESTRICT* const _RESTRICT* const _RESTRICT

#define _RESTRICT_WRITE_FLOAT4_PTR realtype* const _RESTRICT* const _RESTRICT* const _RESTRICT* const _RESTRICT

typedef realtype* FloatPtr;
typedef FloatPtr* Float2Ptr;
typedef Float2Ptr* Float3Ptr;

typedef int* IntPtr;
typedef IntPtr* Int2Ptr;
typedef Int2Ptr* Int3Ptr;

typedef std::string* StringPtr;
typedef StringPtr* String2Ptr;
typedef String2Ptr* String3Ptr;

#define _BIG_FLOAT 340282346638528859811704183484516925440.f
#define _BIG_INTEGER 2147483647

enum DIMENSION
{
    _2D,
    _3D
};

const char _DIM_STR_2D[2][2] = { "X", "Z" };

const char _DIM_STR_3D[3][2] = { "X", "Y", "Z" };

// Generally used for creation of classes with data or function native to the processor type
enum class ProcessorType
{
    _CPU,
    _GPU
};



#endif
