
#ifndef _FILE_CONST
#define _FILE_CONST

/*
   //Macros and constants used in the file module.
   @author: Rahul S. Sampath
   */

#define _FILE_OBJ_READ 1
#define _FILE_OBJ_WRITE 2
#define _FILE_OBJ_APPEND 4
#define _FILE_OBJ_TRUNCATE 8

#define _ASCII 1
#define _RSF 2
#define _SU 4
#define _SEGY 8
#define _BIN 16
#define _SAC 32
#define _SACLST 64
#define _LST 128
#define _MTC 256
#define _GEO 14
#define _P190 512
#define _SEIS 1024

#define IS_A_NUMBER(c) (c >= '0' && c <= '9')

#define HOST_NPE_MAX 32

typedef unsigned long long u64;

#endif
