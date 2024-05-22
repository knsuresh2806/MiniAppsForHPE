
/**
  @author Rahul S. Sampath
  */

#ifndef _GRID_CONST
#define _GRID_CONST

//Regions (For Data-Block)

enum REG_ID_YZ
{
    _REG_ID_CEN_Y_CEN_Z,
    _REG_ID_OWN_NEG_Y_OWN_POS_Z,
    _REG_ID_CEN_Y_OWN_POS_Z,
    _REG_ID_OWN_POS_Y_OWN_POS_Z,
    _REG_ID_OWN_POS_Y_CEN_Z,
    _REG_ID_OWN_POS_Y_OWN_NEG_Z,
    _REG_ID_CEN_Y_OWN_NEG_Z,
    _REG_ID_OWN_NEG_Y_OWN_NEG_Z,
    _REG_ID_OWN_NEG_Y_CEN_Z,
    _REG_ID_OWN_NEG_Y_HALO_NEG_Z,
    _REG_ID_CEN_Y_HALO_NEG_Z,
    _REG_ID_OWN_POS_Y_HALO_NEG_Z,
    _REG_ID_HALO_NEG_Y_OWN_POS_Z,
    _REG_ID_HALO_NEG_Y_CEN_Z,
    _REG_ID_HALO_NEG_Y_OWN_NEG_Z,
    _REG_ID_OWN_POS_Y_HALO_POS_Z,
    _REG_ID_CEN_Y_HALO_POS_Z,
    _REG_ID_OWN_NEG_Y_HALO_POS_Z,
    _REG_ID_HALO_POS_Y_OWN_NEG_Z,
    _REG_ID_HALO_POS_Y_CEN_Z,
    _REG_ID_HALO_POS_Y_OWN_POS_Z,
    _REG_ID_HALO_POS_Y_HALO_POS_Z,
    _REG_ID_HALO_POS_Y_HALO_NEG_Z,
    _REG_ID_HALO_NEG_Y_HALO_NEG_Z,
    _REG_ID_HALO_NEG_Y_HALO_POS_Z,
    _REG_ID_EXTRA_NEG_Y_EXTRA_POS_Z,
    _REG_ID_HALO_NEG_Y_EXTRA_POS_Z,
    _REG_ID_OWN_NEG_Y_EXTRA_POS_Z,
    _REG_ID_CEN_Y_EXTRA_POS_Z,
    _REG_ID_OWN_POS_Y_EXTRA_POS_Z,
    _REG_ID_HALO_POS_Y_EXTRA_POS_Z,
    _REG_ID_EXTRA_POS_Y_EXTRA_POS_Z,
    _REG_ID_EXTRA_POS_Y_HALO_POS_Z,
    _REG_ID_EXTRA_POS_Y_OWN_POS_Z,
    _REG_ID_EXTRA_POS_Y_CEN_Z,
    _REG_ID_EXTRA_POS_Y_OWN_NEG_Z,
    _REG_ID_EXTRA_POS_Y_HALO_NEG_Z,
    _REG_ID_EXTRA_POS_Y_EXTRA_NEG_Z,
    _REG_ID_HALO_POS_Y_EXTRA_NEG_Z,
    _REG_ID_OWN_POS_Y_EXTRA_NEG_Z,
    _REG_ID_CEN_Y_EXTRA_NEG_Z,
    _REG_ID_OWN_NEG_Y_EXTRA_NEG_Z,
    _REG_ID_HALO_NEG_Y_EXTRA_NEG_Z,
    _REG_ID_EXTRA_NEG_Y_EXTRA_NEG_Z,
    _REG_ID_EXTRA_NEG_Y_HALO_NEG_Z,
    _REG_ID_EXTRA_NEG_Y_OWN_NEG_Z,
    _REG_ID_EXTRA_NEG_Y_CEN_Z,
    _REG_ID_EXTRA_NEG_Y_OWN_POS_Z,
    _REG_ID_EXTRA_NEG_Y_HALO_POS_Z,
    _REG_ID_YZ_TOTAL
};

//Split regions in Y or Z
enum SPLIT_REG_ID
{
    _SPLIT_REG_ID_EXTRA_NEG,
    _SPLIT_REG_ID_HALO_NEG,
    _SPLIT_REG_ID_OWN_NEG,
    _SPLIT_REG_ID_CEN,
    _SPLIT_REG_ID_OWN_POS,
    _SPLIT_REG_ID_HALO_POS,
    _SPLIT_REG_ID_EXTRA_POS,
    _SPLIT_REG_ID_TOTAL
};

const int RegYZtoSplitRegY[_REG_ID_YZ_TOTAL] = {
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_CEN_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_OWN_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_CEN_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_OWN_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_CEN_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_HALO_NEG_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_CEN_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_HALO_POS_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_HALO_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_CEN_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  //_REG_ID_HALO_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_HALO_POS_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_CEN_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_CEN_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_EXTRA_NEG  // _REG_ID_EXTRA_NEG_Y_HALO_POS_Z
};

const int RegYZtoSplitRegZ[_REG_ID_YZ_TOTAL] = {
    _SPLIT_REG_ID_CEN,       // _REG_ID_CEN_Y_CEN_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_CEN_Y_OWN_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_OWN_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_OWN_POS_Y_CEN_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_CEN_Y_OWN_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_OWN_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_OWN_NEG_Y_CEN_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_OWN_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_CEN_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_OWN_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_HALO_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_HALO_NEG_Y_CEN_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_HALO_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_OWN_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_CEN_Y_HALO_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_OWN_NEG_Y_HALO_POS_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_HALO_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_HALO_POS_Y_CEN_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_HALO_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_HALO_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_HALO_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_HALO_POS,  //_REG_ID_HALO_NEG_Y_HALO_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_HALO_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_OWN_NEG_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_CEN_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_OWN_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_HALO_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_EXTRA_POS, // _REG_ID_EXTRA_POS_Y_EXTRA_POS_Z
    _SPLIT_REG_ID_HALO_POS,  // _REG_ID_EXTRA_POS_Y_HALO_POS_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_EXTRA_POS_Y_OWN_POS_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_EXTRA_POS_Y_CEN_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_EXTRA_POS_Y_OWN_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_EXTRA_POS_Y_HALO_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_HALO_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_OWN_POS_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_CEN_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_OWN_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_HALO_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_EXTRA_NEG, // _REG_ID_EXTRA_NEG_Y_EXTRA_NEG_Z
    _SPLIT_REG_ID_HALO_NEG,  // _REG_ID_EXTRA_NEG_Y_HALO_NEG_Z
    _SPLIT_REG_ID_OWN_NEG,   // _REG_ID_EXTRA_NEG_Y_OWN_NEG_Z
    _SPLIT_REG_ID_CEN,       // _REG_ID_EXTRA_NEG_Y_CEN_Z
    _SPLIT_REG_ID_OWN_POS,   // _REG_ID_EXTRA_NEG_Y_OWN_POS_Z
    _SPLIT_REG_ID_HALO_POS   // _REG_ID_EXTRA_NEG_Y_HALO_POS_Z
};

//Direction of halo exchange with respect to the receiving side.
enum RECV_TYPE_2D
{
    _RECV_TYPE_2D_NEGX,
    _RECV_TYPE_2D_POSX,
    _RECV_TYPE_2D_NEGZ,
    _RECV_TYPE_2D_POSZ,
    _RECV_TYPE_2D_NEGX_NEGZ,
    _RECV_TYPE_2D_NEGX_POSZ,
    _RECV_TYPE_2D_POSX_NEGZ,
    _RECV_TYPE_2D_POSX_POSZ,
    _RECV_TYPE_2D_TOTAL
};

const RECV_TYPE_2D RECV_TYPE_2D_MAP[_RECV_TYPE_2D_TOTAL] = { _RECV_TYPE_2D_NEGX,      _RECV_TYPE_2D_POSX,
                                                             _RECV_TYPE_2D_NEGZ,      _RECV_TYPE_2D_POSZ,
                                                             _RECV_TYPE_2D_NEGX_NEGZ, _RECV_TYPE_2D_NEGX_POSZ,
                                                             _RECV_TYPE_2D_POSX_NEGZ, _RECV_TYPE_2D_POSX_POSZ };

const int RECV_TYPE_2D_DIM[_RECV_TYPE_2D_TOTAL] = { 1, 1, 1, 1, 2, 2, 2, 2 };

enum RECV_TYPE_2D_DIM1
{
    _RECV_TYPE_2D_DIM1_NEGX,
    _RECV_TYPE_2D_DIM1_POSX,
    _RECV_TYPE_2D_DIM1_NEGZ,
    _RECV_TYPE_2D_DIM1_POSZ,
    _RECV_TYPE_2D_DIM1_TOTAL
};

enum RECV_TYPE_2D_DIM2
{
    _RECV_TYPE_2D_DIM2_NEGX_NEGZ,
    _RECV_TYPE_2D_DIM2_NEGX_POSZ,
    _RECV_TYPE_2D_DIM2_POSX_NEGZ,
    _RECV_TYPE_2D_DIM2_POSX_POSZ,
    _RECV_TYPE_2D_DIM2_TOTAL
};

const int RECV_TYPE_2D_DIM_MAP[_RECV_TYPE_2D_TOTAL] = { _RECV_TYPE_2D_DIM1_NEGX,      _RECV_TYPE_2D_DIM1_POSX,
                                                        _RECV_TYPE_2D_DIM1_NEGZ,      _RECV_TYPE_2D_DIM1_POSZ,
                                                        _RECV_TYPE_2D_DIM2_NEGX_NEGZ, _RECV_TYPE_2D_DIM2_NEGX_POSZ,
                                                        _RECV_TYPE_2D_DIM2_POSX_NEGZ, _RECV_TYPE_2D_DIM2_POSX_POSZ };

//Direction of halo exchange with respect to the receiving side.
//POSY2 is only used for MPI exchanges using the data-Block layout.
enum RECV_TYPE_3D
{
    _RECV_TYPE_3D_NEGX,
    _RECV_TYPE_3D_POSX,
    _RECV_TYPE_3D_NEGY,
    _RECV_TYPE_3D_POSY1,
    _RECV_TYPE_3D_POSY2,
    _RECV_TYPE_3D_NEGZ,
    _RECV_TYPE_3D_POSZ,
    _RECV_TYPE_3D_NEGY_NEGZ,
    _RECV_TYPE_3D_NEGY_POSZ,
    _RECV_TYPE_3D_POSY_NEGZ,
    _RECV_TYPE_3D_POSY_POSZ,
    _RECV_TYPE_3D_NEGX_NEGY,
    _RECV_TYPE_3D_NEGX_POSY1,
    _RECV_TYPE_3D_NEGX_POSY2,
    _RECV_TYPE_3D_POSX_NEGY,
    _RECV_TYPE_3D_POSX_POSY1,
    _RECV_TYPE_3D_POSX_POSY2,
    _RECV_TYPE_3D_NEGX_NEGZ,
    _RECV_TYPE_3D_NEGX_POSZ,
    _RECV_TYPE_3D_POSX_NEGZ,
    _RECV_TYPE_3D_POSX_POSZ,
    _RECV_TYPE_3D_NEGX_NEGY_NEGZ,
    _RECV_TYPE_3D_NEGX_NEGY_POSZ,
    _RECV_TYPE_3D_NEGX_POSY_NEGZ,
    _RECV_TYPE_3D_NEGX_POSY_POSZ,
    _RECV_TYPE_3D_POSX_NEGY_NEGZ,
    _RECV_TYPE_3D_POSX_NEGY_POSZ,
    _RECV_TYPE_3D_POSX_POSY_NEGZ,
    _RECV_TYPE_3D_POSX_POSY_POSZ,
    _RECV_TYPE_3D_TOTAL
};

const RECV_TYPE_3D RECV_TYPE_3D_MAP[_RECV_TYPE_3D_TOTAL] = {
    _RECV_TYPE_3D_NEGX,           _RECV_TYPE_3D_POSX,           _RECV_TYPE_3D_NEGY,
    _RECV_TYPE_3D_POSY1,          _RECV_TYPE_3D_POSY2,          _RECV_TYPE_3D_NEGZ,
    _RECV_TYPE_3D_POSZ,           _RECV_TYPE_3D_NEGY_NEGZ,      _RECV_TYPE_3D_NEGY_POSZ,
    _RECV_TYPE_3D_POSY_NEGZ,      _RECV_TYPE_3D_POSY_POSZ,      _RECV_TYPE_3D_NEGX_NEGY,
    _RECV_TYPE_3D_NEGX_POSY1,     _RECV_TYPE_3D_NEGX_POSY2,     _RECV_TYPE_3D_POSX_NEGY,
    _RECV_TYPE_3D_POSX_POSY1,     _RECV_TYPE_3D_POSX_POSY2,     _RECV_TYPE_3D_NEGX_NEGZ,
    _RECV_TYPE_3D_NEGX_POSZ,      _RECV_TYPE_3D_POSX_NEGZ,      _RECV_TYPE_3D_POSX_POSZ,
    _RECV_TYPE_3D_NEGX_NEGY_NEGZ, _RECV_TYPE_3D_NEGX_NEGY_POSZ, _RECV_TYPE_3D_NEGX_POSY_NEGZ,
    _RECV_TYPE_3D_NEGX_POSY_POSZ, _RECV_TYPE_3D_POSX_NEGY_NEGZ, _RECV_TYPE_3D_POSX_NEGY_POSZ,
    _RECV_TYPE_3D_POSX_POSY_NEGZ, _RECV_TYPE_3D_POSX_POSY_POSZ
};

const int RECV_TYPE_3D_DIM[_RECV_TYPE_3D_TOTAL] = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };

enum RECV_TYPE_3D_DIM1
{
    _RECV_TYPE_3D_DIM1_NEGX,
    _RECV_TYPE_3D_DIM1_POSX,
    _RECV_TYPE_3D_DIM1_NEGY,
    _RECV_TYPE_3D_DIM1_POSY1,
    _RECV_TYPE_3D_DIM1_POSY2,
    _RECV_TYPE_3D_DIM1_NEGZ,
    _RECV_TYPE_3D_DIM1_POSZ,
    _RECV_TYPE_3D_DIM1_TOTAL
};

enum RECV_TYPE_3D_DIM2
{
    _RECV_TYPE_3D_DIM2_NEGY_NEGZ,
    _RECV_TYPE_3D_DIM2_NEGY_POSZ,
    _RECV_TYPE_3D_DIM2_POSY_NEGZ,
    _RECV_TYPE_3D_DIM2_POSY_POSZ,
    _RECV_TYPE_3D_DIM2_NEGX_NEGY,
    _RECV_TYPE_3D_DIM2_NEGX_POSY1,
    _RECV_TYPE_3D_DIM2_NEGX_POSY2,
    _RECV_TYPE_3D_DIM2_POSX_NEGY,
    _RECV_TYPE_3D_DIM2_POSX_POSY1,
    _RECV_TYPE_3D_DIM2_POSX_POSY2,
    _RECV_TYPE_3D_DIM2_NEGX_NEGZ,
    _RECV_TYPE_3D_DIM2_NEGX_POSZ,
    _RECV_TYPE_3D_DIM2_POSX_NEGZ,
    _RECV_TYPE_3D_DIM2_POSX_POSZ,
    _RECV_TYPE_3D_DIM2_TOTAL
};

enum RECV_TYPE_3D_DIM3
{
    _RECV_TYPE_3D_DIM3_NEGX_NEGY_NEGZ,
    _RECV_TYPE_3D_DIM3_NEGX_NEGY_POSZ,
    _RECV_TYPE_3D_DIM3_NEGX_POSY_NEGZ,
    _RECV_TYPE_3D_DIM3_NEGX_POSY_POSZ,
    _RECV_TYPE_3D_DIM3_POSX_NEGY_NEGZ,
    _RECV_TYPE_3D_DIM3_POSX_NEGY_POSZ,
    _RECV_TYPE_3D_DIM3_POSX_POSY_NEGZ,
    _RECV_TYPE_3D_DIM3_POSX_POSY_POSZ,
    _RECV_TYPE_3D_DIM3_TOTAL
};

const int RECV_TYPE_3D_DIM_MAP[_RECV_TYPE_3D_TOTAL] = {
    _RECV_TYPE_3D_DIM1_NEGX,           _RECV_TYPE_3D_DIM1_POSX,           _RECV_TYPE_3D_DIM1_NEGY,
    _RECV_TYPE_3D_DIM1_POSY1,          _RECV_TYPE_3D_DIM1_POSY2,          _RECV_TYPE_3D_DIM1_NEGZ,
    _RECV_TYPE_3D_DIM1_POSZ,           _RECV_TYPE_3D_DIM2_NEGY_NEGZ,      _RECV_TYPE_3D_DIM2_NEGY_POSZ,
    _RECV_TYPE_3D_DIM2_POSY_NEGZ,      _RECV_TYPE_3D_DIM2_POSY_POSZ,      _RECV_TYPE_3D_DIM2_NEGX_NEGY,
    _RECV_TYPE_3D_DIM2_NEGX_POSY1,     _RECV_TYPE_3D_DIM2_NEGX_POSY2,     _RECV_TYPE_3D_DIM2_POSX_NEGY,
    _RECV_TYPE_3D_DIM2_POSX_POSY1,     _RECV_TYPE_3D_DIM2_POSX_POSY2,     _RECV_TYPE_3D_DIM2_NEGX_NEGZ,
    _RECV_TYPE_3D_DIM2_NEGX_POSZ,      _RECV_TYPE_3D_DIM2_POSX_NEGZ,      _RECV_TYPE_3D_DIM2_POSX_POSZ,
    _RECV_TYPE_3D_DIM3_NEGX_NEGY_NEGZ, _RECV_TYPE_3D_DIM3_NEGX_NEGY_POSZ, _RECV_TYPE_3D_DIM3_NEGX_POSY_NEGZ,
    _RECV_TYPE_3D_DIM3_NEGX_POSY_POSZ, _RECV_TYPE_3D_DIM3_POSX_NEGY_NEGZ, _RECV_TYPE_3D_DIM3_POSX_NEGY_POSZ,
    _RECV_TYPE_3D_DIM3_POSX_POSY_NEGZ, _RECV_TYPE_3D_DIM3_POSX_POSY_POSZ
};

enum ROTATE_TYPE
{
    _ROTATE_FROM_XYZ_TO_ZXY,
    _ROTATE_TYPE_TOTAL
};

#endif