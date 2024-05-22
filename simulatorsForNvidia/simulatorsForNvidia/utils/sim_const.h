
#ifndef _SIM_CONST
#define _SIM_CONST

enum class CartVolType
{
    _DEFAULT,
    _CPU,
    _GPU_DEVICE,
    _GPU_HOST
};

enum BoundaryType
{
    _BDY_SPONGE,
    _BDY_SPML,
    _BDY_HSPML,
    _BDY_FREE,
    _BDY_TYPE_TOTAL
};

enum BoundaryDirection
{
    _BDY_SIDE,
    _BDY_TOP,
    _BDY_BOTTOM,
    _BDY_DIR_TOTAL
};

enum BoundaryDirectionExpanded
{
    _BDY_SIDE_FACE,
    _BDY_TOP_FACE,
    _BDY_BOTTOM_FACE,
    _BDY_SIDE_EDGE,
    _BDY_SIDE_TOP_EDGE,
    _BDY_SIDE_BOTTOM_EDGE,
    _BDY_SIDE_TOP_CORNER,
    _BDY_SIDE_BOTTOM_CORNER,
    _BDY_DIR_EXP_TOTAL
};

enum EquationType
{
    _EQ_FIRST_ORDER_STAGGERRED_GRID,
    _EQ_SECOND_ORDER_CENTRAL_GRID
};

#endif
