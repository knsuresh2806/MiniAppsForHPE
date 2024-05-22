#include "fd_coef.h"

using std::vector;

/** Get 1st order staggered FD coefficient
 * @param order: FD order
 * @param c: c[i] = i-th coefficients, 1 <= i <= order/2, c[0] is not used
 */
template <>
void
get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_TAYLOR>(int order, vector<realtype>& c)
{
    using namespace TAYLOR;

    c.resize(order / 2 + 1);
    c[0] = 0.0;

    switch (order) {
        case 4:
            c[1] = DF_order4_1;
            c[2] = DF_order4_2;
            break;

        case 6:
            c[1] = DF_order6_1;
            c[2] = DF_order6_2;
            c[3] = DF_order6_3;
            break;

        case 8:
            c[1] = DF_order8_1;
            c[2] = DF_order8_2;
            c[3] = DF_order8_3;
            c[4] = DF_order8_4;
            break;

        default:
            throw(-1);
    }
}

/** Get 1st order staggered FD coefficient
 * @param order: FD order
 * @param c: c[i] = i-th coefficients, 1 <= i <= order/2, c[0] is not used
 */
template <>
void
get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_LEAST_SQUARE>(int order, vector<realtype>& c)
{
    using namespace LEAST_SQUARE;

    c.resize(order / 2 + 1);
    c[0] = 0.0;

    switch (order) {
        case 4:
            c[1] = DF_order4_1;
            c[2] = DF_order4_2;
            break;

        case 6:
            c[1] = DF_order6_1;
            c[2] = DF_order6_2;
            c[3] = DF_order6_3;
            break;

        case 8:
            c[1] = DF_order8_1;
            c[2] = DF_order8_2;
            c[3] = DF_order8_3;
            c[4] = DF_order8_4;
            break;

        case 12:
            c[1] = DF_order12_1;
            c[2] = DF_order12_2;
            c[3] = DF_order12_3;
            c[4] = DF_order12_4;
            c[5] = DF_order12_5;
            c[6] = DF_order12_6;
            break;

        case 16:
            c[1] = DF_order16_1;
            c[2] = DF_order16_2;
            c[3] = DF_order16_3;
            c[4] = DF_order16_4;
            c[5] = DF_order16_5;
            c[6] = DF_order16_6;
            c[7] = DF_order16_7;
            c[8] = DF_order16_8;
            break;

        default:
            throw(-1);
    }
}

/** Get 2nd order non-staggered FD coefficient
 * @param order: FD order
 * @param c: c[i] = i-th coefficients, 0 <= i <= order/2
 */
template <>
void
get_fd_coef<_EQ_SECOND_ORDER_CENTRAL_GRID, _FD_COEF_LEAST_SQUARE>(int order, vector<realtype>& c)
{
    using namespace LEAST_SQUARE;

    c.resize(order / 2 + 1);

    switch (order) {
        case 8:
            c[0] = DF2_order8_0;
            c[1] = DF2_order8_1;
            c[2] = DF2_order8_2;
            c[3] = DF2_order8_3;
            c[4] = DF2_order8_4;
            break;

        case 12:
            c[0] = DF2_order12_0;
            c[1] = DF2_order12_1;
            c[2] = DF2_order12_2;
            c[3] = DF2_order12_3;
            c[4] = DF2_order12_4;
            c[5] = DF2_order12_5;
            c[6] = DF2_order12_6;
            break;

        case 16:
            c[0] = DF2_order16_0;
            c[1] = DF2_order16_1;
            c[2] = DF2_order16_2;
            c[3] = DF2_order16_3;
            c[4] = DF2_order16_4;
            c[5] = DF2_order16_5;
            c[6] = DF2_order16_6;
            c[7] = DF2_order16_7;
            c[8] = DF2_order16_8;
            break;

        default:
            throw(-1);
    }
}

/** Get 1st order staggered FD coefficient
 * @param order: FD order
 * @param c: c[i] = i-th coefficients, 1 <= i <= order/2, c[0] is not used
 */
template <>
void
get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_TAYLOR_PML>(int order, vector<realtype>& c)
{
    using namespace TAYLOR_PML;

    c.resize(order / 2 + 1);
    c[0] = 0.0;

    switch (order) {
        case 4:
            c[1] = DF_order4_1;
            c[2] = DF_order4_2;
            break;

        case 6:
            c[1] = DF_order6_1;
            c[2] = DF_order6_2;
            c[3] = DF_order6_3;
            break;

        case 8:
            c[1] = DF_order8_1;
            c[2] = DF_order8_2;
            c[3] = DF_order8_3;
            c[4] = DF_order8_4;
            break;

        default:
            throw(-1);
    }
}

/** Get 2nd order non-staggered FD coefficient
 * @param order: FD order
 * @param c: c[i] = i-th coefficients, 0 <= i <= order/2
 */
template <>
void
get_fd_coef<_EQ_SECOND_ORDER_CENTRAL_GRID, _FD_COEF_TAYLOR_PML>(int order, vector<realtype>& c)
{
    using namespace TAYLOR_PML;

    c.resize(order / 2 + 1);

    switch (order) {
        case 8:
            c[0] = DF2_order8_0;
            c[1] = DF2_order8_1;
            c[2] = DF2_order8_2;
            c[3] = DF2_order8_3;
            c[4] = DF2_order8_4;
            break;

        case 12:
            c[0] = DF2_order12_0;
            c[1] = DF2_order12_1;
            c[2] = DF2_order12_2;
            c[3] = DF2_order12_3;
            c[4] = DF2_order12_4;
            c[5] = DF2_order12_5;
            c[6] = DF2_order12_6;
            break;

        case 16:
            c[0] = DF2_order16_0;
            c[1] = DF2_order16_1;
            c[2] = DF2_order16_2;
            c[3] = DF2_order16_3;
            c[4] = DF2_order16_4;
            c[5] = DF2_order16_5;
            c[6] = DF2_order16_6;
            c[7] = DF2_order16_7;
            c[8] = DF2_order16_8;
            break;

        default:
            throw(-1);
    }
}

/** Get midpoint interpolation coefficient
 * @param order: interpolation order
 * @param c: c[i] = i-th coefficient, 0 <= i < order/2
 */
void
get_interp_coef(int order, vector<realtype>& c)
{
    using namespace TAYLOR;

    c.resize(order / 2);

    switch (order) {
        case 4:
            c[0] = INTERP_order4_0;
            c[1] = INTERP_order4_1;
            break;

        case 8:
            c[0] = INTERP_order8_0;
            c[1] = INTERP_order8_1;
            c[2] = INTERP_order8_2;
            c[3] = INTERP_order8_3;
            break;

        default:
            throw(-1);
    }
}
