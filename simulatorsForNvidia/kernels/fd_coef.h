#ifndef _FD_COEF
#define _FD_COEF

#include "sim_const.h"
#include <vector>
#include "std_const.h"

#ifdef EMSL_TARGET_DOUBLE

//Coefficients from Taylor expansion
namespace TAYLOR {

//First order coefficients

const realtype DF_order2_1 = (1.00000000000000e+00);

const realtype DF_order4_1 = (9.0 / 8);
const realtype DF_order4_2 = (-1.0 / 24);

const realtype DF_order6_1 = (1.17187500000000e+00);
const realtype DF_order6_2 = (-6.51041666666667e-02);
const realtype DF_order6_3 = (4.68750000000000e-03);

const realtype DF_order8_1 = (1.19628906250000e+00);
const realtype DF_order8_2 = (-7.97526041666667e-02);
const realtype DF_order8_3 = (9.57031250000000e-03);
const realtype DF_order8_4 = (-6.97544642857143e-04);

const realtype DF_order10_1 = (1.21124267578125e+00);
const realtype DF_order10_2 = (-8.97216796875000e-02);
const realtype DF_order10_3 = (1.38427734375000e-02);
const realtype DF_order10_4 = (-1.76565987723214e-03);
const realtype DF_order10_5 = (1.18679470486111e-04);

const realtype DF_order12_1 = (1.22133636474609e+00);
const realtype DF_order12_2 = (-9.69314575195312e-02);
const realtype DF_order12_3 = (1.74476623535156e-02);
const realtype DF_order12_4 = (-2.96728951590402e-03);
const realtype DF_order12_5 = (3.59005398220486e-04);
const realtype DF_order12_6 = (-2.18478116122159e-05);

const realtype DF_order14_1 = (1.22860622406006e+00);
const realtype DF_order14_2 = (-1.02383852005005e-01);
const realtype DF_order14_3 = (2.04767704010010e-02);
const realtype DF_order14_4 = (-4.17893273489816e-03);
const realtype DF_order14_5 = (6.89453548855252e-04);
const realtype DF_order14_6 = (-7.69225033846768e-05);
const realtype DF_order14_7 = (4.23651475172776e-06);

// Central Grid, first order derivatives
// the central scheme work this way
// sum(coef_i(f(x+h*i) - f(x-h_i)))
const realtype DF_C_order2_1 = (1.0 / 2.0);

const realtype DF_C_order4_1 = (2.0 / 3.0);
const realtype DF_C_order4_2 = (-1.0 / 12.0);

const realtype DF_C_order6_1 = (3.0 / 4.0);
const realtype DF_C_order6_2 = (-3.0 / 20.0);
const realtype DF_C_order6_3 = (1.0 / 60.0);

const realtype DF_C_order8_1 = (4.0 / 5.0);
const realtype DF_C_order8_2 = (-1.0 / 5.0);
const realtype DF_C_order8_3 = (4.0 / 105.0);
const realtype DF_C_order8_4 = (-1.0 / 280.0);

const realtype DF_C_order10_1 = (5.0 / 6.0);
const realtype DF_C_order10_2 = (-5.0 / 21.0);
const realtype DF_C_order10_3 = (5.0 / 84.0);
const realtype DF_C_order10_4 = (-5.0 / 504.0);
const realtype DF_C_order10_5 = (1.0 / 1260.0);

//Second order coefficients

// Central Grid
// Laplacian (from MTC = Taylor)
const realtype DF2_order2_0 = (-2.0);
const realtype DF2_order2_1 = (1.0);

const realtype DF2_order4_0 = (-5.0 / 2.0);
const realtype DF2_order4_1 = (4.0 / 3.0);
const realtype DF2_order4_2 = (-1.0 / 12.0);

const realtype DF2_order8_0 = (-205.0 / 72.0);
const realtype DF2_order8_1 = (8.0 / 5.0);
const realtype DF2_order8_2 = (-1.0 / 5.0);
const realtype DF2_order8_3 = (8.0 / 315.0);
const realtype DF2_order8_4 = (-1.0 / 560.0);

//Taylor coefficients - From the LS-RTM Born Density kernel merge from Research
//const realtype DFC_order16_8  = (-9.71250971208e-06f);
//const realtype DFC_order16_7 =  (1.77600177593e-04f);
//const realtype DFC_order16_6 =  (-1.55400155395e-03f);
//const realtype DFC_order16_5 =  (8.70240870221e-03f);
//const realtype DFC_order16_4 =  (-3.53535353532e-02f);
//const realtype DFC_order16_3 =  (1.13131313132e-01f);
//const realtype DFC_order16_2 =  (-3.11111111131e-01f);
//const realtype DFC_order16_1 =  (8.88888888899e-01f);
//const realtype DFC_order16_0 =  0.0f;

// CJ: 16-th order (M=8) use LS-optimized coefficients for now to be consistent with strain kernel
const realtype DF2_order16_0 = (-0.3162230E+1);
const realtype DF2_order16_1 = (0.1875943E+1);
const realtype DF2_order16_2 = (-0.3859840E+0);
const realtype DF2_order16_3 = (0.1228261E+0);
const realtype DF2_order16_4 = (-0.4233006E-1);
const realtype DF2_order16_5 = (0.1379639E-1);
const realtype DF2_order16_6 = (-0.3846673E-2);
const realtype DF2_order16_7 = (0.8026637E-3);
const realtype DF2_order16_8 = (-0.9257725E-4);
//

/* CJ: 16-th order Taylor   
  const realtype DF2_order16_0 = (-3.05484E+0f);  
  const realtype DF2_order16_1 = ( 1.77778E+0f);
  const realtype DF2_order16_2 = (-3.11111E-1f);
  const realtype DF2_order16_3 = ( 7.54209E-2f);
  const realtype DF2_order16_4 = (-1.76768E-2f);
  const realtype DF2_order16_5 = ( 3.48096E-3f);
  const realtype DF2_order16_6 = (-5.18001E-4f);
  const realtype DF2_order16_7 = ( 5.07429E-5f);
  const realtype DF2_order16_8 = (-2.42813E-6f);
*/

// Trial runs for compensated dispersion
//const realtype DF2_order4_0 = (-2.55567466e+00f);
//const realtype DF2_order4_1 = (1.37106192e+00f);
//const realtype DF2_order4_2 = (-9.322459e-02f);

//const realtype DF2_order8_0 = (-2.97399944e+00f);
//const realtype DF2_order8_1 = (1.70507669e+00f);
//const realtype DF2_order8_2 = (-2.5861812e-01f);
//const realtype DF2_order8_3 = (4.577745e-02f);
//const realtype DF2_order8_4 = (-5.23630e-03f);

//Central Grid - expanded stencils for better dispersion

const realtype DF2_order4_7pt_0 = (-2.534722222222222e+00);
const realtype DF2_order4_7pt_1 = (1.359375000000000e+00);
const realtype DF2_order4_7pt_2 = (-9.375000000000000e-02);
const realtype DF2_order4_7pt_3 = (1.736111111111111e-03);

const realtype DF2_order8_15pt_0 = (-2.875120152756741e+00);
const realtype DF2_order8_15pt_1 = (1.623461723327637e+00);
const realtype DF2_order8_15pt_2 = (-2.138233184814454e-01);
const realtype DF2_order8_15pt_3 = (3.092712826199002e-02);
const realtype DF2_order8_15pt_4 = (-3.195444742838543e-03);
const realtype DF2_order8_15pt_5 = (2.028528849283855e-04);
const realtype DF2_order8_15pt_6 = (-1.335144042968750e-05);
const realtype DF2_order8_15pt_7 = (4.865685287786992e-07);

// Mid-point interpolation
const realtype INTERP_order4_0 = 5.625e-01;
const realtype INTERP_order4_1 = -6.250e-02;

const realtype INTERP_order8_0 = 5.981445e-01;
const realtype INTERP_order8_1 = -1.196289e-01;
const realtype INTERP_order8_2 = 2.392578e-02;
const realtype INTERP_order8_3 = -2.441406e-03;
} //end namespace TAYLOR

//Coefficients based on least-squares optimization
// From table 6 of Liu's paper
// Maximum relative error in Vp^2 is 1E-5, so relative error in Vp is 0.5E-5
namespace LEAST_SQUARE {

//First order coefficients

//TODO: REPLACE THESE TAYLOR COEFFICIENTS WITH LEAST_SQUARE COEFFICIENTS
const realtype DF_order4_1 = (9.0 / 8);
const realtype DF_order4_2 = (-1.0 / 24);

const realtype DF_order6_1 = (1.17187500000000e+00);
const realtype DF_order6_2 = (-6.51041666666667e-02);
const realtype DF_order6_3 = (4.68750000000000e-03);

const realtype DF_order8_1 = (1.19628906250000e+00);
const realtype DF_order8_2 = (-7.97526041666667e-02);
const realtype DF_order8_3 = (9.57031250000000e-03);
const realtype DF_order8_4 = (-6.97544642857143e-04);

const realtype DF_order12_1 = (1.240312E+00);
const realtype DF_order12_2 = (-1.114139E-01);
const realtype DF_order12_3 = (2.575906E-02);
const realtype DF_order12_4 = (-6.420759E-03);
const realtype DF_order12_5 = (1.309587E-03);
const realtype DF_order12_6 = (-1.551277E-04);

const realtype DF_order16_1 = (1.254298E+00);
const realtype DF_order16_2 = (-1.234902E-01);
const realtype DF_order16_3 = (3.470321E-02);
const realtype DF_order16_4 = (-1.201822E-02);
const realtype DF_order16_5 = (4.188332E-03);
const realtype DF_order16_6 = (-1.310232E-03);
const realtype DF_order16_7 = (3.233353E-04);
const realtype DF_order16_8 = (-4.675068E-05);

//First order coefficients central grid coefficients

//TODO: REPLACE THESE TAYLOR COEFFICIENTS WITH LEAST_SQUARE COEFFICIENTS
//TODO: add more different orders of the coefficients
const realtype DFC_order8_1 = (0.80000e+00);
const realtype DFC_order8_2 = (-0.2000e+00);
const realtype DFC_order8_3 = (3.80952381e-02);
const realtype DFC_order8_4 = (-3.57142857e-03);

const realtype DFC_order10_1 = (5.0f / 6.0f);
const realtype DFC_order10_2 = (-5.0f / 21.0f);
const realtype DFC_order10_3 = (5.0f / 84.0f);
const realtype DFC_order10_4 = (-5.0f / 504.0f);
const realtype DFC_order10_5 = (1.0f / 1260.0f);

//LS coefficients order 16
const realtype DFC_order16_8 = -0.00096808f;
const realtype DFC_order16_7 = 0.00508888f;
const realtype DFC_order16_6 = -0.0168854f;
const realtype DFC_order16_5 = 0.04386203f;
const realtype DFC_order16_4 = -0.09774383f;
const realtype DFC_order16_3 = 0.19899179f;
const realtype DFC_order16_2 = -0.39899235f;
const realtype DFC_order16_1 = 0.94563191f;
const realtype DFC_order16_0 = 0.0f;
//Second order coefficients

// 8-th order (M=4)
// Trusted region k*h < 1.13
const realtype DF2_order8_0 = (-0.2903117E+1);
const realtype DF2_order8_1 = (0.1645351E+1);
const realtype DF2_order8_2 = (-0.2236606E+0);
const realtype DF2_order8_3 = (0.3265851E-1);
const realtype DF2_order8_4 = (-0.2790073E-2);

// 12-th order (M=6)
// Trusted region k*h < 1.71
const realtype DF2_order12_0 = (-0.3076778E+1);
const realtype DF2_order12_1 = (0.1796858E+1);
const realtype DF2_order12_2 = (-0.3234789E+0);
const realtype DF2_order12_3 = (0.8104263E-1);
const realtype DF2_order12_4 = (-0.1913579E-1);
const realtype DF2_order12_5 = (0.3443490E-2);
const realtype DF2_order12_6 = (-0.3401326E-3);

// 16-th order (M=8)
// Trusted region k*h < 2.08
const realtype DF2_order16_0 = (-0.3162230E+1);
const realtype DF2_order16_1 = (0.1875943E+1);
const realtype DF2_order16_2 = (-0.3859840E+0);
const realtype DF2_order16_3 = (0.1228261E+0);
const realtype DF2_order16_4 = (-0.4233006E-1);
const realtype DF2_order16_5 = (0.1379639E-1);
const realtype DF2_order16_6 = (-0.3846673E-2);
const realtype DF2_order16_7 = (0.8026637E-3);
const realtype DF2_order16_8 = (-0.9257725E-4);

} //end namespace LEAST_SQUARE

namespace TAYLOR_PML {
// Specific for PML zone
// 1st-order coefficients identical to TAYLOR
// 2nd-order coefficients by convolving two 1st-order staggered coefficients
const realtype DF_order4_1 = (9.0 / 8);
const realtype DF_order4_2 = (-1.0 / 24);

const realtype DF_order6_1 = (75.0 / 64);
const realtype DF_order6_2 = (-25.0 / 384);
const realtype DF_order6_3 = (3.0 / 640);

const realtype DF_order8_1 = (1225.0 / 1024);
const realtype DF_order8_2 = (-245.0 / 3072);
const realtype DF_order8_3 = (49.0 / 5120);
const realtype DF_order8_4 = (-5.0 / 7168);

const realtype DF2_order8_0 = (-2.5347222);
const realtype DF2_order8_1 = (1.3593750);
const realtype DF2_order8_2 = (-0.0937500);
const realtype DF2_order8_3 = (0.0017361);
const realtype DF2_order8_4 = (0.0000000);

const realtype DF2_order12_0 = (-2.7551031);
const realtype DF2_order12_1 = (1.5264893);
const realtype DF2_order12_2 = (-0.1635742);
const realtype DF2_order12_3 = (0.0152249);
const realtype DF2_order12_4 = (-0.0006104);
const realtype DF2_order12_5 = (0.0000220);
const realtype DF2_order12_6 = (0.0000000);

const realtype DF2_order16_0 = (-2.8751202);
const realtype DF2_order16_1 = (1.6234617);
const realtype DF2_order16_2 = (-0.2138233);
const realtype DF2_order16_3 = (0.0309271);
const realtype DF2_order16_4 = (-0.0031954);
const realtype DF2_order16_5 = (0.0002029);
const realtype DF2_order16_6 = (-0.0000134);
const realtype DF2_order16_7 = (0.0000005);
const realtype DF2_order16_8 = (0.0000000);
} // end of namespace TAYLOR_PML

#else

//Coefficients from Taylor expansion
namespace TAYLOR {

//First order coefficients

const float DF_order2_1 = (1.00000000000000e+00f);

const float DF_order4_1 = (9.0f / 8);
const float DF_order4_2 = (-1.0f / 24);

const float DF_order6_1 = (1.17187500000000e+00f);
const float DF_order6_2 = (-6.51041666666667e-02f);
const float DF_order6_3 = (4.68750000000000e-03f);

const float DF_order8_1 = (1.19628906250000e+00f);
const float DF_order8_2 = (-7.97526041666667e-02f);
const float DF_order8_3 = (9.57031250000000e-03f);
const float DF_order8_4 = (-6.97544642857143e-04f);

const float DF_order10_1 = (1.21124267578125e+00f);
const float DF_order10_2 = (-8.97216796875000e-02f);
const float DF_order10_3 = (1.38427734375000e-02f);
const float DF_order10_4 = (-1.76565987723214e-03f);
const float DF_order10_5 = (1.18679470486111e-04f);

const float DF_order12_1 = (1.22133636474609e+00f);
const float DF_order12_2 = (-9.69314575195312e-02f);
const float DF_order12_3 = (1.74476623535156e-02f);
const float DF_order12_4 = (-2.96728951590402e-03f);
const float DF_order12_5 = (3.59005398220486e-04f);
const float DF_order12_6 = (-2.18478116122159e-05f);

const float DF_order14_1 = (1.22860622406006e+00f);
const float DF_order14_2 = (-1.02383852005005e-01f);
const float DF_order14_3 = (2.04767704010010e-02f);
const float DF_order14_4 = (-4.17893273489816e-03f);
const float DF_order14_5 = (6.89453548855252e-04f);
const float DF_order14_6 = (-7.69225033846768e-05f);
const float DF_order14_7 = (4.23651475172776e-06f);

// Central Grid, first order derivatives
// the central scheme work this way
// sum(coef_i(f(x+h*i) - f(x-h_i)))
const float DF_C_order2_1 = (1.0f / 2.0f);

const float DF_C_order4_1 = (2.0f / 3.0f);
const float DF_C_order4_2 = (-1.0f / 12.0f);

const float DF_C_order6_1 = (3.0f / 4.0f);
const float DF_C_order6_2 = (-3.0f / 20.0f);
const float DF_C_order6_3 = (1.0f / 60.0f);

const float DF_C_order8_1 = (4.0f / 5.0f);
const float DF_C_order8_2 = (-1.0f / 5.0f);
const float DF_C_order8_3 = (4.0f / 105.0f);
const float DF_C_order8_4 = (-1.0f / 280.0f);

const float DF_C_order10_1 = (5.0f / 6.0f);
const float DF_C_order10_2 = (-5.0f / 21.0f);
const float DF_C_order10_3 = (5.0f / 84.0f);
const float DF_C_order10_4 = (-5.0f / 504.0f);
const float DF_C_order10_5 = (1.0f / 1260.0f);

//Second order coefficients

// Central Grid
// Laplacian (from MTC = Taylor)
const float DF2_order2_0 = (-2.0f);
const float DF2_order2_1 = (1.0f);

const float DF2_order4_0 = (-5.0f / 2.0f);
const float DF2_order4_1 = (4.0f / 3.0f);
const float DF2_order4_2 = (-1.0f / 12.0f);

const float DF2_order8_0 = (-205.0f / 72.0f);
const float DF2_order8_1 = (8.0f / 5.0f);
const float DF2_order8_2 = (-1.0f / 5.0f);
const float DF2_order8_3 = (8.0f / 315.0f);
const float DF2_order8_4 = (-1.0f / 560.0f);

//Taylor coefficients - From the LS-RTM Born Density kernel merge from Research
//const float DFC_order16_8  = (-9.71250971208e-06f);
//const float DFC_order16_7 =  (1.77600177593e-04f);
//const float DFC_order16_6 =  (-1.55400155395e-03f);
//const float DFC_order16_5 =  (8.70240870221e-03f);
//const float DFC_order16_4 =  (-3.53535353532e-02f);
//const float DFC_order16_3 =  (1.13131313132e-01f);
//const float DFC_order16_2 =  (-3.11111111131e-01f);
//const float DFC_order16_1 =  (8.88888888899e-01f);
//const float DFC_order16_0 =  0.0f;

// CJ: 16-th order (M=8) use LS-optimized coefficients for now to be consistent with strain kernel
const float DF2_order16_0 = (-0.3162230E+1f);
const float DF2_order16_1 = (0.1875943E+1f);
const float DF2_order16_2 = (-0.3859840E+0f);
const float DF2_order16_3 = (0.1228261E+0f);
const float DF2_order16_4 = (-0.4233006E-1f);
const float DF2_order16_5 = (0.1379639E-1f);
const float DF2_order16_6 = (-0.3846673E-2f);
const float DF2_order16_7 = (0.8026637E-3f);
const float DF2_order16_8 = (-0.9257725E-4f);
//

/* CJ: 16-th order Taylor   
  const float DF2_order16_0 = (-3.05484E+0f);  
  const float DF2_order16_1 = ( 1.77778E+0f);
  const float DF2_order16_2 = (-3.11111E-1f);
  const float DF2_order16_3 = ( 7.54209E-2f);
  const float DF2_order16_4 = (-1.76768E-2f);
  const float DF2_order16_5 = ( 3.48096E-3f);
  const float DF2_order16_6 = (-5.18001E-4f);
  const float DF2_order16_7 = ( 5.07429E-5f);
  const float DF2_order16_8 = (-2.42813E-6f);
*/

// Trial runs for compensated dispersion
//const float DF2_order4_0 = (-2.55567466e+00f);
//const float DF2_order4_1 = (1.37106192e+00f);
//const float DF2_order4_2 = (-9.322459e-02f);

//const float DF2_order8_0 = (-2.97399944e+00f);
//const float DF2_order8_1 = (1.70507669e+00f);
//const float DF2_order8_2 = (-2.5861812e-01f);
//const float DF2_order8_3 = (4.577745e-02f);
//const float DF2_order8_4 = (-5.23630e-03f);

//Central Grid - expanded stencils for better dispersion

const float DF2_order4_7pt_0 = (-2.534722222222222e+00f);
const float DF2_order4_7pt_1 = (1.359375000000000e+00f);
const float DF2_order4_7pt_2 = (-9.375000000000000e-02f);
const float DF2_order4_7pt_3 = (1.736111111111111e-03f);

const float DF2_order8_15pt_0 = (-2.875120152756741e+00f);
const float DF2_order8_15pt_1 = (1.623461723327637e+00f);
const float DF2_order8_15pt_2 = (-2.138233184814454e-01f);
const float DF2_order8_15pt_3 = (3.092712826199002e-02f);
const float DF2_order8_15pt_4 = (-3.195444742838543e-03f);
const float DF2_order8_15pt_5 = (2.028528849283855e-04f);
const float DF2_order8_15pt_6 = (-1.335144042968750e-05f);
const float DF2_order8_15pt_7 = (4.865685287786992e-07f);

// Mid-point interpolation
const float INTERP_order4_0 = 5.625e-01f;
const float INTERP_order4_1 = -6.250e-02f;

const float INTERP_order8_0 = 5.981445e-01f;
const float INTERP_order8_1 = -1.196289e-01f;
const float INTERP_order8_2 = 2.392578e-02f;
const float INTERP_order8_3 = -2.441406e-03f;
} //end namespace TAYLOR

//Coefficients based on least-squares optimization
// From table 6 of Liu's paper
// Maximum relative error in Vp^2 is 1E-5, so relative error in Vp is 0.5E-5
namespace LEAST_SQUARE {

//First order coefficients

//TODO: REPLACE THESE TAYLOR COEFFICIENTS WITH LEAST_SQUARE COEFFICIENTS
const float DF_order4_1 = (9.0f / 8);
const float DF_order4_2 = (-1.0f / 24);

const float DF_order6_1 = (1.17187500000000e+00f);
const float DF_order6_2 = (-6.51041666666667e-02f);
const float DF_order6_3 = (4.68750000000000e-03f);

const float DF_order8_1 = (1.19628906250000e+00f);
const float DF_order8_2 = (-7.97526041666667e-02f);
const float DF_order8_3 = (9.57031250000000e-03f);
const float DF_order8_4 = (-6.97544642857143e-04f);

const float DF_order12_1 = (1.240312E+00);
const float DF_order12_2 = (-1.114139E-01);
const float DF_order12_3 = (2.575906E-02);
const float DF_order12_4 = (-6.420759E-03);
const float DF_order12_5 = (1.309587E-03);
const float DF_order12_6 = (-1.551277E-04);

const float DF_order16_1 = (1.254298E+00);
const float DF_order16_2 = (-1.234902E-01);
const float DF_order16_3 = (3.470321E-02);
const float DF_order16_4 = (-1.201822E-02);
const float DF_order16_5 = (4.188332E-03);
const float DF_order16_6 = (-1.310232E-03);
const float DF_order16_7 = (3.233353E-04);
const float DF_order16_8 = (-4.675068E-05);

//First order coefficients central grid coefficients

//TODO: REPLACE THESE TAYLOR COEFFICIENTS WITH LEAST_SQUARE COEFFICIENTS
//TODO: add more different orders of the coefficients
const float DFC_order8_1 = (0.80000e+00f);
const float DFC_order8_2 = (-0.2000e+00f);
const float DFC_order8_3 = (3.80952381e-02f);
const float DFC_order8_4 = (-3.57142857e-03f);

const float DFC_order10_1 = (5.0f / 6.0f);
const float DFC_order10_2 = (-5.0f / 21.0f);
const float DFC_order10_3 = (5.0f / 84.0f);
const float DFC_order10_4 = (-5.0f / 504.0f);
const float DFC_order10_5 = (1.0f / 1260.0f);

//LS coefficients order 16
const float DFC_order16_8 = -0.00096808f;
const float DFC_order16_7 = 0.00508888f;
const float DFC_order16_6 = -0.0168854f;
const float DFC_order16_5 = 0.04386203f;
const float DFC_order16_4 = -0.09774383f;
const float DFC_order16_3 = 0.19899179f;
const float DFC_order16_2 = -0.39899235f;
const float DFC_order16_1 = 0.94563191f;
const float DFC_order16_0 = 0.0f;
//Second order coefficients

// 8-th order (M=4)
// Trusted region k*h < 1.13
const float DF2_order8_0 = (-0.2903117E+1f);
const float DF2_order8_1 = (0.1645351E+1f);
const float DF2_order8_2 = (-0.2236606E+0f);
const float DF2_order8_3 = (0.3265851E-1f);
const float DF2_order8_4 = (-0.2790073E-2f);

// 12-th order (M=6)
// Trusted region k*h < 1.71
const float DF2_order12_0 = (-0.3076778E+1f);
const float DF2_order12_1 = (0.1796858E+1f);
const float DF2_order12_2 = (-0.3234789E+0f);
const float DF2_order12_3 = (0.8104263E-1f);
const float DF2_order12_4 = (-0.1913579E-1f);
const float DF2_order12_5 = (0.3443490E-2f);
const float DF2_order12_6 = (-0.3401326E-3f);

// 16-th order (M=8)
// Trusted region k*h < 2.08
const float DF2_order16_0 = (-0.3162230E+1f);
const float DF2_order16_1 = (0.1875943E+1f);
const float DF2_order16_2 = (-0.3859840E+0f);
const float DF2_order16_3 = (0.1228261E+0f);
const float DF2_order16_4 = (-0.4233006E-1f);
const float DF2_order16_5 = (0.1379639E-1f);
const float DF2_order16_6 = (-0.3846673E-2f);
const float DF2_order16_7 = (0.8026637E-3f);
const float DF2_order16_8 = (-0.9257725E-4f);

} //end namespace LEAST_SQUARE

namespace TAYLOR_PML {
// Specific for PML zone
// 1st-order coefficients identical to TAYLOR
// 2nd-order coefficients by convolving two 1st-order staggered coefficients
const float DF_order4_1 = (9.0f / 8);
const float DF_order4_2 = (-1.0f / 24);

const float DF_order6_1 = (75.0 / 64);
const float DF_order6_2 = (-25.0 / 384);
const float DF_order6_3 = (3.0 / 640);

const float DF_order8_1 = (1225.0 / 1024);
const float DF_order8_2 = (-245.0 / 3072);
const float DF_order8_3 = (49.0 / 5120);
const float DF_order8_4 = (-5.0 / 7168);

const float DF2_order8_0 = (-2.5347222);
const float DF2_order8_1 = (1.3593750);
const float DF2_order8_2 = (-0.0937500);
const float DF2_order8_3 = (0.0017361);
const float DF2_order8_4 = (0.0000000);

const float DF2_order12_0 = (-2.7551031);
const float DF2_order12_1 = (1.5264893);
const float DF2_order12_2 = (-0.1635742);
const float DF2_order12_3 = (0.0152249);
const float DF2_order12_4 = (-0.0006104);
const float DF2_order12_5 = (0.0000220);
const float DF2_order12_6 = (0.0000000);

const float DF2_order16_0 = (-2.8751202);
const float DF2_order16_1 = (1.6234617);
const float DF2_order16_2 = (-0.2138233);
const float DF2_order16_3 = (0.0309271);
const float DF2_order16_4 = (-0.0031954);
const float DF2_order16_5 = (0.0002029);
const float DF2_order16_6 = (-0.0000134);
const float DF2_order16_7 = (0.0000005);
const float DF2_order16_8 = (0.0000000);
} // end of namespace TAYLOR_PML

#endif

enum FD_COEF_TYPE
{
    _FD_COEF_TAYLOR,
    _FD_COEF_LEAST_SQUARE,
    _FD_COEF_TAYLOR_PML
};

// Get finite difference coefficient
template <EquationType eType, FD_COEF_TYPE cType>
void get_fd_coef(int order, std::vector<realtype>& c);

template <>
void get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_TAYLOR>(int order, std::vector<realtype>& c);

template <>
void get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_LEAST_SQUARE>(int order, std::vector<realtype>& c);

template <>
void get_fd_coef<_EQ_SECOND_ORDER_CENTRAL_GRID, _FD_COEF_LEAST_SQUARE>(int order, std::vector<realtype>& c);

template <>
void get_fd_coef<_EQ_FIRST_ORDER_STAGGERRED_GRID, _FD_COEF_TAYLOR_PML>(int order, std::vector<realtype>& c);

template <>
void get_fd_coef<_EQ_SECOND_ORDER_CENTRAL_GRID, _FD_COEF_TAYLOR_PML>(int order, std::vector<realtype>& c);

// Get midpoint interpolation coefficient
void get_interp_coef(int order, std::vector<realtype>& c);

#endif
