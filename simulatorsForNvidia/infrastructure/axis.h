/*************************************************************************
 * Axis definition 
 *
 * Author: Rahul S. Sampath
 * Change Log: 
 *
 *************************************************************************/

#ifndef _AXIS
#define _AXIS

#include <cstdio>
#include <string>
#include <iostream>
#include "std_const.h"

struct axes3d
{
    int n1;
    realtype o1;
    realtype d1;
    int nghost1;
    int n2;
    realtype o2;
    realtype d2;
    int nghost2;
    int n3;
    realtype o3;
    realtype d3;
    int nghost3;
};
//
// this emum is used to pad the fastest dimension, IN TERMS OF ELEMENTS, as a multiple of
// 32 elements
// or ther
// Attention: axis does not know what kind of data it is going to represent. That's why we are
// using ELEMENTS rather than BYTES
//
enum class AlignMemBytes // in bytes
{
    NOEXTENSION = 0, // no extension at all
    CACHELINE = 128, // best alignement 128 bytes
    VEC16 = 16       // minimum alignemnt for 16 bytes vectors
};

struct AlignmentElem
{ // in mb of elements (bytes/sizeof(kind of data) )
    AlignmentElem() : alignValue(AlignMemBytes::NOEXTENSION), nElems(0) {}

    //Ensure that sizeOfElem is greater than 0, if not nElems is set to 0
    //to prevent a floating point exception.
    AlignmentElem(AlignMemBytes alm, int sizeOfElem)
        : alignValue(alm), nElems(sizeOfElem > 0 ? static_cast<uint>(alm) / sizeOfElem : 0)
    {}

    AlignMemBytes alignValue;
    uint nElems;
};
// FIXME this class has a number of extremely bad problems and
// unfortunately, it is a fairly low-level class for the EMSL
// libraries.
//
// - Constructors requiring writeable pointers for inputs
//
// - Constructors modifying external state
//
// - Multiple sources of truth for information that are not
//   protected to ensure concistency.
//
// - Unecessary number of constructors, some of which can be
//   invoked by accident because of implicit casting.
//
// - String member data that should not be part of this class. They
//   are not included in any of the constructors, but you can just
//   set them whenever because they are not protected.
//
// - Member functions that should be outside of the class definition
//   because they are not part of the minimal but complete iterface.
//
// - Using 32 bit floating-point values is an optimization and should
//   be reserved for code that is performance critical.  The axis class
//   is not performance critical.
//
// - Explicit use of 32 integer values for quantities may be insufficient
//   and there is not any plausible justification for not using a larger
//   type.
//
// All of these factors and more contribute to EMSL code being fragile, and
// reduce the effectivness developers and researchers working in this code.

//! axis object
class axis
{
public:
    //! constructor
    axis(realtype o_, realtype d_, int n_, AlignmentElem alignMem = AlignmentElem(AlignMemBytes::NOEXTENSION, 1));

    //! constructor
    axis(realtype o_, realtype e_, realtype d_, AlignmentElem alignMem = AlignmentElem(AlignMemBytes::NOEXTENSION, 1));

    //! constructor
    axis(realtype o_, realtype d_, int n_, int nghost_,
         AlignmentElem alignMem = AlignmentElem(AlignMemBytes::NOEXTENSION, 1));

    //! copy constructor
    axis(const axis* axin);

    //! copy constructor and alter alignment
    axis(const axis* axin, AlignmentElem alignMem);

    // TODO It makes absolutely no sense to pass a 64bit pointer around
    // to represent a 32 bit float this should all be fixed.
    //
    //! constructs an axis snapped to axin and within the bounds of axin
    //! o_, e_, and d_ may be altered
    axis(const axis* axin, realtype* o_, realtype* e_, realtype* d_);

    //! constructs an axis snapped to axin and within the bounds of axin
    //! o_, e_, and d_ may be altered. but grids will be on o, e, d system not from axin
    axis(realtype* o_, realtype* e_, realtype* d_, axis* axin);

    //Destructor
    ~axis();

    //! scale axis
    void scale(realtype sc);

    //! get linear interpolant
    int get_interp(int* ind1, realtype* val1, realtype* val2, realtype x);

    //! get nearest interior index
    int get_index_interior(realtype x) const;

    //! get nearest index
    int get_index(realtype x) const;

    //! get floored index - returns -1 if out of bounds
    int get_index_round(realtype x);

    // ! round the index
    int get_index_floor(double x);

    int get_index_ceil(double x);

    //! get lower index
    int get_lower_index(realtype x);

    //! get upper index
    int get_upper_index(realtype x);

    //! shift the axis by the specified length
    void shift(realtype shift);

    //! reverse axis
    void reverse();

    //! diagnostic print
    void print(std::string message);
    void print(FILE* fd);

    //! diagnostic print
    void print(FILE* fd, std::string axis_label);

    //! set label
    void set_label(std::string lbl);

    //! set unit
    void set_unit(std::string un);

    //! returns true if the point is within the axis bounds interior (i.e. excluding halo/pad region) and false otherwise.
    bool inbounds_int(realtype x) { return ((x >= o) && (x <= e)); }

    //! returns true if the point is within the interior axis bounds extended by one grid spacing and false otherwise.
    bool inbounds_ext_shared(realtype x) const { return ((x > (o - d)) && (x < (e + d))); }

    //! returns true if the point is inbounds_ext_shared and is owned and false otherwise.
    bool inbounds_ext_unique(realtype x, axis* axg)
    {
        return (((x >= (o - (0.5f * d))) && (x < (e + (0.5f * d)))) || ((axg->o == o) && (x > (o - d)) && (x < o)) ||
                ((axg->e == e) && (x > e) && (x < (e + d))));
    }

    //! get value at index (compute in double precision)
    double get_val_double(int i) const { return (((double)opad) + (((double)i) * ((double)d))); }

    //! get value at index (compute in single precision)
    float get_val(int i) const { return (opad + (((float)i) * d)); } //this is linear interp

    //! get indices to copy from another axis
    int get_copy_indices(axis* ax, int* axbeg, int* mybeg, int* myend);

    //! get indices to copy interior from another axis
    int get_interior_copy_indices(axis* ax, int* axbeg, int* mybeg, int* myend);

    //! get indices to interpolate from another axis
    void get_interp_indices(axis* ax, int* axbeg, int* axend, int* axn);

    //! compute index and weights for linear interpolation
    void computeIdxWtsLinearInterp(int& i, double& w1, double& w2, const realtype p) const;

    /*
    float get_val_cosine_interp(int i);
    float get_val_cubic_interp(int i);
    float get_val_catmull_rom_interp(int i);
    float get_val_hermite_interp(int i);
    */

    //! label
    std::string label;

    //! unit
    std::string unit;

    //! origin
    realtype o;

    //! padded origin
    realtype opad;

    //! location of last point
    realtype e;

    //! location of padded last point
    realtype epad;

    //! sampling delta
    realtype d;

    //! inverse sampling delta
    realtype invd;

    //! number of points
    int n;

    //! number of points including ghosts and trail padding
    int ntot;

    //! Total number of allocated elements for axis length, including trailing padding.
    int nallocated_elements;

    //! number of points including ghosts without trail padding
    int nvalid;

    //! iteration begin
    int ibeg;

    //! iteration end
    int iend;

    //! number of ghost points
    int nghost;

    //! nb to add at the origin so ntot (= n+ npad_leading + npad_trailing) is a mutiple of or another int defined by alignMnt

    int npad_trailing;

    // what kind of alignement has been used for the axis
    AlignmentElem alignMnt;
};

bool operator==(const axis& lhs, const axis& rhs);

#endif
