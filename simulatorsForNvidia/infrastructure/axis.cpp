/*************************************************************************
 * Axis definition 
 *
 * Author: Rahul S. Sampath
 *
 *************************************************************************/

#include "axis.h"
#include "emsl_error.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

//
// nalm is the number  of elements to be multiple of
//
inline uint
getPadTrailing(uint& nalm, const uint& n)
{
    if (nalm == 0)
        return 0;
    // get the int value of the enum

    // npad_trailing is the difference between the fully aligned n_ and the n_ from the input
    return nalm * ((n - 1) / nalm + 1) - n;
}

axis ::axis(realtype o_, realtype d_, int n_, AlignmentElem alignMem) : alignMnt(alignMem)
{

    npad_trailing = getPadTrailing(alignMem.nElems, n_);

    n = n_;
    o = o_;
    d = d_;
    // we don't change n but only ntot. the values inside npad_trailing are not used for anything, just for mem alignment
    ntot = n_ + npad_trailing;

    // opad and epad have to be modified according to npad_trailing
    opad = o;
    e = (((n - 1)) * d) + o;

    epad = e + npad_trailing * d;
    invd = 1. / d;
    nghost = 0;
    ibeg = 0;
    iend = n - 1;

    nvalid = n + (2 * nghost);
    nallocated_elements = nvalid + npad_trailing;
}

axis ::axis(realtype o_, realtype e_, realtype d_, AlignmentElem alignMem) : alignMnt(alignMem)
{

    n = ceil((e_ - o_) / d_ - .5) + 1;

    npad_trailing = getPadTrailing(alignMem.nElems, n);

    // opad and epad have to be modified according to npad_trailing

    ntot = n + npad_trailing;
    o = o_;
    d = d_;
    opad = o;
    e = (((n - 1)) * d) + o;
    epad = e + npad_trailing * d;
    invd = 1. / d;
    nghost = 0;
    ibeg = 0;
    iend = n - 1;

    nvalid = n + (2 * nghost);
    nallocated_elements = nvalid + npad_trailing;
}

axis::axis(realtype o_, realtype d_, int n_, int nghost_, AlignmentElem alignMem) : alignMnt(alignMem)
{

    o = o_;
    d = d_;
    n = n_;

    nghost = nghost_;
    ntot = n + (2 * nghost);
    npad_trailing = getPadTrailing(alignMem.nElems, ntot);

    ntot += npad_trailing;
    opad = o - (nghost * d);
    e = ((n - 1) * d) + o;
    epad = e + (nghost * d) + npad_trailing * d;
    invd = 1. / d;

    ibeg = nghost;
    iend = nghost + n - 1;

    nvalid = n + (2 * nghost);
    nallocated_elements = nvalid + npad_trailing;
}

axis ::axis(const axis* axin)
{
    EMSL_VERIFY(axin != NULL);
    unit = axin->unit;
    label = axin->label;
    o = axin->o;
    opad = axin->opad;
    e = axin->e;
    epad = axin->epad;
    d = axin->d;
    invd = axin->invd;
    n = axin->n;
    ntot = axin->ntot;
    ibeg = axin->ibeg;
    iend = axin->iend;
    nghost = axin->nghost;

    npad_trailing = axin->npad_trailing;
    alignMnt = axin->alignMnt;

    nvalid = axin->nvalid;
    nallocated_elements = axin->nallocated_elements;
}

axis ::axis(const axis* axin, AlignmentElem alignMem)
{
    EMSL_VERIFY(axin != NULL);
    n = axin->n;
    unit = axin->unit;
    label = axin->label;
    o = axin->o;
    opad = axin->opad;
    e = axin->e;
    d = axin->d;
    invd = axin->invd;
    ibeg = axin->ibeg;
    iend = axin->iend;
    nghost = axin->nghost;

    alignMnt = alignMem;
    nvalid = n + (2 * nghost);
    npad_trailing = getPadTrailing(alignMem.nElems, nvalid);
    nallocated_elements = nvalid + npad_trailing;
    epad = e + (nghost * d) + npad_trailing * d;

    ntot = nallocated_elements;
}

// Note: A lack of critique on other functions in this file
// does not imply that they are well coded.
//
// FIXME This is not a reasonable constructor and should be
// eleminated. Here is what it appears to do.
//   1. Checks that axin is not null (+)
//   2. Dereferences o_ without checking if it is null (-) [FIXED]
//   3. Dereferences e_ without checking if it is null (-) [FIXED]
//   4. Dereferences d_ without checking if it is null (-) [FIXED]
//   5. Does some undocumented calculations (~)
//   6. Sets a value at the input pointer o_ (-)
//   7. Sets a value at the input pointer e_ (-)
//   8. Sets a value at the input pointer d_ (-)
//   9. Sets some of the memer variables
//
// From the signature, seems okay that it is changing values
// pointed at by the input pointer. Afterall, they're not
// const pointers.  However, the very concept here seems,
// completely misguided.  To change some the state of some
// some external object during construction is not a good
// idea.
//
// The documentation in axis.h for this constructor is
// contradictory and confusing, but is does say that the
// inputs might be changed.  What is not clear, is if this
// function is using the input values to  intentionally
// modify them or modifying the because why-not.
//
axis ::axis(const axis* axin, realtype* o_, realtype* e_, realtype* d_)
{

    EMSL_VERIFY(axin != NULL);
    EMSL_VERIFY(o_ != NULL);
    EMSL_VERIFY(e_ != NULL);
    EMSL_VERIFY(d_ != NULL);

    realtype wrk_o = *o_;
    realtype wrk_e = *e_;
    realtype wrk_d = *d_;

    wrk_o = std::max(wrk_o, axin->o);
    int itmp = axin->get_index(wrk_o);
    wrk_o = axin->get_val(itmp);
    wrk_e = std::min(wrk_e, axin->e);
    itmp = ceil(wrk_d / axin->d - .5);
    wrk_d = itmp * axin->d;
    itmp = ceil((wrk_e - wrk_o) / wrk_d - .5);
    wrk_e = wrk_o + itmp * wrk_d;
    if (wrk_e > axin->e)
        wrk_e -= wrk_d;
    *o_ = wrk_o;
    *e_ = wrk_e;
    *d_ = wrk_d;

    n = ceil((wrk_e - wrk_o) / wrk_d - .5) + 1;
    ntot = n;
    o = wrk_o;
    d = wrk_d;
    opad = o;
    e = ((n - 1) * d) + o;
    epad = e;
    invd = 1. / d;
    nghost = 0;
    ibeg = 0;
    iend = n - 1;

    alignMnt = axin->alignMnt;
    nvalid = n + (2 * nghost);
    npad_trailing = getPadTrailing(alignMnt.nElems, nvalid);
    nallocated_elements = nvalid + npad_trailing;
    epad = e + (nghost * d) + npad_trailing * d;
    ntot += npad_trailing;
}

// "set range based on both"  ??? what does that mean ???
//! constructs an axis.. grids will be on o,e,d system not from axin set range based on both
axis ::axis(realtype* o_, realtype* e_, realtype* d_, axis* axin)
{
    realtype wrk_o = *o_;
    realtype wrk_e = *e_;
    realtype wrk_d = *d_;

    wrk_o = std::max(wrk_o, axin->o);
    int itmp = axin->get_index(wrk_o);
    wrk_o = axin->get_val(itmp);

    itmp = ceil(wrk_d / axin->d - .5);
    wrk_d = itmp * axin->d;

    // if wrk_o is not on grids on *o,*e,*d
    if ((int)((wrk_o - *o_) / wrk_d) * wrk_d != (wrk_o - *o_)) {
        wrk_o = *o_ + (int)((wrk_o - *o_) / wrk_d + 1) * wrk_d;
    }

    wrk_e = std::min(wrk_e, axin->e);
    itmp = std::ceil((wrk_e - wrk_o) / wrk_d - .5);
    wrk_e = wrk_o + itmp * wrk_d;

    if (wrk_e > axin->e)
        wrk_e -= wrk_d;

    *o_ = wrk_o;
    *e_ = wrk_e;
    *d_ = wrk_d;

    n = ceil((wrk_e - wrk_o) / wrk_d - .5) + 1;
    ntot = n;
    o = wrk_o;
    d = wrk_d;
    opad = o;
    e = ((n - 1) * d) + o;
    invd = 1. / d;
    nghost = 0;
    ibeg = 0;
    iend = n - 1;

    alignMnt = axin->alignMnt;
    nvalid = n + (2 * nghost);
    npad_trailing = getPadTrailing(alignMnt.nElems, nvalid);
    nallocated_elements = nvalid + npad_trailing;
    epad = e + (nghost * d) + npad_trailing * d;
    ntot += npad_trailing;
}

axis ::~axis()
{}

void
axis::scale(realtype sc)
{
    o *= sc;
    opad *= sc;
    d *= sc;
    e = ((n - 1) * d) + o;
    epad = ((ntot - 1) * d) + opad;
}

void
axis ::reverse()
{
    realtype tmp;
    tmp = opad;
    opad = epad;
    epad = tmp;
    tmp = o;
    o = e;
    e = tmp;
    d *= -1.0f;
    invd *= -1.0f;
}

void
axis::shift(realtype shift)
{
    o += shift;
    opad += shift;
    e += shift;
    epad += shift;
}

void
axis::print(std::string message)
{
    std::cout << message << std::endl;
    std::cout << "ntot " << ntot << std::endl;
    std::cout << "opad " << opad << " epad " << epad << std::endl;
    std::cout << "ibeg " << ibeg << " iend " << iend << std::endl;
}
void
axis ::print(FILE* fd)
{
    fprintf(fd, "*** begin axis ***\n");
    fprintf(fd, "* axis: %s\n", label.c_str());
    fprintf(fd, "* unit: %s\n", unit.c_str());
    fprintf(fd, "* origin: %16.8e, end: %16.8e\n", o, e);
    fprintf(fd, "* span: %16.8e\n", d * (n - 1));
    fprintf(fd, "* delta: %16.8e, n: %d\n", d, n);
    if (nghost) {
        fprintf(fd, "* nghost: %d, ntot: %d\n", nghost, ntot);
    }
    fprintf(fd, "**** end axis ****\n");
}

void
axis ::print(FILE* fd, std::string axis_label)
{
    int nstar = -1;
    if (!(axis_label.empty())) {
        nstar = (39 - axis_label.length()) / 2;
        nstar = std::max(1, nstar);
        for (int i = 0; i < nstar; i++) {
            fprintf(fd, "*");
        }
        fprintf(fd, " %s ", (axis_label.c_str()));
        for (int i = 0; i < nstar; i++) {
            fprintf(fd, "*");
        }
        fprintf(fd, "\n");
    } else {
        fprintf(fd, "***************************************\n");
    }
    fprintf(fd, "* axis: %s\n", label.c_str());
    fprintf(fd, "* unit: %s\n", unit.c_str());
    fprintf(fd, "* origin: %16.8e, end: %16.8e\n", o, e);
    fprintf(fd, "* span: %16.8e\n", d * (n - 1));
    fprintf(fd, "* delta: %16.8e, n: %d\n", d, n);
    if (nghost > 0) {
        fprintf(fd, "* nghost: %d, ntot: %d\n", nghost, ntot);
    }
    if (!(axis_label.empty())) {
        for (size_t i = 0; i < (axis_label.length() + 2 + 2 * nstar); i++) {
            fprintf(fd, "*");
        }
        fprintf(fd, "\n");
    } else {
        fprintf(fd, "***************************************\n");
    }
}

void
axis ::set_label(std::string lbl)
{
    label = lbl;
}

void
axis ::set_unit(std::string un)
{
    unit = un;
}

int
axis ::get_interp(int* ind1, realtype* val1, realtype* val2, realtype x)
{
    int index, status;
    realtype v1, v2;
    index = ceil((x - opad) / d - .5);
    status = 1;
    if (index < -1 || index >= ntot) {
        *ind1 = 0;
        *val1 = 0.f;
        *val2 = 0.f;
        status = 0;
    } else if (index == -1) {
        *ind1 = 0;
        v1 = opad + index * d;
        *val1 = (x - v1) / d;
        *val2 = 0.f;
    } else if (index == ntot - 1) {
        *ind1 = ntot - 2;
        *val1 = 0.f;
        v2 = opad + (index + 1) * d;
        *val2 = (v2 - x) / d;
    } else {
        *ind1 = index;
        v1 = opad + index * d;
        v2 = opad + (index + 1) * d;
        *val1 = (v2 - x) / d;
        *val2 = (x - v1) / d;
    }
    return status;
}

int
axis ::get_index(realtype x) const
{
    int index = (int)(std::ceil(((x - opad) / d) - 0.5f));
    if (index < 0 || index >= ntot)
        index = -1;
    return index;
}

int
axis ::get_index_round(realtype x)
{
    int index = (int)(std::floor(((x - opad) / d) + 0.5f));
    if (index < 0)
        index = 0;
    else if (index >= ntot)
        index = ntot - 1;
    return index;
}

int
axis ::get_index_interior(realtype x) const
{
    int index = (int)(std::ceil(((x - o) / d) - 0.5f));
    if (index < 0 || index >= n)
        index = -1;
    else
        index += nghost;
    return index;
}

int
axis ::get_index_floor(double x)
{
    int index = (int)(std::floor((x - ((double)opad)) / ((double)d)));
    if (index < 0)
        index = -1;
    else if (index >= ntot)
        index = -2;
    return index;
}

int
axis ::get_index_ceil(double x)
{
    int index = (int)(std::ceil((x - ((double)opad)) / ((double)d)));
    if (index < 0)
        index = -1;
    else if (index >= ntot)
        index = -2;
    return index;
}

int
axis ::get_lower_index(realtype x)
{
    int index = (int)(std::floor((x - opad) / d));
    if (index < 0)
        index = 0;
    else if (index >= ntot)
        index = ntot - 1;
    return index;
}

int
axis ::get_upper_index(realtype x)
{
    int index = (int)(std::ceil((x - opad) / d));
    if (index < 0)
        index = 0;
    else if (index >= ntot)
        index = ntot - 1;
    return index;
}

int
axis ::get_copy_indices(axis* ax, int* axbeg, int* mybeg, int* myend)
{
    int status = 0;
    if (ax->d == d) {
        if (opad <= ax->epad && epad >= ax->opad) {
            if (opad >= ax->opad) {
                *axbeg = ax->get_index(opad);
                *mybeg = 0;
            } else {
                *mybeg = get_index(ax->opad);
                *axbeg = 0;
            }
            if (epad <= ax->epad) {
                *myend = ntot - 1;
            } else {
                *myend = get_index(ax->epad);
            }
            status = 1;
        }
    }
    return status;
}

int
axis ::get_interior_copy_indices(axis* ax, int* axbeg, int* mybeg, int* myend)
{
    int status = 0;
    if (ax->d == d) {
        if (o <= ax->epad && e >= ax->opad) {
            if (o >= ax->o) {
                *axbeg = ax->get_index(o);
                *mybeg = ibeg;
            } else {
                *mybeg = get_index(ax->o);
                *axbeg = ax->ibeg;
            }
            if (e <= ax->e) {
                *myend = iend;
            } else {
                *myend = get_index(ax->e);
            }
            status = 1;
        }
    }
    return status;
}

void
axis ::get_interp_indices(axis* ax, int* axbeg, int* axend, int* axn)
{
    *axbeg = ax->get_lower_index(opad);
    *axend = ax->get_upper_index(epad);
    *axn = *axend - *axbeg + 1;
}

void
axis ::computeIdxWtsLinearInterp(int& i, double& w1, double& w2, const realtype p) const
{
    //make sure that hte point is in (o-d, e+d)
    this->inbounds_ext_shared(p);

    if (p < o) {
        i = ibeg - 1;
    } else {
        i = ibeg +
            static_cast<int>(std::floor((static_cast<double>(p) - static_cast<double>(o)) / static_cast<double>(d)));
    }
    EMSL_VERIFY(i >= (ibeg - 1));
    EMSL_VERIFY(i <= iend);
    w2 = (static_cast<double>(p) - static_cast<double>(o) - (static_cast<double>(i - ibeg) * static_cast<double>(d))) /
         static_cast<double>(d);
    w1 = 1.0 - w2;
}

/*
float
axis ::get_val_cosine_interp(int i)
{

    return 0;
}

float
axis ::get_val_cubic_interp(int i)
{

    return 0;
}

float
axis ::get_val_catmull_rom_interp(int i)
{

    return 0;
}

float
axis ::get_val_hermite_interp(int i)
{

    return 0;
}
*/

bool
operator==(const axis& lhs, const axis& rhs)
{
    return (lhs.o == rhs.o && lhs.e == rhs.e && lhs.opad == rhs.opad && lhs.epad == rhs.epad && lhs.ntot == rhs.ntot &&
            lhs.d == rhs.d && lhs.invd == rhs.invd && lhs.ibeg == rhs.ibeg && lhs.iend == rhs.iend &&
            lhs.nghost == rhs.nghost);
}
