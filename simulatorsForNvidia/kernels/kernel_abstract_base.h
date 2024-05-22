#ifndef KERNEL_ABSTRACT_BASE_H
#define KERNEL_ABSTRACT_BASE_H

#include <string>
#include <vector>

#include "pml_volumes_cg3d.h"
#include "pml_coef3d.h"
#include "sponge_coef3d.h"

template <class T>
class cart_volume;

class kernel_abstract_base
{
public:
    static const int NMECH;

    kernel_abstract_base(int order, int bc_opt);

    virtual ~kernel_abstract_base() = default;

    void kernel_loop2(int isub);
    void kernel_loop1(int isub);
    void kernel_loop3(int isub);
    void adj_kernel_loop1(int isub);
    void adj_kernel_loop3(int isub);
    void kernel_loop4(int isub);

private:
    /** Ensure that any volumes that were passed to the constructor have been cast to the appropriate type */
    virtual void ensure_volumes() = 0;

    virtual void kernel_loop2_inner(int isub) = 0;
    virtual void kernel_loop1_inner(int isub) = 0;
    virtual void kernel_loop3_inner(int isub) = 0;
    virtual void adj_kernel_loop1_inner(int isub) = 0;
    virtual void adj_kernel_loop3_inner(int isub) = 0;
    virtual void kernel_loop4_inner(int isub) = 0;

private:
    int order;
    int bc_opt;
};

inline void
kernel_abstract_base::kernel_loop2(int isub)
{
    ensure_volumes();
    kernel_loop2_inner(isub);
}

inline void
kernel_abstract_base::kernel_loop1(int isub)
{
    ensure_volumes();
    kernel_loop1_inner(isub);
}

inline void
kernel_abstract_base::kernel_loop3(int isub)
{
    ensure_volumes();
    kernel_loop3_inner(isub);
}

inline void
kernel_abstract_base::adj_kernel_loop1(int isub)
{
    ensure_volumes();
    adj_kernel_loop1_inner(isub);
}

inline void
kernel_abstract_base::adj_kernel_loop3(int isub)
{
    ensure_volumes();
    adj_kernel_loop3_inner(isub);
}

inline void
kernel_abstract_base::kernel_loop4(int isub)
{
    ensure_volumes();
    kernel_loop4_inner(isub);
}

#endif
