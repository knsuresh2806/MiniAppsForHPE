#ifndef _CART_VOLUME_REGULAR_HOST
#define _CART_VOLUME_REGULAR_HOST

// #include "cart_volume.h"
#include "cart_volume_regular.h"
#include "cart_volume_regular_gpu.h"
#include "axis.h"
#include "volume_index.h"
class cart_volume_regular_gpu;

class cart_volume_regular_host : public cart_volume_regular
{
public:
    /*! Constructor 
     * \param ax1_ 1st dimension axis
     * \param ax2_ 2nd dimension axis
     * \param ax3_ 3rd dimension axis
     * \param set_to_zero - true to set all values to zero (false by default)
     */
    cart_volume_regular_host(const axis* ax1_, const axis* ax2_, const axis* ax3_, bool set_to_zero = false);

    // FIXME This constructor should not accept a pointer to a cart volume.  Accepting a pointer
    // implicitly creates a contract that a null pointer can be handled.  Crashing is not
    // handling a pointer. Additionally, it should not be a writable pointer.

    /*! Copy constructor with option to leave off halo 
     * \param vol volume to be copied
     * \param skipHalos - true to skip halos (false by default)
     * \param skipInterleaving - true to only allocate for one volume (false by default)
     * \param set_to_zero - true to set all values to zero (false by default)
     */
    cart_volume_regular_host(cart_volume_regular* vol, bool skipHalos = false, bool skipInterleaving = false,
                             bool set_to_zero = false);

    // FIXME This constructor should not accept a pointer to a cart volume.  Accepting a pointer
    // implicitly creates a contract that a null pointer can be handled.  Crashing is not
    // handling a pointer.
    //Copy constructor with axis rotation
    cart_volume_regular_host(cart_volume_regular* vol, ROTATE_TYPE type);

    // Destructor
    ~cart_volume_regular_host();

    //! set all values to a constant
    void set_constant(float val, bool skipHalos = true, bool skipInterleaving = false);

    //! copy data from argument volume (used for checkpointing)
    void copyData(cart_volume<float>* vol, bool skipHalos = true, bool skipInterleaving = false);

    volume_index vol_idx();

    // As function to access the variables
    float* data() const { return this->_data; }
    float*** data3d() const { return this->_data3d; }

    void getDims(int& n0, int& n1, int& n2, bool skipHalosAndPadding = false);

    size_t getSize() const { return data_size; }

private:
    cart_volume_regular_host() {}

    //! copy data from argument volume (used for checkpointing)
    void copyFrom(cart_volume_regular_gpu* from, bool skipHalos = true);

    void allocate_data();

    float* _data;
    float*** _data3d;

    size_t data_size;

    void copyAxis(const axis* ax1_, const axis* ax2_, const axis* ax3_);
};
#endif //_CART_VOLUME_REGULAR_HOST
