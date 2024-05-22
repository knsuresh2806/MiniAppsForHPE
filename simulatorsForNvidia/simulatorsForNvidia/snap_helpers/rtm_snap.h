#ifndef RTM_SNAP
#define RTM_SNAP

#include <vector>
#include <zfp.h>

#include "axis.h"
#include "std_const.h"

class file_snap;
template <class T>
class cart_volume;

class rtm_snap
{
public:
    rtm_snap() = default;

    virtual void start_read_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& corrBuffList,
                                                      std::vector<int>& corr_buff_size, file_snap* fd_snap_p) = 0;

    virtual void finish_read_uncompressed_fwi_snapshot() = 0;

    virtual void start_read_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& corrBuffList,
                                                    std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                                    std::vector<realtype*>& zipped_corr_buff,
                                                    std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream) = 0;

    virtual void finish_read_compressed_fwi_snapshot() = 0;

    virtual void start_write_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& corrBuffList,
                                                       std::vector<int>& corr_buff_size, file_snap* fd_snap_p) = 0;

    virtual void lock_for_write_uncompressed_fwi_snapshot() = 0;

    virtual void start_write_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& corrBuffList,
                                                     std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                                     std::vector<realtype*>& zipped_corr_buff,
                                                     std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream) = 0;

    virtual void lock_for_write_compressed_fwi_snapshot() = 0;

    virtual ~rtm_snap() = default;
};

#endif // RTM_SNAP
