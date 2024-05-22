#ifndef RTM_SNAP_GPU
#define RTM_SNAP_GPU

class rtm_snap_gpu;

#include "rtm_snap.h"
#include "std_utils.h"
#include "cuda_utils.h"
#include "helper_thread_synchronizer.h"
#include "cart_volume_regular_host.h"

#include <thread>

class file_snap;

class rtm_snap_gpu : public rtm_snap
{
public:
    rtm_snap_gpu(std::vector<cart_volume<realtype>*>& corrBuffList, cudaEvent_t writeSnapshotsCompleteEvent,
                 cudaEvent_t readSnapshotsCompleteEvent, bool useZFP);

    ~rtm_snap_gpu()
    {
        deleteList(cpu_corrBuffList);

        command = snapshot_command::close;
        sync.main_thread_release();
        helper_thread.join();
    }

    void start_read_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                              std::vector<int>& corr_buff_size, file_snap* fd_snap_p) override;

    void finish_read_uncompressed_fwi_snapshot() override;

    void start_read_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                            std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                            std::vector<realtype*>& zipped_corr_buff,
                                            std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream) override;

    void finish_read_compressed_fwi_snapshot() override;

    void start_write_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                               std::vector<int>& corr_buff_size, file_snap* fd_snap_p) override;

    void lock_for_write_uncompressed_fwi_snapshot() override;

    void start_write_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                             std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                             std::vector<realtype*>& zipped_corr_buff,
                                             std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream) override;

    void lock_for_write_compressed_fwi_snapshot() override;

private:
    cudaStream_t helperStream;
    cudaStream_t zfpCudaStream;
    cudaEvent_t writeSnapshotsCompleteEvent_;
    cudaEvent_t readSnapshotsCompleteEvent_;
    void helper_thread_loop();
    void helper_thread_read();
    void helper_thread_write();
    void helper_thread_loop_compressed();
    void helper_thread_read_compressed();
    void helper_thread_write_compressed();

    helper_thread_synchronizer sync;
    std::thread helper_thread;

    // The following variables are set by the main thread and read by the helper thread.
    // Access is controlled by the sync object.
    enum class snapshot_command
    {
        write,
        read,
        close
    } command;
    file_snap* fd_snap_p;
    std::vector<int> corrBuffSize;
    std::vector<cart_volume<realtype>*> gpu_corrBuffList;
    std::vector<cart_volume<realtype>*> cpu_corrBuffList;
    std::vector<zfp_field*> zfpFields;
    std::vector<float*> zipped_corr_buff;
    zfp_stream* zfpStream;
};

#endif // RTM_SNAP_GPU
