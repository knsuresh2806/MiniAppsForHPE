#include "rtm_snap_gpu.h"
#include "cart_volume_regular_gpu.h"
#include "file_snap.h"
#include "emsl_error.h"
#include "timer.h"

rtm_snap_gpu::rtm_snap_gpu(std::vector<cart_volume<realtype>*>& corrBuffList, cudaEvent_t writeSnapshotsCompleteEvent,
                           cudaEvent_t readSnapshotsCompleteEvent, bool useZFP)
    : rtm_snap(), cpu_corrBuffList(corrBuffList.size(), nullptr),
      helper_thread{ std::bind(
          useZFP ? &rtm_snap_gpu::helper_thread_loop_compressed : &rtm_snap_gpu::helper_thread_loop, this) },
      writeSnapshotsCompleteEvent_(writeSnapshotsCompleteEvent), readSnapshotsCompleteEvent_(readSnapshotsCompleteEvent)
{
    // create temporary volumes in cpu
    for (int dom = 0; dom < cpu_corrBuffList.size(); ++dom) {
        if (!cpu_corrBuffList[dom]) {
            axis cpu_xaxis{ corrBuffList[dom]->as<cart_volume_regular>()->ax1(),
                            AlignmentElem(AlignMemBytes::NOEXTENSION, 1) };
            cpu_corrBuffList[dom] =
                new cart_volume_regular_host(&cpu_xaxis, corrBuffList[dom]->as<cart_volume_regular>()->ax2(),
                                             corrBuffList[dom]->as<cart_volume_regular>()->ax3());
        }
    }

    // create stream which isn't blocking with the main thread default stream
    CUDA_TRY(cudaStreamCreateWithFlags(&helperStream, cudaStreamNonBlocking));
    // Use default stream for zfp
    zfpCudaStream = 0;
}

void
rtm_snap_gpu::start_read_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                                   std::vector<int>& corr_buff_size, file_snap* fd_snap_p)
{
    EMSL_VERIFY(gpu_corrBuffList.size() == cpu_corrBuffList.size());

    sync.main_thread_wait();

    // the helper thread reads these variables, so only set them between main_thread_wait and release
    this->gpu_corrBuffList = gpu_corrBuffList;
    this->corrBuffSize = corr_buff_size;
    this->command = snapshot_command::read;
    this->fd_snap_p = fd_snap_p;

    sync.main_thread_release();
}

void
rtm_snap_gpu::finish_read_uncompressed_fwi_snapshot()
{
    sync.main_thread_wait();
}

void
rtm_snap_gpu::start_read_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                                 std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                                 std::vector<realtype*>& zipped_corr_buff,
                                                 std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream)
{
    EMSL_VERIFY(gpu_corrBuffList.size() == cpu_corrBuffList.size());

    sync.main_thread_wait();

    // the helper thread reads these variables, so only set them between main_thread_wait and release
    this->gpu_corrBuffList = gpu_corrBuffList;
    this->zfpFields = zfpFields;
    this->zfpStream = zfpStream;
    this->zipped_corr_buff = zipped_corr_buff;
    this->corrBuffSize = corr_buff_size;
    this->command = snapshot_command::read;
    this->fd_snap_p = fd_snap_p;

    sync.main_thread_release();
}

void
rtm_snap_gpu::finish_read_compressed_fwi_snapshot()
{
    sync.main_thread_wait();
}

void
rtm_snap_gpu::start_write_uncompressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                                    std::vector<int>& corr_buff_size, file_snap* fd_snap_p)
{
    EMSL_VERIFY(gpu_corrBuffList.size() == cpu_corrBuffList.size());

    // the helper thread reads these variables, so only set them between main_thread_wait and release
    this->gpu_corrBuffList = gpu_corrBuffList;
    this->corrBuffSize = corr_buff_size;
    this->command = snapshot_command::write;
    this->fd_snap_p = fd_snap_p;

    sync.main_thread_release();
}

void
rtm_snap_gpu::lock_for_write_uncompressed_fwi_snapshot()
{
    sync.main_thread_wait();
}

void
rtm_snap_gpu::start_write_compressed_fwi_snapshot(std::vector<cart_volume<realtype>*>& gpu_corrBuffList,
                                                  std::vector<int>& corr_buff_size, file_snap* fd_snap_p,
                                                  std::vector<realtype*>& zipped_corr_buff,
                                                  std::vector<zfp_field*>& zfpFields, zfp_stream* zfpStream)
{
    EMSL_VERIFY(gpu_corrBuffList.size() == cpu_corrBuffList.size());

    // the helper thread reads these variables, so only set them between main_thread_wait and release
    this->gpu_corrBuffList = gpu_corrBuffList;
    this->zfpFields = zfpFields;
    this->zipped_corr_buff = zipped_corr_buff;
    this->zfpStream = zfpStream;
    this->corrBuffSize = corr_buff_size;
    this->command = snapshot_command::write;
    this->fd_snap_p = fd_snap_p;

    sync.main_thread_release();
}

void
rtm_snap_gpu::lock_for_write_compressed_fwi_snapshot()
{
    sync.main_thread_wait();
}

/*
 NOTE: this executes in a different thread, and is controlled through the
 command and sync variables.
*/
void
rtm_snap_gpu::helper_thread_loop()
{
    while (true) {
        sync.helper_thread_wait();

        switch (command) {
            case snapshot_command::close:
                sync.helper_thread_release();
                return;
            case snapshot_command::write:
                helper_thread_write();
                break;
            case snapshot_command::read:
                helper_thread_read();
                break;
            default:
                EMSL_ERROR("got unexpected command in helper_thread")
        }

        sync.helper_thread_release();
    }
}

/*
 NOTE: this executes in a different thread, and is controlled through the
 command and sync variables.
*/
void
rtm_snap_gpu::helper_thread_loop_compressed()
{
    while (true) {
        sync.helper_thread_wait();

        switch (command) {
            case snapshot_command::close:
                sync.helper_thread_release();
                return;
            case snapshot_command::write:
                helper_thread_write_compressed();
                break;
            case snapshot_command::read:
                helper_thread_read_compressed();
                break;
            default:
                EMSL_ERROR("got unexpected command in helper_thread")
        }

        sync.helper_thread_release();
    }
}

void
rtm_snap_gpu::helper_thread_write()
{
    cudaStreamWaitEvent(helperStream, writeSnapshotsCompleteEvent_);
    // copy the snapshot data to temporary cpu volumes from original gpu volumes
    for (int dom = 0; dom < cpu_corrBuffList.size(); ++dom) {
        gpu_corrBuffList[dom]->as<cart_volume_regular_gpu>()->copyToAsync(
            cpu_corrBuffList[dom]->as<cart_volume_regular_host>(), helperStream, false);
    }
    CUDA_TRY(cudaStreamSynchronize(helperStream));

    int nbuf = cpu_corrBuffList.size();
    if (nbuf > 0) {
        std::vector<realtype*> data_vec;
        for (int i = 0; i < nbuf; ++i) {
            data_vec.push_back(cpu_corrBuffList[i]->as<cart_volume_regular_host>()->data());
        }
        fd_snap_p->write_step(data_vec, corrBuffSize);
    }
}

void
rtm_snap_gpu::helper_thread_read()
{
    cudaStreamWaitEvent(helperStream, readSnapshotsCompleteEvent_);
    int nbuf = cpu_corrBuffList.size();
    if (nbuf > 0) {
        std::vector<realtype*> data_vec;
        for (int i = 0; i < nbuf; ++i) {
            cart_volume<realtype>* snap = cpu_corrBuffList[i];
            data_vec.push_back(snap->as<cart_volume_regular_host>()->data());
        }
        fd_snap_p->read_step(data_vec, corrBuffSize);
    }

    // copy the snapshot data from temporary cpu volumes to original gpu volumes
    for (int dom = 0; dom < cpu_corrBuffList.size(); ++dom) {
        gpu_corrBuffList[dom]->as<cart_volume_regular_gpu>()->copyFromAsync(
            cpu_corrBuffList[dom]->as<cart_volume_regular_host>(), helperStream, false);
    }
    CUDA_TRY(cudaStreamSynchronize(helperStream));
}

void
rtm_snap_gpu::helper_thread_write_compressed()
{
    cudaStreamWaitEvent(zfpCudaStream, writeSnapshotsCompleteEvent_);
    int nbuf = gpu_corrBuffList.size();
    if (nbuf > 0) {
        // Compress from corrBuffList to zipped_corr_buff
        for (int i = 0; i < nbuf; ++i) {
            // Input
            zfp_field_set_pointer(zfpFields[i], gpu_corrBuffList[i]->as<cart_volume_regular_gpu>()->getData());

            // Output
            bitstream* bstream = stream_open(zipped_corr_buff[i], corrBuffSize[i] * sizeof(realtype));
            zfp_stream_set_bit_stream(zfpStream, bstream);

            zfp_stream_rewind(zfpStream);
            zfp_compress(zfpStream, zfpFields[i]);

            // Clean up
            stream_close(bstream);
        }
        CUDA_TRY(cudaStreamSynchronize(zfpCudaStream));
        fd_snap_p->write_step(zipped_corr_buff, corrBuffSize);
    }
}

void
rtm_snap_gpu::helper_thread_read_compressed()
{
    cudaStreamWaitEvent(zfpCudaStream, readSnapshotsCompleteEvent_);

    int nbuf = gpu_corrBuffList.size();
    if (nbuf > 0) {
        fd_snap_p->read_step(zipped_corr_buff, corrBuffSize);

        // Decompress from zipped_corr_buff to corrBuffList
        for (int i = 0; i < nbuf; ++i) {
            // Ouptut
            zfp_field_set_pointer(zfpFields[i], gpu_corrBuffList[i]->as<cart_volume_regular_gpu>()->getData());

            // Input
            bitstream* bstream = stream_open(zipped_corr_buff[i], corrBuffSize[i] * sizeof(realtype));
            zfp_stream_set_bit_stream(zfpStream, bstream);
            zfp_stream_rewind(zfpStream);

            zfp_decompress(zfpStream, zfpFields[i]);

            // Clean up
            stream_close(bstream);
        }
    }
    CUDA_TRY(cudaStreamSynchronize(zfpCudaStream));
}
