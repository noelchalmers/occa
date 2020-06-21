#include <occa/modes/cuda/memory.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        cuPtr((CUdeviceptr&) ptr),
        isUnified(false) {}

    memory::~memory() {
      if (isOrigin) {
        if (useHostPtr) {
          OCCA_CUDA_ERROR("Device: mappedFree()",
                          cuMemFreeHost(ptr));
        } else if (cuPtr) {
          cuMemFree(cuPtr);
        }
      }
      ptr = nullptr;
      cuPtr = 0;
      size = 0;
    }

    CUstream& memory::getCuStream() const {
      return ((device*) modeDevice)->getCuStream();
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      if (useHostPtr) {
        arg.data.void_ = (void*) &ptr;
      } else {
        arg.data.void_ = (void*) &cuPtr;
      }
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      if (useHostPtr) {
        m->ptr = ptr + offset;
      } else {
        m->cuPtr = cuPtr + offset;
      }
      m->useHostPtr = useHostPtr;
      m->isUnified = isUnified;
      return m;
    }

    void* memory::getPtr() {
      if (useHostPtr) {
        return ptr;
      } else {
        return (void*) cuPtr;
      }
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(ptr+offset, src, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyHtoD(cuPtr + offset,
                                       src,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyHtoDAsync(cuPtr + offset,
                                            src,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);

      if (useHostPtr && src->useHostPtr) {
        ::memcpy(ptr + destOffset, src->ptr + srcOffset, bytes);
      } else if (src->useHostPtr) {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyHtoD(cuPtr + destOffset,
                                       src->ptr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyHtoDAsync(cuPtr + destOffset,
                                            src->ptr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      } else if (useHostPtr) {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoH(ptr + destOffset,
                                       ((memory*) src)->cuPtr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoHAsync(ptr + destOffset,
                                            ((memory*) src)->cuPtr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoD(cuPtr + destOffset,
                                       ((memory*) src)->cuPtr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoDAsync(cuPtr + destOffset,
                                            ((memory*) src)->cuPtr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(dest, ptr+offset, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoH(dest,
                                       cuPtr + offset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoHAsync(dest,
                                            cuPtr + offset,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::detach() {
      cuPtr = 0;
      size = 0;
    }
  }
}
