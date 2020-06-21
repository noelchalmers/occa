#include <occa/modes/hip/memory.hpp>
#include <occa/modes/hip/device.hpp>
#include <occa/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    inline hipDeviceptr_t addHipPtrOffset(hipDeviceptr_t hipPtr, const udim_t offset) {
      return (hipDeviceptr_t) (((char*) hipPtr) + offset);
    }

    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_),
#ifdef __HIP_PLATFORM_HCC__
      hipPtr(ptr)  {}
#else
      hipPtr((hipDeviceptr_t&) ptr) {}
#endif

    memory::~memory() {
      if (isOrigin) {
        if (useHostPtr) {
          OCCA_HIP_ERROR("Device: mappedFree()",
                         hipHostFree(ptr));
        } else if (hipPtr) {
          hipFree((void*) hipPtr);
        }
      }
      ptr = nullptr;
      hipPtr = 0;
      size = 0;
    }

    hipStream_t& memory::getHipStream() const {
      return ((device*) modeDevice)->getHipStream();
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      if (useHostPtr) {
        arg.data.void_ = ptr;
      } else {
        arg.data.void_ = (void*) hipPtr;
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
        m->hipPtr = addHipPtrOffset(hipPtr, offset);
      }
      m->useHostPtr = useHostPtr;
      return m;
    }

    void* memory::getPtr() {
      if (useHostPtr) {
        return ptr;
      } else {
        return hipPtr;
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyHtoD(addHipPtrOffset(hipPtr, offset),
                                       const_cast<void*>(src),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyHtoDAsync(addHipPtrOffset(hipPtr, offset),
                                            const_cast<void*>(src),
                                            bytes,
                                            getHipStream()) );
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyHtoD(addHipPtrOffset(hipPtr, destOffset),
                                       src->ptr + srcOffset,
                                       bytes));
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyHtoDAsync(addHipPtrOffset(hipPtr, destOffset),
                                            src->ptr + srcOffset,
                                            bytes,
                                            getHipStream()));
        }
      } else if (useHostPtr) {
        if (!async) {
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoH(ptr + destOffset,
                                       addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                       bytes));
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoHAsync(ptr + destOffset,
                                            addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                            bytes,
                                            getHipStream()));
        }
      } else {
        if (!async) {
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoD(addHipPtrOffset(hipPtr, destOffset),
                                       addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoDAsync(addHipPtrOffset(hipPtr, destOffset),
                                            addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                            bytes,
                                            getHipStream()) );
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoH(dest,
                                       addHipPtrOffset(hipPtr, offset),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoHAsync(dest,
                                            addHipPtrOffset(hipPtr, offset),
                                            bytes,
                                            getHipStream()) );
        }
      }
    }

    void memory::detach() {
      hipPtr = 0;
      size = 0;
    }
  }
}
