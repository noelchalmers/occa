#include "occa.hpp"

// Use events for timing!

namespace occa {
  //---[ Helper Classes ]-------------
  kernelInfo defaultKernelInfo;

  const char* deviceInfo::header = "| Name                                      | Num | Available Modes                  |";
  const char* deviceInfo::sLine  = "+-------------------------------------------+-----+----------------------------------+";

  const char* deviceInfo::dLine1 = "+--------------------------------------------------------+";
  const char* deviceInfo::dLine2 = "+  -  -  -  -  -  +  -  -  -  -  -  -  -  -  -  -  -  -  +";

  const int uint8FormatIndex  = 0;
  const int uint16FormatIndex = 1;
  const int uint32FormatIndex = 2;
  const int int8FormatIndex   = 3;
  const int int16FormatIndex  = 4;
  const int int32FormatIndex  = 5;
  const int halfFormatIndex   = 6;
  const int floatFormatIndex  = 7;

  const int sizeOfFormats[8] = {1, 2, 4,
                                1, 2, 4,
                                2, 4};

  formatType::formatType(const int format__, const int count__){
    format_ = format__;
    count_  = count__;
  }

  formatType::formatType(const formatType &ft){
    format_ = ft.format_;
    count_  = ft.count_;
  }

  formatType& formatType::operator = (const formatType &ft){
    format_ = ft.format_;
    count_  = ft.count_;

    return *this;
  }

  int formatType::count() const {
    return count_;
  }

  size_t formatType::bytes() const {
    return (sizeOfFormats[format_] * count_);
  }

  const int readOnly  = 1;
  const int readWrite = 2;

  const occa::formatType uint8Format(uint8FormatIndex  , 1);
  const occa::formatType uint8x2Format(uint8FormatIndex, 2);
  const occa::formatType uint8x4Format(uint8FormatIndex, 4);

  const occa::formatType uint16Format(uint16FormatIndex  , 1);
  const occa::formatType uint16x2Format(uint16FormatIndex, 2);
  const occa::formatType uint16x4Format(uint16FormatIndex, 4);

  const occa::formatType uint32Format(uint32FormatIndex  , 1);
  const occa::formatType uint32x2Format(uint32FormatIndex, 2);
  const occa::formatType uint32x4Format(uint32FormatIndex, 4);

  const occa::formatType int8Format(int8FormatIndex  , 1);
  const occa::formatType int8x2Format(int8FormatIndex, 2);
  const occa::formatType int8x4Format(int8FormatIndex, 4);

  const occa::formatType int16Format(int16FormatIndex  , 1);
  const occa::formatType int16x2Format(int16FormatIndex, 2);
  const occa::formatType int16x4Format(int16FormatIndex, 4);

  const occa::formatType int32Format(int32FormatIndex  , 1);
  const occa::formatType int32x2Format(int32FormatIndex, 2);
  const occa::formatType int32x4Format(int32FormatIndex, 4);

  const occa::formatType halfFormat(halfFormatIndex  , 1);
  const occa::formatType halfx2Format(halfFormatIndex, 2);
  const occa::formatType halfx4Format(halfFormatIndex, 4);

  const occa::formatType floatFormat(floatFormatIndex  , 1);
  const occa::formatType floatx2Format(floatFormatIndex, 2);
  const occa::formatType floatx4Format(floatFormatIndex, 4);
  //==================================

  //---[ Kernel ]---------------------
  kernel::kernel() :
    mode_(),
    strMode(""),

    kHandle(NULL) {}

  kernel::kernel(const kernel &k) :
    mode_(k.mode_),
    strMode(k.strMode),

    kHandle(k.kHandle) {}

  kernel& kernel::operator = (const kernel &k){
    mode_   = k.mode_;
    strMode = k.strMode;

    kHandle = k.kHandle;

    return *this;
  }

  std::string& kernel::mode(){
    return strMode;
  }

  kernel& kernel::buildFromSource(const std::string &filename,
                                  const std::string &functionName_,
                                  const kernelInfo &info_){
    kHandle->buildFromSource(filename, functionName_, info_);

    return *this;
  }

  kernel& kernel::buildFromBinary(const std::string &filename,
                                  const std::string &functionName_){
    kHandle->buildFromBinary(filename, functionName_);

    return *this;
  }

  kernel& kernel::loadFromLibrary(const char *cache,
                                  const std::string &functionName_){
    kHandle->loadFromLibrary(cache, functionName_);

    return *this;
  }

  void kernel::setWorkingDims(int dims, occa::dim inner, occa::dim outer){
    for(int i = 0; i < dims; ++i){
      inner[i] += (inner[i] ? 0 : 1);
      outer[i] += (outer[i] ? 0 : 1);
    }

    for(int i = dims; i < 3; ++i)
      inner[i] = outer[i] = 1;

    if(kHandle->nestedKernelCount == 0){
      kHandle->dims  = dims;
      kHandle->inner = inner;
      kHandle->outer = outer;
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k)
        kHandle->nestedKernels[k].setWorkingDims(dims, inner, outer);
    }
  }

  int kernel::preferredDimSize(){
    if(kHandle->nestedKernelCount == 0){
      return kHandle->preferredDimSize();
    }
    else{
      std::cout << "Cannot get preferred size for fused kernels\n";
      throw 1;
    }

    return 1;
  }

  void kernel::clearArgumentList(){
    argumentCount = 0;
  }

  void kernel::addArgument(const int argPos,
                           const kernelArg &arg){
    if(argumentCount < (argPos + 1)){
      OCCA_CHECK(argPos < OCCA_MAX_ARGS);

      argumentCount = (argPos + 1);
    }

    arguments[argPos] = arg;
  }

  void kernel::runFromArguments(){
    // [-] OCCA_MAX_ARGS = 25
#include "operators/occaRunFromArguments.cpp"

    return;
  }

#include "operators/occaOperatorDefinitions.cpp"

  double kernel::timeTaken(){
    if(kHandle->nestedKernelCount == 0){
      return kHandle->timeTaken();
    }
    else{
      kernel &k1 = kHandle->nestedKernels[0];
      kernel &k2 = kHandle->nestedKernels[kHandle->nestedKernelCount - 1];

      void *start = k1.kHandle->startTime;
      void *end   = k2.kHandle->endTime;

      return k1.kHandle->timeTakenBetween(start, end);
    }
  }

  void kernel::free(){
    kHandle->free();
    delete kHandle;

    if(kHandle->nestedKernelCount){
      for(int k = 0; k < kHandle->nestedKernelCount; ++k)
        kHandle->nestedKernels[k].free();

      delete [] kHandle->nestedKernels;
    }
  }

  kernelDatabase::kernelDatabase() :
    kernelName(""),
    modelKernelCount(0),
    kernelCount(0) {}

  kernelDatabase::kernelDatabase(const std::string kernelName_) :
    kernelName(kernelName_),
    modelKernelCount(0),
    kernelCount(0) {}

  kernelDatabase::kernelDatabase(const kernelDatabase &kdb) :
    kernelName(kdb.kernelName),

    modelKernelCount(kdb.modelKernelCount),
    modelKernelAvailable(kdb.modelKernelAvailable),

    kernelCount(kdb.kernelCount),
    kernels(kdb.kernels),
    kernelAllocated(kdb.kernelAllocated) {}


  kernelDatabase& kernelDatabase::operator = (const kernelDatabase &kdb){
    kernelName = kdb.kernelName;

    modelKernelCount     = kdb.modelKernelCount;
    modelKernelAvailable = kdb.modelKernelAvailable;

    kernelCount     = kdb.kernelCount;
    kernels         = kdb.kernels;
    kernelAllocated = kdb.kernelAllocated;

    return *this;
  }

  void kernelDatabase::modelKernelIsAvailable(const int id){
    OCCA_CHECK(0 <= id);

    if(modelKernelCount <= id){
      modelKernelCount = (id + 1);
      modelKernelAvailable.resize(modelKernelCount, false);
    }

    modelKernelAvailable[id] = true;
  }

  void kernelDatabase::addKernel(device d, kernel k){
    addKernel(d.id_, k);
  }

  void kernelDatabase::addKernel(const int id, kernel k){
    OCCA_CHECK(0 <= id);

    if(kernelCount <= id){
      kernelCount = (id + 1);

      kernels.resize(kernelCount);
      kernelAllocated.resize(kernelCount, false);
    }

    kernels[id] = k;
    kernelAllocated[id] = true;
  }

  void kernelDatabase::loadKernelFromLibrary(device &d){
    addKernel(d, library::loadKernel(d, kernelName));
  }
  //==================================


  //---[ Memory ]---------------------
  memory::memory() :
    mode_(),
    strMode(""),
    mHandle(NULL) {}

  memory::memory(const memory &m) :
    mode_(m.mode_),
    strMode(m.strMode),
    mHandle(m.mHandle) {}

  memory& memory::operator = (const memory &m){
    mode_   = m.mode_;
    strMode = m.strMode;

    mHandle = m.mHandle;

    return *this;
  }

  std::string& memory::mode(){
    return strMode;
  }

  void* memory::textureArg() const {
    return (void*) ((mHandle->textureInfo).arg);
  }

  void* memory::getMemoryHandle(){
    return mHandle->getMemoryHandle();
  }

  void* memory::getTextureHandle(){
    return mHandle->getTextureHandle();
  }

  void memory::copyFrom(const void *source,
                        const uintptr_t bytes,
                        const uintptr_t offset){
    mHandle->copyFrom(source, bytes, offset);
  }

  void memory::copyFrom(const memory &source,
                        const uintptr_t bytes,
                        const uintptr_t destOffset,
                        const uintptr_t srcOffset){
    mHandle->copyFrom(source.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::copyTo(void *dest,
                      const uintptr_t bytes,
                      const uintptr_t offset){
    mHandle->copyTo(dest, bytes, offset);
  }

  void memory::copyTo(memory &dest,
                      const uintptr_t bytes,
                      const uintptr_t destOffset,
                      const uintptr_t srcOffset){
    mHandle->copyTo(dest.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::asyncCopyFrom(const void *source,
                             const uintptr_t bytes,
                             const uintptr_t offset){
    mHandle->asyncCopyFrom(source, bytes, offset);
  }

  void memory::asyncCopyFrom(const memory &source,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset){
    mHandle->asyncCopyFrom(source.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::asyncCopyTo(void *dest,
                           const uintptr_t bytes,
                           const uintptr_t offset){
    mHandle->asyncCopyTo(dest, bytes, offset);
  }

  void memory::asyncCopyTo(memory &dest,
                           const uintptr_t bytes,
                           const uintptr_t destOffset,
                           const uintptr_t srcOffset){
    mHandle->asyncCopyTo(dest.mHandle, bytes, destOffset, srcOffset);
  }

  void memcpy(memory &dest,
              const void *source,
              const uintptr_t bytes,
              const uintptr_t offset){
    dest.copyFrom(source, bytes, offset);
  }

  void memcpy(memory &dest,
              const memory &source,
              const uintptr_t bytes,
              const uintptr_t destOffset,
              const uintptr_t srcOffset){
    dest.copyFrom(source, bytes, destOffset, srcOffset);
  }

  void memcpy(void *dest,
              memory &source,
              const uintptr_t bytes,
              const uintptr_t offset){
    source.copyTo(dest, bytes, offset);
  }

  void memcpy(memory &dest,
              memory &source,
              const uintptr_t bytes,
              const uintptr_t destOffset,
              const uintptr_t srcOffset){
    source.copyTo(dest, bytes, destOffset, srcOffset);
  }

  void asyncMemcpy(memory &dest,
                   const void *source,
                   const uintptr_t bytes,
                   const uintptr_t offset){
    dest.asyncCopyFrom(source, bytes, offset);
  }

  void asyncMemcpy(memory &dest,
                   const memory &source,
                   const uintptr_t bytes,
                   const uintptr_t destOffset,
                   const uintptr_t srcOffset){
    dest.asyncCopyFrom(source, bytes, destOffset, srcOffset);
  }

  void asyncMemcpy(void *dest,
                   memory &source,
                   const uintptr_t bytes,
                   const uintptr_t offset){
    source.asyncCopyTo(dest, bytes, offset);
  }

  void asyncMemcpy(memory &dest,
                   memory &source,
                   const uintptr_t bytes,
                   const uintptr_t destOffset,
                   const uintptr_t srcOffset){
    source.asyncCopyTo(dest, bytes, destOffset, srcOffset);
  }

  void memory::swap(memory &m){
    occa::mode mode2 = m.mode_;
    m.mode_        = mode_;
    mode_          = mode2;

    std::string strMode2 = m.strMode;
    m.strMode            = strMode;
    strMode              = strMode2;

    memory_v *mHandle2 = m.mHandle;
    m.mHandle          = mHandle;
    mHandle            = mHandle2;
  }

  void memory::free(){
    mHandle->free();
    delete mHandle;
  }
  //==================================


  //---[ Device ]---------------------
  device::device() :
    modelID_(-1),
    id_(-1),

    dHandle(NULL),
    currentStream(NULL) {}

  device::device(const device &d) :
    mode_(d.mode_),
    strMode(d.strMode),

    modelID_(d.modelID_),
    id_(d.id_),

    dHandle(d.dHandle),

    currentStream(d.currentStream),
    streams(d.streams) {

    if(dHandle)
      dHandle->dev = this;
  }

  device& device::operator = (const device &d){
    mode_ = d.mode_;

    modelID_ = d.modelID_;
    id_      = d.id_;

    dHandle = d.dHandle;

    currentStream = d.currentStream;
    streams       = d.streams;

    if(dHandle)
      dHandle->dev = this;

    return *this;
  }

  void device::setup(occa::mode m,
                     const int arg1, const int arg2){
    mode_   = m;
    strMode = modeToStr(m);

    switch(m){
    case Pthreads:
#if OCCA_PTHREADS_ENABLED
      dHandle = new device_t<Pthreads>(); break;
#else
      std::cout << "OCCA mode [Pthreads] is not enabled\n";
      throw 1;
#endif

    case OpenMP:
      dHandle = new device_t<OpenMP>(); break;

    case OpenCL:
#if OCCA_OPENCL_ENABLED
      dHandle = new device_t<OpenCL>(); break;
#else
      std::cout << "OCCA mode [OpenCL] is not enabled\n";
      throw 1;
#endif

    case CUDA:
#if OCCA_CUDA_ENABLED
      dHandle = new device_t<CUDA>(); break;
#else
      std::cout << "OCCA mode [CUDA] is not enabled\n";
      throw 1;
#endif

    case COI:
#if OCCA_COI_ENABLED
      dHandle = new device_t<COI>(); break;
#else
      std::cout << "OCCA mode [COI] is not enabled\n";
      throw 1;
#endif

    default:
      std::cout << "Incorrect OCCA mode given\n";
      throw 1;
    }

    dHandle->dev = this;
    dHandle->setup(arg1, arg2);

    modelID_ = library::deviceModelID(getIdentifier());
    id_      = library::genDeviceID();

    currentStream = genStream();
  }

  deviceIdentifier device::getIdentifier() const {
    return dHandle->getIdentifier();
  }

  void device::setup(const std::string &m,
                     const int arg1, const int arg2){
    setup(strToMode(m), arg1, arg2);
  }

  void device::setCompiler(const std::string &compiler_){
    dHandle->setCompiler(compiler_);
  }

  void device::setCompilerEnvScript(const std::string &compilerEnvScript_){
    dHandle->setCompilerEnvScript(compilerEnvScript_);
  }

  void device::setCompilerFlags(const std::string &compilerFlags_){
    dHandle->setCompilerFlags(compilerFlags_);
  }

  std::string& device::getCompiler(){
    return dHandle->getCompiler();
  }

  std::string& device::getCompilerEnvScript(){
    return dHandle->getCompilerEnvScript();
  }

  std::string& device::getCompilerFlags(){
    return dHandle->getCompilerFlags();
  }

  int device::modelID(){
    return modelID_;
  }

  int device::id(){
    return id_;
  }

  int device::modeID(){
    return mode_;
  }

  std::string& device::mode(){
    return strMode;
  }

  void device::flush(){
    dHandle->flush();
  }

  void device::finish(){
    dHandle->finish();
  }

  stream device::genStream(){
    streams.push_back( dHandle->genStream() );
    return streams.back();
  }

  stream device::getStream(){
    return currentStream;
  }

  void device::setStream(stream s){
    currentStream = s;
  }

  stream device::wrapStream(void *handle_){
    return dHandle->wrapStream(handle_);
  }

  tag device::tagStream(){
    return dHandle->tagStream();
  }

  double device::timeBetween(const tag &startTag, const tag &endTag){
    return dHandle->timeBetween(startTag, endTag);
  }

  void device::free(stream s){
    dHandle->freeStream(s);
  }

  kernel device::buildKernelFromSource(const std::string &filename,
                                       const std::string &functionName,
                                       const kernelInfo &info_){
    kernel ker;

    ker.mode_   = mode_;
    ker.strMode = strMode;

    ker.kHandle      = dHandle->buildKernelFromSource(filename, functionName, info_);
    ker.kHandle->dev = this;

    return ker;
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &functionName){
    kernel ker;

    ker.mode_   = mode_;
    ker.strMode = strMode;

    ker.kHandle      = dHandle->buildKernelFromBinary(filename, functionName);
    ker.kHandle->dev = this;

    return ker;
  }

  void device::cacheKernelInLibrary(const std::string &filename,
                                    const std::string &functionName,
                                    const kernelInfo &info_){
    dHandle->cacheKernelInLibrary(filename, functionName, info_);
  }

  kernel device::loadKernelFromLibrary(const char *cache,
                                       const std::string &functionName){
    kernel ker;

    ker.mode_   = mode_;
    ker.strMode = strMode;

    ker.kHandle      = dHandle->loadKernelFromLibrary(cache, functionName);
    ker.kHandle->dev = this;

    return ker;
  }

  kernel device::buildKernelFromLoopy(const std::string &filename,
                                      const std::string &functionName,
                                      const int useLoopyOrFloopy){
    return buildKernelFromLoopy(filename, functionName, "", useLoopyOrFloopy);
  }

  kernel device::buildKernelFromLoopy(const std::string &filename,
                                      const std::string &functionName,
                                      const std::string &pythonCode,
                                      const int useLoopyOrFloopy){
    std::string cachedBinary = getCachedName(filename, pythonCode);

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found loo.py cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildKernelFromBinary(cachedBinary, functionName);
    }

    std::string prefix, cacheName;

    getFilePrefixAndName(cachedBinary, prefix, cacheName);

    const std::string pCachedBinary = prefix + "p_" + cacheName;
    const std::string iCachedBinary = prefix + "i_" + cacheName;

    std::string loopyLang   = "loopy";
    std::string loopyHeader = pythonCode;

    if(useLoopyOrFloopy == occa::useFloopy){
      loopyHeader = "!$loopy begin transform\n" + loopyHeader + "\n!$loopy end transform\n";

      loopyLang = "floopy";
    }

    std::ofstream fs;
    fs.open(pCachedBinary.c_str());

    fs << loopyHeader << "\n\n" << readFile(filename);

    fs.close();

    std::stringstream command;

    command << "floopy --lang=" << loopyLang << " --target=cl:0,0 "
            << pCachedBinary << " " << iCachedBinary;

    const std::string &sCommand = command.str();

    std::cout << sCommand << '\n';

    system(sCommand.c_str());

    return buildKernelFromSource(iCachedBinary, functionName);
  }

  memory device::wrapMemory(void *handle_,
                            const uintptr_t bytes){
    memory mem;

    mem.mode_   = mode_;
    mem.strMode = strMode;

    mem.mHandle = dHandle->wrapMemory(handle_, bytes);
    mem.mHandle->dev = this;

    return mem;
  }

  memory device::wrapTexture(void *handle_,
                             const int dim, const occa::dim &dims,
                             occa::formatType type, const int permissions){
    if((dim != 1) && (dim != 2)){
      printf("Textures of [%dD] are not supported, only 1D or 2D are supported at the moment.\n", dim);
      throw 1;
    }

    memory mem;

    mem.mode_   = mode_;
    mem.strMode = strMode;

    mem.mHandle = dHandle->wrapTexture(handle_,
                                       dim, dims,
                                       type, permissions);
    mem.mHandle->dev = this;

    return mem;
  }

  memory device::malloc(const uintptr_t bytes,
                        void *source){
    memory mem;

    mem.mode_   = mode_;
    mem.strMode = strMode;

    mem.mHandle      = dHandle->malloc(bytes, source);
    mem.mHandle->dev = this;

    return mem;
  }

  memory device::talloc(const int dim, const occa::dim &dims,
                        void *source,
                        occa::formatType type, const int permissions){
    if((dim != 1) && (dim != 2)){
      printf("Textures of [%dD] are not supported, only 1D or 2D are supported at the moment.\n", dim);
      throw 1;
    }

    if(source == NULL){
      printf("Non-NULL source is required for [talloc] (texture allocation).\n");
      throw 1;
    }

    memory mem;

    mem.mode_   = mode_;
    mem.strMode = strMode;

    mem.mHandle      = dHandle->talloc(dim, dims, source, type, permissions);
    mem.mHandle->dev = this;

    return mem;
  }

  void device::free(){
    const int streamCount = streams.size();

    for(int i = 0; i < streamCount; ++i)
      dHandle->freeStream(streams[i]);

    dHandle->free();

    delete dHandle;
  }

  int device::simdWidth(){
    return dHandle->simdWidth();
  }

  mutex_t deviceListMutex;
  std::vector<device> deviceList;

  std::vector<device>& getDeviceList(){

    deviceListMutex.lock();

    if(deviceList.size()){
      deviceListMutex.unlock();
      return deviceList;
    }

    device_t<OpenMP>::appendAvailableDevices(deviceList);

#if OCCA_PTHREADS_ENABLED
    device_t<Pthreads>::appendAvailableDevices(deviceList);
#endif
#if OCCA_OPENCL_ENABLED
    device_t<OpenCL>::appendAvailableDevices(deviceList);
#endif
#if OCCA_CUDA_ENABLED
    device_t<CUDA>::appendAvailableDevices(deviceList);
#endif
#if OCCA_COI_ENABLED
    device_t<COI>::appendAvailableDevices(deviceList);
#endif

    deviceListMutex.unlock();

    return deviceList;
  }

  deviceIdentifier::deviceIdentifier() :
    mode_(OpenMP) {}

  deviceIdentifier::deviceIdentifier(occa::mode m,
                                     const char *c, const size_t chars){
    mode_ = m;
    load(c, chars);
  }

  deviceIdentifier::deviceIdentifier(occa::mode m, const std::string &s){
    mode_ = m;
    load(s);
  }

  deviceIdentifier::deviceIdentifier(const deviceIdentifier &di) :
    mode_(di.mode_),
    flagMap(di.flagMap) {}

  deviceIdentifier& deviceIdentifier::operator = (const deviceIdentifier &di){
    mode_ = di.mode_;
    flagMap = di.flagMap;

    return *this;
  }

  void deviceIdentifier::load(const char *c, const size_t chars){
    const char *c1 = c;

    while(((c1 - c) < chars) && (*c1 != '\0')){
      const char *c2 = c1;
      const char *c3;

      while(*c2 != '|')
        ++c2;

      c3 = (c2 + 1);

      while(((c3 - c) < chars) &&
            (*c3 != '\0') && (*c3 != '|'))
        ++c3;

      flagMap[std::string(c1, c2 - c1)] = std::string(c2 + 1, c3 - c2 - 1);

      c1 = (c3 + 1);
    }
  }

  void deviceIdentifier::load(const std::string &s){
    return load(s.c_str(), s.size());
  }

  std::string deviceIdentifier::flattenFlagMap() const {
    std::string ret = "";

    cFlagMapIterator it = flagMap.begin();

    if(it == flagMap.end())
      return "";

    ret += it->first;
    ret += '|';
    ret += it->second;
    ++it;

    while(it != flagMap.end()){
      ret += '|';
      ret += it->first;
      ret += '|';
      ret += it->second;

      ++it;
    }

    return ret;
  }

  int deviceIdentifier::compare(const deviceIdentifier &b) const {
    if(mode_ != b.mode_)
      return (mode_ < b.mode_) ? -1 : 1;

    cFlagMapIterator it1 =   flagMap.begin();
    cFlagMapIterator it2 = b.flagMap.begin();

    while(it1 != flagMap.end()){
      const std::string &s1 = it1->second;
      const std::string &s2 = it2->second;

      const int cmp = s1.compare(s2);

      if(cmp)
        return cmp;

      ++it1;
      ++it2;
    }

    return 0;
  }
  //==================================
};
