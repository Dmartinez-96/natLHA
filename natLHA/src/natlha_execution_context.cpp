#include "natlha_execution_context.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#ifdef NATLHA_HAS_CUDA
#include <boost/fiber/fss.hpp>
#endif

namespace {

#ifdef NATLHA_HAS_CUDA
void retainBorrowedContext(natlha::detail::CudaExecutionContext*) {}

boost::fibers::fiber_specific_ptr<natlha::detail::CudaExecutionContext>
    executionContext(retainBorrowedContext);
boost::fibers::fiber_specific_ptr<unsigned> suppressionDepth;
boost::fibers::fiber_specific_ptr<unsigned> dispatchDepth;

unsigned& fiberCounter(boost::fibers::fiber_specific_ptr<unsigned>& counter) {
    if (counter.get() == nullptr) counter.reset(new unsigned(0));
    return *counter;
}
#else
thread_local const natlha::detail::CudaExecutionContext* executionContext = nullptr;
thread_local unsigned suppressionDepth = 0;
thread_local unsigned dispatchDepth = 0;
#endif

}  // namespace

namespace natlha::detail {

const CudaExecutionContext* currentCudaExecutionContext() {
#ifdef NATLHA_HAS_CUDA
    return fiberCounter(suppressionDepth) == 0 ? executionContext.get() : nullptr;
#else
    return suppressionDepth == 0 ? executionContext : nullptr;
#endif
}

bool cudaDispatchSuppressed() {
#ifdef NATLHA_HAS_CUDA
    return fiberCounter(suppressionDepth) != 0;
#else
    return suppressionDepth != 0;
#endif
}

ScopedCudaExecutionContext::ScopedCudaExecutionContext(
        const CudaExecutionContext& replacement)
#ifdef NATLHA_HAS_CUDA
    : previous_(executionContext.get()) {
    executionContext.reset(const_cast<CudaExecutionContext*>(&replacement));
#else
    : previous_(executionContext) {
    executionContext = &replacement;
#endif
}

ScopedCudaExecutionContext::~ScopedCudaExecutionContext() {
#ifdef NATLHA_HAS_CUDA
    executionContext.reset(const_cast<CudaExecutionContext*>(previous_));
#else
    executionContext = previous_;
#endif
}

ScopedCudaDispatchSuppression::ScopedCudaDispatchSuppression()
#ifdef NATLHA_HAS_CUDA
    : previousDepth_(fiberCounter(suppressionDepth)) {
    unsigned& depth = fiberCounter(suppressionDepth);
    if (depth == std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("CUDA dispatch suppression depth overflow");
    }
    ++depth;
#else
    : previousDepth_(suppressionDepth) {
    if (suppressionDepth == std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("CUDA dispatch suppression depth overflow");
    }
    ++suppressionDepth;
#endif
}

ScopedCudaDispatchSuppression::~ScopedCudaDispatchSuppression() {
#ifdef NATLHA_HAS_CUDA
    fiberCounter(suppressionDepth) = previousDepth_;
#else
    suppressionDepth = previousDepth_;
#endif
}

ScopedCudaDispatchCall::ScopedCudaDispatchCall(const char* operation) {
#ifdef NATLHA_HAS_CUDA
    unsigned& depth = fiberCounter(dispatchDepth);
    if (depth != 0) {
#else
    if (dispatchDepth != 0) {
#endif
        throw std::logic_error(
            std::string("recursive CUDA numerical dispatch while submitting ")
            + operation);
    }
#ifdef NATLHA_HAS_CUDA
    ++depth;
#else
    ++dispatchDepth;
#endif
    active_ = true;
}

ScopedCudaDispatchCall::~ScopedCudaDispatchCall() {
    if (active_) {
#ifdef NATLHA_HAS_CUDA
        --fiberCounter(dispatchDepth);
#else
        --dispatchDepth;
#endif
    }
}

}  // namespace natlha::detail
