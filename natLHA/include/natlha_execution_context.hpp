#ifndef NATLHA_EXECUTION_CONTEXT_HPP
#define NATLHA_EXECUTION_CONTEXT_HPP

#include <functional>
#include <vector>

struct QSusyResult;

namespace natlha::detail {

using OdeSubmitFunction = std::function<std::vector<double>(
    std::vector<double>, double, double, double)>;
using QSusySubmitFunction = std::function<QSusyResult(
    const std::vector<double>&, double, double, double)>;

/// Logical-point-local RGE and Q_SUSY CUDA submission surface (fiber-local in CUDA builds,
/// thread-local in CPU-only builds). Device schedulers must call their batch primitives
/// directly; invoking either public dispatcher from a submit function is rejected as
/// recursion instead of silently creating a scheduler cycle.
struct CudaExecutionContext {
    const OdeSubmitFunction* submitOde = nullptr;
    const QSusySubmitFunction* submitQSusy = nullptr;
};

const CudaExecutionContext* currentCudaExecutionContext();
bool cudaDispatchSuppressed();

class ScopedCudaExecutionContext {
public:
    explicit ScopedCudaExecutionContext(const CudaExecutionContext& replacement);
    ~ScopedCudaExecutionContext();

    ScopedCudaExecutionContext(const ScopedCudaExecutionContext&) = delete;
    ScopedCudaExecutionContext& operator=(const ScopedCudaExecutionContext&) = delete;

private:
    const CudaExecutionContext* previous_ = nullptr;
};

/// Temporarily force the public RGE and Q_SUSY dispatchers through their CPU authority paths.
class ScopedCudaDispatchSuppression {
public:
    ScopedCudaDispatchSuppression();
    ~ScopedCudaDispatchSuppression();

    ScopedCudaDispatchSuppression(const ScopedCudaDispatchSuppression&) = delete;
    ScopedCudaDispatchSuppression& operator=(const ScopedCudaDispatchSuppression&) = delete;

private:
    unsigned previousDepth_ = 0;
};

/// Guard one call from a public RGE or Q_SUSY dispatcher into a CUDA submit function.
class ScopedCudaDispatchCall {
public:
    explicit ScopedCudaDispatchCall(const char* operation);
    ~ScopedCudaDispatchCall();

    ScopedCudaDispatchCall(const ScopedCudaDispatchCall&) = delete;
    ScopedCudaDispatchCall& operator=(const ScopedCudaDispatchCall&) = delete;

private:
    bool active_ = false;
};

}  // namespace natlha::detail

#endif  // NATLHA_EXECUTION_CONTEXT_HPP
