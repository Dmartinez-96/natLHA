#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "natlha_execution_context.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NATLHA_HAS_CUDA
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
#endif

namespace {

bool expect(bool condition, const std::string& description) {
    if (condition) return true;
    std::cerr << description << "\n";
    return false;
}

}  // namespace

int main() {
    using namespace natlha::detail;
    bool ok = true;

#ifdef NATLHA_HAS_CUDA
    std::cout << "execution-context mode=cuda-fiber\n";
#else
    std::cout << "execution-context mode=cpu-thread\n";
#endif

    int odeSubmissions = 0;
    OdeSubmitFunction odeSubmit = [&](std::vector<double> state,
                                       double start,
                                       double end,
                                       double step) {
        ++odeSubmissions;
        state[0] = start + end + step;
        return state;
    };
    int qSusySubmissions = 0;
    QSusySubmitFunction qSusySubmit = [&](const std::vector<double>&,
                                          double high,
                                          double,
                                          double) {
        ++qSusySubmissions;
        QSusyResult result;
        result.logScale = high;
        result.rootsFound = 1;
        return result;
    };
    const CudaExecutionContext context{&odeSubmit, &qSusySubmit};
    {
        const ScopedCudaExecutionContext scope(context);
        std::vector<double> state(44, 0.0);
        const std::vector<double> submitted = solveODEs(state, 1.0, 2.0, 0.25);
        ok &= expect(odeSubmissions == 1 && submitted[0] == 3.25,
                     "ODE dispatch did not use the installed execution context once");

        const QSusyResult submittedRoot = findQSusy({}, 17.0, -1.0e-6, 0.1);
        ok &= expect(qSusySubmissions == 1 && submittedRoot.logScale == 17.0,
                     "Q_SUSY dispatch did not use the installed execution context once");

        {
            const ScopedCudaDispatchSuppression suppress;
            ok &= expect(currentCudaExecutionContext() == nullptr
                             && cudaDispatchSuppressed(),
                         "dispatch suppression left the CUDA context visible");
            try {
                findQSusy({}, 17.0, -1.0e-6, 0.1);
                ok &= expect(false, "suppressed Q_SUSY dispatch did not enter CPU validation");
            } catch (const NumericalFailure&) {
                ok &= expect(qSusySubmissions == 1,
                             "suppressed Q_SUSY dispatch resubmitted to CUDA");
            }
        }
        ok &= expect(currentCudaExecutionContext() == &context
                         && !cudaDispatchSuppressed(),
                     "dispatch suppression did not restore the previous context");
    }
    ok &= expect(currentCudaExecutionContext() == nullptr,
                 "execution context escaped its scope");

    OdeSubmitFunction recursiveSubmit = [&](std::vector<double> state,
                                             double start,
                                             double end,
                                             double step) {
        return solveODEs(std::move(state), start, end, step);
    };
    const CudaExecutionContext recursiveContext{&recursiveSubmit, nullptr};
    {
        const ScopedCudaExecutionContext scope(recursiveContext);
        try {
            solveODEs(std::vector<double>(44, 0.0), 1.0, 2.0, 0.25);
            ok &= expect(false, "recursive CUDA ODE submission was accepted");
        } catch (const std::logic_error& failure) {
            ok &= expect(std::string(failure.what()).find("recursive CUDA")
                             != std::string::npos,
                         "recursive dispatch lost its distinct diagnostic");
        }
    }
    {
        const ScopedCudaExecutionContext scope(context);
        solveODEs(std::vector<double>(44, 0.0), 1.0, 2.0, 0.25);
        ok &= expect(odeSubmissions == 2,
                     "ODE dispatch depth did not recover after recursive rejection");
    }

    QSusySubmitFunction recursiveQSusySubmit = [&] (
            const std::vector<double>& state,
            double high,
            double step,
            double maxDeltaLogQ) {
        return findQSusy(state, high, step, maxDeltaLogQ);
    };
    const CudaExecutionContext recursiveQSusyContext{
        nullptr, &recursiveQSusySubmit};
    {
        const ScopedCudaExecutionContext scope(recursiveQSusyContext);
        try {
            findQSusy({}, 17.0, -1.0e-6, 0.1);
            ok &= expect(false, "recursive CUDA Q_SUSY submission was accepted");
        } catch (const std::logic_error& failure) {
            ok &= expect(std::string(failure.what()).find("recursive CUDA")
                             != std::string::npos,
                         "recursive Q_SUSY dispatch lost its distinct diagnostic");
        }
    }
    {
        const ScopedCudaExecutionContext scope(context);
        findQSusy({}, 17.0, -1.0e-6, 0.1);
        ok &= expect(qSusySubmissions == 2,
                     "Q_SUSY dispatch depth did not recover after recursive rejection");
    }

    OdeSubmitFunction outerSubmit = odeSubmit;
    OdeSubmitFunction innerSubmit = odeSubmit;
    const CudaExecutionContext outer{&outerSubmit, nullptr};
    const CudaExecutionContext inner{&innerSubmit, nullptr};
    {
        const ScopedCudaExecutionContext outerScope(outer);
        try {
            const ScopedCudaExecutionContext innerScope(inner);
            throw std::runtime_error("intentional context-unwind test");
        } catch (const std::runtime_error&) {
            ok &= expect(currentCudaExecutionContext() == &outer,
                         "exception unwind did not restore the outer context");
        }
    }
    ok &= expect(currentCudaExecutionContext() == nullptr,
                 "outer execution context was not restored after exception test");

#ifdef NATLHA_HAS_CUDA
    bool firstFiberIsolated = false;
    bool secondFiberIsolated = false;
    const CudaExecutionContext firstFiberContext{&outerSubmit, nullptr};
    const CudaExecutionContext secondFiberContext{&innerSubmit, nullptr};
    boost::fibers::fiber firstFiber([&] {
        const ScopedCudaExecutionContext scope(firstFiberContext);
        boost::this_fiber::yield();
        firstFiberIsolated = currentCudaExecutionContext() == &firstFiberContext;
    });
    boost::fibers::fiber secondFiber([&] {
        const ScopedCudaExecutionContext scope(secondFiberContext);
        boost::this_fiber::yield();
        secondFiberIsolated = currentCudaExecutionContext() == &secondFiberContext;
    });
    firstFiber.join();
    secondFiber.join();
    ok &= expect(firstFiberIsolated && secondFiberIsolated
                     && currentCudaExecutionContext() == nullptr,
                 "CUDA execution contexts leaked across interleaved point fibers");
#endif

    return ok ? 0 : 1;
}
