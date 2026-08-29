#include "natlha_cuda_backend.hpp"
#include "MSSM_RGE_derivatives.inl"
#define NATLHA_QSUSY_HD __host__ __device__
#include "MSSM_QSUSY_helpers.inl"
#undef NATLHA_QSUSY_HD
#include "MSSM_RGE_solver.hpp"
#include "natlha_execution_context.hpp"

#include <cuda_runtime.h>

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace natlha::detail {
namespace {

// A finite attempt ceiling turns a divergent adaptive solve into an explicit status instead
// of a hung kernel. It is a safety bound, not a claim that a useful solve needs this many.
constexpr unsigned long long kMaxOdeAttempts = 1000000;
// Four warps per block is a conservative launch geometry. Occupancy is device- and
// precision-dependent; this is not asserted to be the performance optimum.
constexpr int kThreadsPerBlock = 128;
static_assert(
    std::is_same<std::underlying_type<CudaOdeStatus>::type, int>::value,
    "CUDA status device buffers and host enum storage must share int representation");
static_assert(
    std::is_same<std::underlying_type<CudaQSusyStatus>::type, int>::value,
    "CUDA Q_SUSY status device buffers and host enum storage must share int representation");

#define NATLHA_CUDA_HD __host__ __device__

struct DoubleDouble {
    double high = 0.0;
    double low = 0.0;

    NATLHA_CUDA_HD DoubleDouble() = default;
    NATLHA_CUDA_HD DoubleDouble(double value) : high(value), low(0.0) {}
    NATLHA_CUDA_HD DoubleDouble(double highPart, double lowPart)
        : high(highPart), low(lowPart) {}

    NATLHA_CUDA_HD explicit operator double() const { return high + low; }

};

NATLHA_CUDA_HD DoubleDouble quickTwoSum(double left, double right) {
    const double sum = left + right;
    return {sum, right - (sum - left)};
}

NATLHA_CUDA_HD DoubleDouble twoSum(double left, double right) {
    const double sum = left + right;
    const double rightVirtual = sum - left;
    const double error = (left - (sum - rightVirtual)) + (right - rightVirtual);
    return {sum, error};
}

NATLHA_CUDA_HD DoubleDouble twoProduct(double left, double right) {
    const double product = left * right;
    return {product, fma(left, right, -product)};
}

NATLHA_CUDA_HD DoubleDouble operator+(DoubleDouble left, DoubleDouble right) {
    const DoubleDouble sum = twoSum(left.high, right.high);
    const DoubleDouble normalized = quickTwoSum(
        sum.high, sum.low + left.low + right.low);
    return normalized;
}

NATLHA_CUDA_HD DoubleDouble& operator+=(DoubleDouble& left, DoubleDouble right) {
    left = left + right;
    return left;
}

NATLHA_CUDA_HD DoubleDouble operator-(DoubleDouble value) {
    return {-value.high, -value.low};
}

NATLHA_CUDA_HD DoubleDouble operator-(DoubleDouble left, DoubleDouble right) {
    return left + (-right);
}

NATLHA_CUDA_HD DoubleDouble operator*(DoubleDouble left, DoubleDouble right) {
    const DoubleDouble product = twoProduct(left.high, right.high);
    return quickTwoSum(
        product.high,
        product.low + left.high * right.low + left.low * right.high
            + left.low * right.low);
}

NATLHA_CUDA_HD DoubleDouble operator/(DoubleDouble left, DoubleDouble right) {
    const double first = left.high / right.high;
    const DoubleDouble remainder = left - right * DoubleDouble(first);
    const double second = remainder.high / right.high;
    return quickTwoSum(first, second);
}

NATLHA_CUDA_HD double toDouble(double value) {
    return value;
}

NATLHA_CUDA_HD double toDouble(DoubleDouble value) {
    return value.high + value.low;
}

NATLHA_CUDA_HD DoubleDouble pow(DoubleDouble value, double exponent) {
    if (exponent == 2.0) return value * value;
    if (exponent == 3.0) return value * value * value;
    if (exponent == 4.0) {
        const DoubleDouble square = value * value;
        return square * square;
    }
    // Device-reachable MSSM derivatives currently use only exponents 2.0, 3.0, and 4.0. Fail
    // closed if a future shared derivative adds an unsupported exponent instead of silently
    // reducing the double-double adjudication tier to binary64 for that term.
    return DoubleDouble(NAN);
}

void requireCuda(cudaError_t status, const std::string& operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { reserve(count); }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (data_ != nullptr) cudaFree(data_);
        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
        return *this;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

    void reserve(std::size_t count) {
        if (count <= count_) return;
        T* replacement = nullptr;
        requireCuda(
            cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)),
            "cudaMalloc");
        if (data_ != nullptr) {
            const cudaError_t releaseStatus = cudaFree(data_);
            if (releaseStatus != cudaSuccess) {
                cudaFree(replacement);
                requireCuda(releaseStatus, "cudaFree while growing a device buffer");
            }
        }
        data_ = replacement;
        count_ = count;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename Real>
__device__ bool finiteState(const Real (&state)[kRgeStateSize]) {
    for (std::size_t i = 0; i < kRgeStateSize; ++i) {
        if (!isfinite(toDouble(state[i]))) return false;
    }
    return true;
}

template <typename Real>
__device__ void combineStage(
        Real (&output)[kRgeStateSize],
        const Real (&state)[kRgeStateSize],
        double step,
        const Real* const* derivatives,
        const double* coefficients,
        int terms) {
    for (std::size_t component = 0; component < kRgeStateSize; ++component) {
        Real increment = Real(0.0);
        for (int term = 0; term < terms; ++term) {
            increment += coefficients[term] * derivatives[term][component];
        }
        output[component] = state[component] + step * increment;
    }
}

template <typename Real>
__global__ void integrateKernel(
        double* states,
        const double* startTimes,
        const double* endTimes,
        const double* initialSteps,
        int* statuses,
        unsigned long long* acceptedSteps,
        unsigned long long* rejectedSteps,
        std::size_t points,
        double absoluteTolerance,
        double relativeTolerance) {
    const std::size_t point =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= points) return;

    Real state[kRgeStateSize];
    for (std::size_t i = 0; i < kRgeStateSize; ++i) {
        state[i] = Real(states[point * kRgeStateSize + i]);
    }

    int status = static_cast<int>(CudaOdeStatus::Success);
    unsigned long long accepted = 0;
    unsigned long long rejected = 0;
    double time = startTimes[point];
    const double end = endTimes[point];
    const double suppliedStep = initialSteps[point];
    if (!finiteState(state) || !isfinite(time) || !isfinite(end)
            || !isfinite(suppliedStep)) {
        status = static_cast<int>(CudaOdeStatus::NonFiniteInput);
    }

    const double direction = end >= time ? 1.0 : -1.0;
    double step = direction * fabs(suppliedStep);
    if (step == 0.0) step = direction * 1.0e-6;

    Real k1[kRgeStateSize];
    Real k2[kRgeStateSize];
    Real k3[kRgeStateSize];
    Real k4[kRgeStateSize];
    Real k5[kRgeStateSize];
    Real k6[kRgeStateSize];
    Real k7[kRgeStateSize];
    Real stage[kRgeStateSize];
    Real fifthOrder[kRgeStateSize];

    for (unsigned long long attempt = 0;
         status == static_cast<int>(CudaOdeStatus::Success)
             && direction * (end - time) > 0.0;
         ++attempt) {
        if (attempt >= kMaxOdeAttempts) {
            status = static_cast<int>(CudaOdeStatus::StepLimit);
            break;
        }
        if (direction * (time + step - end) > 0.0) step = end - time;
        if (time + step == time) {
            status = static_cast<int>(CudaOdeStatus::StepUnderflow);
            break;
        }

        MSSMRGEDerivatives(state, k1);
        {
            const Real* derivatives[] = {k1};
            const double coefficients[] = {1.0 / 5.0};
            combineStage(stage, state, step, derivatives, coefficients, 1);
        }
        MSSMRGEDerivatives(stage, k2);
        {
            const Real* derivatives[] = {k1, k2};
            const double coefficients[] = {3.0 / 40.0, 9.0 / 40.0};
            combineStage(stage, state, step, derivatives, coefficients, 2);
        }
        MSSMRGEDerivatives(stage, k3);
        {
            const Real* derivatives[] = {k1, k2, k3};
            const double coefficients[] = {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0};
            combineStage(stage, state, step, derivatives, coefficients, 3);
        }
        MSSMRGEDerivatives(stage, k4);
        {
            const Real* derivatives[] = {k1, k2, k3, k4};
            const double coefficients[] = {
                19372.0 / 6561.0, -25360.0 / 2187.0,
                64448.0 / 6561.0, -212.0 / 729.0};
            combineStage(stage, state, step, derivatives, coefficients, 4);
        }
        MSSMRGEDerivatives(stage, k5);
        {
            const Real* derivatives[] = {k1, k2, k3, k4, k5};
            const double coefficients[] = {
                9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0,
                49.0 / 176.0, -5103.0 / 18656.0};
            combineStage(stage, state, step, derivatives, coefficients, 5);
        }
        MSSMRGEDerivatives(stage, k6);
        {
            const Real* derivatives[] = {k1, k2, k3, k4, k5, k6};
            const double coefficients[] = {
                35.0 / 384.0, 0.0, 500.0 / 1113.0,
                125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0};
            combineStage(fifthOrder, state, step, derivatives, coefficients, 6);
        }
        MSSMRGEDerivatives(fifthOrder, k7);

        double maximumError = 0.0;
        for (std::size_t component = 0; component < kRgeStateSize; ++component) {
            const Real fourthOrder = state[component] + step * (
                (5179.0 / 57600.0) * k1[component]
                + (7571.0 / 16695.0) * k3[component]
                + (393.0 / 640.0) * k4[component]
                - (92097.0 / 339200.0) * k5[component]
                + (187.0 / 2100.0) * k6[component]
                + (1.0 / 40.0) * k7[component]);
            const double scale = absoluteTolerance + relativeTolerance * (
                fabs(toDouble(state[component]))
                + fabs(step * toDouble(k1[component])));
            const double error =
                fabs(toDouble(fifthOrder[component] - fourthOrder)) / scale;
            maximumError = fmax(maximumError, error);
        }

        if (!finiteState(fifthOrder) || !isfinite(maximumError)) {
            status = static_cast<int>(CudaOdeStatus::NonFiniteState);
            break;
        }

        double factor = 5.0;
        if (maximumError > 0.0) {
            factor = 0.9 * ::pow(maximumError, -0.2);
            factor = fmin(5.0, fmax(0.2, factor));
        }
        if (maximumError <= 1.0) {
            for (std::size_t i = 0; i < kRgeStateSize; ++i) state[i] = fifthOrder[i];
            time += step;
            ++accepted;
        } else {
            ++rejected;
            factor = fmin(1.0, factor);
        }
        step *= factor;
    }

    if (status == static_cast<int>(CudaOdeStatus::Success)) {
        for (std::size_t i = 0; i < kRgeStateSize; ++i) {
            states[point * kRgeStateSize + i] = toDouble(state[i]);
        }
    }
    statuses[point] = status;
    acceptedSteps[point] = accepted;
    rejectedSteps[point] = rejected;
}

struct QSusyHelperDeviceInput {
    double state[kRgeStateSize];
    double logScale;
    double currentLogScale;
    double lowerLogScale;
    double maxDeltaLogQ;
    natlha::qsusy_numeric::StopPoint highPoint;
    natlha::qsusy_numeric::StopPoint lowPoint;
    natlha::qsusy_numeric::ScanState scanState{};
};

struct QSusyHelperDeviceResult {
    natlha::qsusy_numeric::StopPoint evaluatedPoint;
    double nextLogScale;
    int scanStepStatus;
    unsigned scanEvents;
    bool classificationOk;
    natlha::qsusy_numeric::ScanState scanState{};
};

__global__ void qSusyHelperKernel(
        const QSusyHelperDeviceInput* inputs,
        QSusyHelperDeviceResult* results,
        std::size_t points) {
    const std::size_t point =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= points) return;
    const QSusyHelperDeviceInput& input = inputs[point];
    QSusyHelperDeviceResult result{};
    result.evaluatedPoint = natlha::qsusy_numeric::evaluateStopPoint(
        input.state, kRgeStateSize, input.logScale, natlha::qsusy_numeric::kMZ);
    result.scanStepStatus = static_cast<int>(
        natlha::qsusy_numeric::nextScanLogScale(
            input.currentLogScale, input.lowerLogScale, input.maxDeltaLogQ,
            result.nextLogScale));
    result.scanState = input.scanState;
    result.classificationOk = natlha::qsusy_numeric::classifySegment(
        input.highPoint, input.lowPoint, result.scanState, result.scanEvents);
    results[point] = result;
}

struct QSusyDeviceResult {
    double stateAtRoot[kRgeStateSize];
    double logScale;
    double residual;
    double stop1Squared;
    double stop2Squared;
    double refinedBracketWidth;
    int status;
    unsigned long long acceptedSteps;
    unsigned long long rejectedSteps;
    unsigned long long scanSegments;
    double maxObservedDeltaLogQ;
    unsigned long long rootsFound;
    unsigned long long invalidBoundaries;
    unsigned long long nonFiniteBoundaries;
    unsigned long long refinementEvaluations;
};

template <typename Real>
__device__ void dopriDenseState(
        double queryTime,
        Real (&output)[kRgeStateSize],
        const Real (&oldState)[kRgeStateSize],
        const Real (&k1)[kRgeStateSize],
        const Real (&k3)[kRgeStateSize],
        const Real (&k4)[kRgeStateSize],
        const Real (&k5)[kRgeStateSize],
        const Real (&k6)[kRgeStateSize],
        const Real (&k7)[kRgeStateSize],
        double oldTime,
        double newTime) {
    // Adapted from Boost.Odeint runge_kutta_dopri5::calc_state.
    // Copyright 2010-2013 Karsten Ahnert, Mario Mulansky; 2012 Christoph Koke.
    // Boost Software License 1.0; see third_party/boost-derived/.
    const double dt = newTime - oldTime;
    const double theta = (queryTime - oldTime) / dt;
    const double x1 = 5.0 * (2558722523.0 - 31403016.0 * theta)
                      / 11282082432.0;
    const double x3 = 100.0 * (882725551.0 - 15701508.0 * theta)
                      / 32700410799.0;
    const double x4 = 25.0 * (443332067.0 - 31403016.0 * theta)
                      / 1880347072.0;
    const double x5 = 32805.0 * (23143187.0 - 3489224.0 * theta)
                      / 199316789632.0;
    const double x6 = 55.0 * (29972135.0 - 7076736.0 * theta)
                      / 822651844.0;
    const double x7 = 10.0 * (7414447.0 - 829305.0 * theta) / 29380423.0;
    const double thetaMinusOne = theta - 1.0;
    const double thetaSquared = theta * theta;
    const double a = thetaSquared * (3.0 - 2.0 * theta);
    const double b = thetaSquared * thetaMinusOne;
    const double c = thetaSquared * thetaMinusOne * thetaMinusOne;
    const double d = theta * thetaMinusOne * thetaMinusOne;
    const double b1 = a * (35.0 / 384.0) - c * x1 + d;
    const double b3 = a * (500.0 / 1113.0) + c * x3;
    const double b4 = a * (125.0 / 192.0) - c * x4;
    const double b5 = a * (-2187.0 / 6784.0) + c * x5;
    const double b6 = a * (11.0 / 84.0) - c * x6;
    const double b7 = b + c * x7;
    for (std::size_t component = 0; component < kRgeStateSize; ++component) {
        output[component] = oldState[component] + dt * (
            b1 * k1[component] + b3 * k3[component] + b4 * k4[component]
            + b5 * k5[component] + b6 * k6[component] + b7 * k7[component]);
    }
}

template <typename Real>
__device__ bool denseResidual(
        double queryTime,
        double& residual,
        natlha::qsusy_numeric::StopPoint& stop,
        Real (&scratch)[kRgeStateSize],
        const Real (&oldState)[kRgeStateSize],
        const Real (&k1)[kRgeStateSize],
        const Real (&k3)[kRgeStateSize],
        const Real (&k4)[kRgeStateSize],
        const Real (&k5)[kRgeStateSize],
        const Real (&k6)[kRgeStateSize],
        const Real (&k7)[kRgeStateSize],
        double oldTime,
        double newTime) {
    dopriDenseState(
        queryTime, scratch, oldState, k1, k3, k4, k5, k6, k7,
        oldTime, newTime);
    stop = natlha::qsusy_numeric::evaluateStopPoint(
        scratch, kRgeStateSize, queryTime, natlha::qsusy_numeric::kMZ);
    residual = stop.logResidual;
    return stop.numericallyValid && stop.physical;
}

template <typename Real>
struct DenseRootContext {
    Real* scratch;
    const Real* oldState;
    const Real* k1;
    const Real* k3;
    const Real* k4;
    const Real* k5;
    const Real* k6;
    const Real* k7;
    double oldTime;
    double newTime;
    int* status;

    __device__ bool evaluate(double query, double& residual) {
        natlha::qsusy_numeric::StopPoint stop;
        if (!denseResidual(
                query, residual, stop,
                *reinterpret_cast<Real (*)[kRgeStateSize]>(scratch),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(oldState),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k1),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k3),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k4),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k5),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k6),
                *reinterpret_cast<const Real (*)[kRgeStateSize]>(k7),
                oldTime, newTime)) {
            // The CPU authority can isolate this boundary and continue scanning. The
            // device path instead fails closed: its structured status forces CPU/MPFR
            // adjudication, avoiding extra divergent control flow inside this kernel.
            *status = stop.numericallyValid
                ? static_cast<int>(CudaQSusyStatus::NonPhysicalBracket)
                : static_cast<int>(CudaQSusyStatus::NonFiniteBoundary);
            return false;
        }
        return true;
    }
};

__device__ double tomsSafeDivide(double numerator, double denominator, double fallback) {
    constexpr double maximumFinite = 1.7976931348623157e308;
    if (::fabs(denominator) < 1.0
            && ::fabs(denominator * maximumFinite) <= ::fabs(numerator)) {
        return fallback;
    }
    return numerator / denominator;
}

__device__ int tomsSign(double value) {
    return (0.0 < value) - (value < 0.0);
}

__device__ double tomsSecant(
        double a, double b, double fa, double fb) {
    constexpr double tolerance = 5.0 * 2.2204460492503130808472633361816e-16;
    const double candidate = a - (fa / (fb - fa)) * (b - a);
    if (candidate <= a + ::fabs(a) * tolerance
            || candidate >= b - ::fabs(b) * tolerance) {
        return a + 0.5 * (b - a);
    }
    return candidate;
}

__device__ double tomsQuadratic(
        double a, double b, double d,
        double fa, double fb, double fd, unsigned iterations) {
    double coefficientB = tomsSafeDivide(fb - fa, b - a, 1.7976931348623157e308);
    double coefficientA = tomsSafeDivide(fd - fb, d - b, 1.7976931348623157e308);
    coefficientA = tomsSafeDivide(coefficientA - coefficientB, d - a, 0.0);
    if (coefficientA == 0.0) return tomsSecant(a, b, fa, fb);
    double candidate = (coefficientA < 0.0) == (fa < 0.0) ? a : b;
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        candidate -= tomsSafeDivide(
            fa + (coefficientB + coefficientA * (candidate - b)) * (candidate - a),
            coefficientB + coefficientA * (2.0 * candidate - a - b),
            1.0 + candidate - a);
    }
    return candidate <= a || candidate >= b
        ? tomsSecant(a, b, fa, fb) : candidate;
}

__device__ double tomsCubic(
        double a, double b, double d, double e,
        double fa, double fb, double fd, double fe) {
    const double q11 = (d - e) * fd / (fe - fd);
    const double q21 = (b - d) * fb / (fd - fb);
    const double q31 = (a - b) * fa / (fb - fa);
    const double d21 = (b - d) * fd / (fd - fb);
    const double d31 = (a - b) * fb / (fb - fa);
    const double q22 = (d21 - q11) * fb / (fe - fb);
    const double q32 = (d31 - q21) * fa / (fd - fa);
    const double d32 = (d31 - q21) * fd / (fd - fa);
    const double q33 = (d32 - q22) * fa / (fe - fa);
    const double candidate = q31 + q32 + q33 + a;
    return candidate <= a || candidate >= b
        ? tomsQuadratic(a, b, d, fa, fb, fd, 3) : candidate;
}

template <typename Real>
__device__ bool tomsBracket(
        DenseRootContext<Real>& context,
        double& a, double& b, double candidate,
        double& fa, double& fb, double& d, double& fd) {
    constexpr double tolerance = 2.0 * 2.2204460492503130808472633361816e-16;
    if ((b - a) < 2.0 * tolerance * a) candidate = a + 0.5 * (b - a);
    else if (candidate <= a + ::fabs(a) * tolerance)
        candidate = a + ::fabs(a) * tolerance;
    else if (candidate >= b - ::fabs(b) * tolerance)
        candidate = b - ::fabs(b) * tolerance;
    double fc = 0.0;
    if (!context.evaluate(candidate, fc)) return false;
    if (fc == 0.0) {
        a = candidate; fa = 0.0; d = 0.0; fd = 0.0;
    } else if (tomsSign(fa) * tomsSign(fc) < 0) {
        d = b; fd = fb; b = candidate; fb = fc;
    } else {
        d = a; fd = fa; a = candidate; fa = fc;
    }
    return true;
}

__device__ bool tomsClose(double a, double b) {
    constexpr double tolerance = 4.0 * 2.2204460492503130808472633361816e-16;
    return ::fabs(a - b) <= tolerance * ::fmin(::fabs(a), ::fabs(b));
}

template <typename Real>
__device__ bool refineDenseBracket(
        double& low,
        double& high,
        double& lowResidual,
        double& highResidual,
        unsigned long long& evaluations,
        int& status,
        Real (&scratch)[kRgeStateSize],
        const Real (&oldState)[kRgeStateSize],
        const Real (&k1)[kRgeStateSize],
        const Real (&k3)[kRgeStateSize],
        const Real (&k4)[kRgeStateSize],
        const Real (&k5)[kRgeStateSize],
        const Real (&k6)[kRgeStateSize],
        const Real (&k7)[kRgeStateSize],
        double oldTime,
        double newTime) {
    // Adapted from Boost.Math toms748_solve.
    // Copyright John Maddock 2006. Boost Software License 1.0; see
    // third_party/boost-derived/.
    // Bound refinement to twice the binary64 value-bit count (106 evaluations).
    constexpr unsigned long long maximumEvaluations =
        2ULL * static_cast<unsigned long long>(std::numeric_limits<double>::digits);
    unsigned long long remaining = maximumEvaluations;
    DenseRootContext<Real> context{
        scratch, oldState, k1, k3, k4, k5, k6, k7, oldTime, newTime, &status};
    double a = low, b = high, fa = lowResidual, fb = highResidual;
    double c = 0.0, d = 1.0e5, fd = 1.0e5, e = 1.0e5, fe = 1.0e5;
    if (tomsClose(a, b) || fa == 0.0 || fb == 0.0) {
        if (fa == 0.0) b = a;
        else if (fb == 0.0) a = b;
        low = a; high = b; lowResidual = fa; highResidual = fb;
        return true;
    }
    c = tomsSecant(a, b, fa, fb);
    ++evaluations;
    if (!tomsBracket(context, a, b, c, fa, fb, d, fd)) return false;
    --remaining;
    if (remaining && fa != 0.0 && !tomsClose(a, b)) {
        c = tomsQuadratic(a, b, d, fa, fb, fd, 2);
        e = d; fe = fd;
        ++evaluations;
        if (!tomsBracket(context, a, b, c, fa, fb, d, fd)) return false;
        --remaining;
    }
    while (remaining && fa != 0.0 && !tomsClose(a, b)) {
        const double oldA = a, oldB = b;
        constexpr double minimumDifference =
            32.0 * 2.2250738585072013830902327173324e-308;
        bool duplicate = ::fabs(fa-fb) < minimumDifference
            || ::fabs(fa-fd) < minimumDifference || ::fabs(fa-fe) < minimumDifference
            || ::fabs(fb-fd) < minimumDifference || ::fabs(fb-fe) < minimumDifference
            || ::fabs(fd-fe) < minimumDifference;
        c = duplicate ? tomsQuadratic(a,b,d,fa,fb,fd,2)
                      : tomsCubic(a,b,d,e,fa,fb,fd,fe);
        e = d; fe = fd;
        ++evaluations;
        if (!tomsBracket(context,a,b,c,fa,fb,d,fd)) return false;
        if (--remaining == 0 || fa == 0.0 || tomsClose(a,b)) break;
        duplicate = ::fabs(fa-fb) < minimumDifference
            || ::fabs(fa-fd) < minimumDifference || ::fabs(fa-fe) < minimumDifference
            || ::fabs(fb-fd) < minimumDifference || ::fabs(fb-fe) < minimumDifference
            || ::fabs(fd-fe) < minimumDifference;
        c = duplicate ? tomsQuadratic(a,b,d,fa,fb,fd,3)
                      : tomsCubic(a,b,d,e,fa,fb,fd,fe);
        ++evaluations;
        if (!tomsBracket(context,a,b,c,fa,fb,d,fd)) return false;
        if (--remaining == 0 || fa == 0.0 || tomsClose(a,b)) break;
        const double u = ::fabs(fa) < ::fabs(fb) ? a : b;
        const double fu = ::fabs(fa) < ::fabs(fb) ? fa : fb;
        c = u - 2.0 * (fu / (fb-fa)) * (b-a);
        if (::fabs(c-u) > 0.5*(b-a)) c = a + 0.5*(b-a);
        e = d; fe = fd;
        ++evaluations;
        if (!tomsBracket(context,a,b,c,fa,fb,d,fd)) return false;
        if (--remaining == 0 || fa == 0.0 || tomsClose(a,b)) break;
        if ((b-a) >= 0.5*(oldB-oldA)) {
            e = d; fe = fd;
            ++evaluations;
            if (!tomsBracket(context,a,b,a+0.5*(b-a),fa,fb,d,fd)) return false;
            --remaining;
        }
    }
    if (fa == 0.0) b = a;
    else if (fb == 0.0) a = b;
    low = a; high = b; lowResidual = fa; highResidual = fb;
    if (!tomsClose(a,b)) {
        status = static_cast<int>(CudaQSusyStatus::RefinementFailure);
        return false;
    }
    return true;
}

template <typename Real>
__global__ void qSusyKernel(
        const double* states,
        const double* highLogScales,
        const double* initialSteps,
        const double* maxDeltaLogQ,
        QSusyDeviceResult* results,
        std::size_t points,
        double absoluteTolerance,
        double relativeTolerance) {
    const std::size_t point =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= points) return;
    QSusyDeviceResult result{};
    result.status = static_cast<int>(CudaQSusyStatus::Success);
    Real state[kRgeStateSize];
    for (std::size_t i = 0; i < kRgeStateSize; ++i) {
        state[i] = Real(states[point * kRgeStateSize + i]);
    }
    double time = highLogScales[point];
    const double lower = ::log(natlha::qsusy_numeric::kSearchLowerScale);
    const double scanUpper = ::fmin(
        time,
        ::nextafter(::log(natlha::qsusy_numeric::kRootUpperScale), lower));
    double step = -::fabs(initialSteps[point]);
    const double maxSpacing = maxDeltaLogQ[point];
    if (!finiteState(state) || !natlha::qsusy_numeric::finiteDouble(time)
            || time <= lower || !natlha::qsusy_numeric::finiteDouble(step)
            || step == 0.0 || !natlha::qsusy_numeric::finiteDouble(maxSpacing)
            || maxSpacing <= 0.0) {
        result.status = static_cast<int>(CudaQSusyStatus::NonFiniteInput);
        results[point] = result;
        return;
    }

    Real k1[kRgeStateSize], k2[kRgeStateSize], k3[kRgeStateSize];
    Real k4[kRgeStateSize], k5[kRgeStateSize], k6[kRgeStateSize];
    Real k7[kRgeStateSize], stage[kRgeStateSize], fifth[kRgeStateSize];
    Real oldState[kRgeStateSize], scanHighState[kRgeStateSize];
    Real scanLowState[kRgeStateSize], scratch[kRgeStateSize];
    natlha::qsusy_numeric::ScanState scanState{};
    double storedRootScale = 0.0;

    for (unsigned long long attempt = 0; time > lower; ++attempt) {
        if (attempt >= kMaxOdeAttempts) {
            result.status = static_cast<int>(CudaQSusyStatus::StepLimit);
            break;
        }
        if (time + step < lower) step = lower - time;
        if (time + step == time) {
            result.status = static_cast<int>(CudaQSusyStatus::StepUnderflow);
            break;
        }
        for (std::size_t i = 0; i < kRgeStateSize; ++i) oldState[i] = state[i];
        const double oldTime = time;
        MSSMRGEDerivatives(oldState, k1);
        { const Real* d[] = {k1}; const double c[] = {1.0/5.0};
          combineStage(stage, oldState, step, d, c, 1); }
        MSSMRGEDerivatives(stage, k2);
        { const Real* d[] = {k1,k2}; const double c[] = {3.0/40.0,9.0/40.0};
          combineStage(stage, oldState, step, d, c, 2); }
        MSSMRGEDerivatives(stage, k3);
        { const Real* d[] = {k1,k2,k3}; const double c[] = {44.0/45.0,-56.0/15.0,32.0/9.0};
          combineStage(stage, oldState, step, d, c, 3); }
        MSSMRGEDerivatives(stage, k4);
        { const Real* d[] = {k1,k2,k3,k4}; const double c[] = {19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0};
          combineStage(stage, oldState, step, d, c, 4); }
        MSSMRGEDerivatives(stage, k5);
        { const Real* d[] = {k1,k2,k3,k4,k5}; const double c[] = {9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0};
          combineStage(stage, oldState, step, d, c, 5); }
        MSSMRGEDerivatives(stage, k6);
        { const Real* d[] = {k1,k2,k3,k4,k5,k6}; const double c[] = {35.0/384.0,0.0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0};
          combineStage(fifth, oldState, step, d, c, 6); }
        MSSMRGEDerivatives(fifth, k7);
        double maximumError = 0.0;
        for (std::size_t i = 0; i < kRgeStateSize; ++i) {
            const Real fourth = oldState[i] + step * (
                (5179.0/57600.0)*k1[i] + (7571.0/16695.0)*k3[i]
                + (393.0/640.0)*k4[i] - (92097.0/339200.0)*k5[i]
                + (187.0/2100.0)*k6[i] + (1.0/40.0)*k7[i]);
            const double scale = absoluteTolerance + relativeTolerance * (
                ::fabs(toDouble(oldState[i])) + ::fabs(step * toDouble(k1[i])));
            maximumError = ::fmax(maximumError,
                ::fabs(toDouble(fifth[i] - fourth)) / scale);
        }
        if (!finiteState(fifth) || !natlha::qsusy_numeric::finiteDouble(maximumError)) {
            result.status = static_cast<int>(CudaQSusyStatus::NonFiniteState);
            break;
        }
        double factor = maximumError > 0.0
            ? ::fmin(5.0, ::fmax(0.2, 0.9 * ::pow(maximumError, -0.2))) : 5.0;
        if (maximumError > 1.0) {
            ++result.rejectedSteps;
            step *= ::fmin(1.0, factor);
            continue;
        }

        for (std::size_t i = 0; i < kRgeStateSize; ++i) state[i] = fifth[i];
        time += step;
        ++result.acceptedSteps;
        const double segmentHigh = ::fmin(oldTime, scanUpper);
        const double segmentLow = ::fmax(time, lower);
        if (segmentHigh > segmentLow) {
            dopriDenseState(segmentHigh, scanHighState, oldState, k1,k3,k4,k5,k6,k7,
                            oldTime, time);
            auto highStop = natlha::qsusy_numeric::evaluateStopPoint(
                scanHighState, kRgeStateSize, segmentHigh,
                natlha::qsusy_numeric::kMZ);
            double scanHigh = segmentHigh;
            while (scanHigh > segmentLow
                    && result.status == static_cast<int>(CudaQSusyStatus::Success)) {
                double scanLow = 0.0;
                if (natlha::qsusy_numeric::nextScanLogScale(
                        scanHigh, segmentLow, maxSpacing, scanLow)
                        != natlha::qsusy_numeric::ScanStepStatus::Success) {
                    result.status = static_cast<int>(CudaQSusyStatus::ScanSpacing);
                    break;
                }
                ++result.scanSegments;
                result.maxObservedDeltaLogQ = ::fmax(
                    result.maxObservedDeltaLogQ, scanHigh - scanLow);
                dopriDenseState(scanLow, scanLowState, oldState, k1,k3,k4,k5,k6,k7,
                                oldTime, time);
                auto lowStop = natlha::qsusy_numeric::evaluateStopPoint(
                    scanLowState, kRgeStateSize, scanLow,
                    natlha::qsusy_numeric::kMZ);
                unsigned events = 0;
                if (!natlha::qsusy_numeric::classifySegment(
                        highStop, lowStop, scanState, events)) {
                    result.status = static_cast<int>(
                        CudaQSusyStatus::BoundaryCounterOverflow);
                    break;
                }
                auto storeRoot = [&](double rootTime, const Real* rootState,
                                     const natlha::qsusy_numeric::StopPoint& rootStop,
                                     double refinedBracketWidth) {
                    if (result.rootsFound != 0 && storedRootScale == rootTime) return;
                    ++result.rootsFound;
                    if (result.rootsFound == 1) {
                        storedRootScale = rootTime;
                        result.logScale = rootTime;
                        result.residual = rootStop.logResidual;
                        result.stop1Squared = rootStop.stop1Squared;
                        result.stop2Squared = rootStop.stop2Squared;
                        result.refinedBracketWidth = refinedBracketWidth;
                        for (std::size_t i = 0; i < kRgeStateSize; ++i) {
                            result.stateAtRoot[i] = toDouble(rootState[i]);
                        }
                    }
                };
                if ((events & natlha::qsusy_numeric::ExactHigh) != 0)
                    storeRoot(scanHigh, scanHighState, highStop, 0.0);
                if ((events & natlha::qsusy_numeric::ExactLow) != 0)
                    storeRoot(scanLow, scanLowState, lowStop, 0.0);
                if ((events & natlha::qsusy_numeric::SignBracket) != 0) {
                    double rootLow = scanLow, rootHigh = scanHigh;
                    double lowResidual = lowStop.logResidual;
                    double highResidual = highStop.logResidual;
                    if (refineDenseBracket(rootLow, rootHigh, lowResidual, highResidual,
                            result.refinementEvaluations, result.status, scratch,
                            oldState,k1,k3,k4,k5,k6,k7,oldTime,time)) {
                        const double rootTime = rootLow + 0.5 * (rootHigh - rootLow);
                        double rootResidual = 0.0;
                        natlha::qsusy_numeric::StopPoint rootStop;
                        if (!denseResidual(rootTime, rootResidual, rootStop, scratch,
                                oldState,k1,k3,k4,k5,k6,k7,oldTime,time)) {
                            result.status = static_cast<int>(
                                CudaQSusyStatus::NonFiniteBoundary);
                        } else if (::fabs(rootResidual)
                                   > ::fmax(absoluteTolerance, relativeTolerance)) {
                            result.status = static_cast<int>(
                                CudaQSusyStatus::ResidualFailure);
                        } else {
                            storeRoot(
                                rootTime, scratch, rootStop,
                                rootHigh - rootLow);
                        }
                    }
                }
                scanHigh = scanLow;
                for (std::size_t i = 0; i < kRgeStateSize; ++i)
                    scanHighState[i] = scanLowState[i];
                highStop = lowStop;
            }
        }
        // A terminal scan/refinement status cannot recover on a later ODE segment.
        // Stop this trajectory immediately instead of evolving unused state to 500 GeV.
        if (result.status != static_cast<int>(CudaQSusyStatus::Success)) break;
        step *= factor;
    }
    result.invalidBoundaries = scanState.invalidBoundaries;
    result.nonFiniteBoundaries = scanState.nonFiniteBoundaries;
    if (result.status == static_cast<int>(CudaQSusyStatus::Success)) {
        if (result.nonFiniteBoundaries != 0) {
            result.status = static_cast<int>(CudaQSusyStatus::NonFiniteBoundary);
        } else if (result.rootsFound != 1) {
            result.status = static_cast<int>(CudaQSusyStatus::NonUniqueRoot);
        }
    }
    results[point] = result;
}

template <typename Real>
std::size_t automaticBatchSize(std::size_t points, int device) {
    requireCuda(cudaSetDevice(device), "cudaSetDevice for automatic batch sizing");
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    requireCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
    if (freeBytes > totalBytes) {
        throw std::runtime_error(
            "cudaMemGetInfo reported free memory larger than total memory");
    }
    cudaFuncAttributes attributes{};
    requireCuda(
        cudaFuncGetAttributes(&attributes, integrateKernel<Real>),
        "cudaFuncGetAttributes for automatic batch sizing");
    constexpr std::size_t explicitBytesPerPoint =
        kRgeStateSize * sizeof(double) + 3 * sizeof(double) + sizeof(int)
        + 2 * sizeof(unsigned long long);
    const std::size_t bytesPerPoint = explicitBytesPerPoint + attributes.localSizeBytes;
    // Leave 20% of currently free memory unclaimed for CUDA runtime bookkeeping and other
    // processes. Compiled per-thread local storage is accounted above rather than guessed.
    // The reserve is a conservative allocation policy, not a measured performance optimum.
    const std::size_t memoryBound =
        static_cast<std::size_t>(0.8 * static_cast<double>(freeBytes))
        / bytesPerPoint;
    return std::max<std::size_t>(1, std::min(points, memoryBound));
}

template <typename Real>
std::size_t automaticQSusyBatchSize(std::size_t points, int device) {
    requireCuda(cudaSetDevice(device), "cudaSetDevice for automatic Q_SUSY batch sizing");
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    requireCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo for Q_SUSY");
    if (freeBytes > totalBytes) {
        throw std::runtime_error(
            "cudaMemGetInfo reported free memory larger than total memory");
    }
    cudaFuncAttributes attributes{};
    requireCuda(
        cudaFuncGetAttributes(&attributes, qSusyKernel<Real>),
        "cudaFuncGetAttributes for automatic Q_SUSY batch sizing");
    constexpr std::size_t explicitBytesPerPoint =
        kRgeStateSize * sizeof(double) + 3 * sizeof(double)
        + sizeof(QSusyDeviceResult);
    const std::size_t bytesPerPoint = explicitBytesPerPoint + attributes.localSizeBytes;
    if (bytesPerPoint == 0) {
        throw std::runtime_error("CUDA Q_SUSY byte accounting produced zero bytes per point");
    }
    const std::size_t memoryBound =
        static_cast<std::size_t>(0.8 * static_cast<double>(freeBytes))
        / bytesPerPoint;
    return std::max<std::size_t>(1, std::min(points, memoryBound));
}

std::string describeCudaOdeStatus(CudaOdeStatus status) {
    switch (status) {
        case CudaOdeStatus::Success: return "success";
        case CudaOdeStatus::NonFiniteInput: return "non-finite input";
        case CudaOdeStatus::NonFiniteState: return "non-finite evolved state";
        case CudaOdeStatus::StepLimit: return "ODE attempt limit exhausted";
        case CudaOdeStatus::StepUnderflow: return "ODE step underflow";
    }
    return "unknown CUDA ODE status";
}

class CudaOdeFailure : public std::runtime_error {
public:
    explicit CudaOdeFailure(CudaOdeStatus status)
        : std::runtime_error("CUDA ODE solve failed: " + describeCudaOdeStatus(status)),
          reasons_(reasonsFor(status)) {}

    AdjudicationReasons reasons() const { return reasons_; }

private:
    static AdjudicationReasons reasonsFor(CudaOdeStatus status) {
        switch (status) {
            case CudaOdeStatus::NonFiniteInput:
            case CudaOdeStatus::NonFiniteState:
                return adjudicationReason(AdjudicationReason::NonFiniteState);
            case CudaOdeStatus::StepLimit:
                return adjudicationReason(AdjudicationReason::OdeStepLimit);
            case CudaOdeStatus::StepUnderflow:
                return adjudicationReason(AdjudicationReason::ErrorEstimate);
            case CudaOdeStatus::Success:
                return adjudicationReason(AdjudicationReason::None);
        }
        return adjudicationReason(AdjudicationReason::InfrastructureFailure);
    }

    AdjudicationReasons reasons_ = 0;
};

std::string describeCudaQSusyStatus(CudaQSusyStatus status) {
    switch (status) {
        case CudaQSusyStatus::Success: return "success";
        case CudaQSusyStatus::NonFiniteInput: return "non-finite input";
        case CudaQSusyStatus::NonFiniteState: return "non-finite evolved state";
        case CudaQSusyStatus::StepLimit: return "ODE attempt limit exhausted";
        case CudaQSusyStatus::StepUnderflow: return "ODE step underflow";
        case CudaQSusyStatus::ScanSpacing: return "invalid scan spacing";
        case CudaQSusyStatus::BoundaryCounterOverflow: return "boundary counter overflow";
        case CudaQSusyStatus::NonPhysicalBracket: return "non-physical root bracket";
        case CudaQSusyStatus::RefinementFailure: return "root refinement failed";
        case CudaQSusyStatus::ResidualFailure: return "refined root failed residual gate";
        case CudaQSusyStatus::NonUniqueRoot: return "root count is not exactly one";
        case CudaQSusyStatus::NonFiniteBoundary: return "non-finite root boundary";
    }
    return "unknown CUDA Q_SUSY status";
}

class CudaQSusyFailure : public std::runtime_error {
public:
    explicit CudaQSusyFailure(CudaQSusyStatus status)
        : std::runtime_error(
              "CUDA Q_SUSY search failed: " + describeCudaQSusyStatus(status)),
          reasons_(reasonsFor(status)) {}

    AdjudicationReasons reasons() const { return reasons_; }

private:
    static AdjudicationReasons reasonsFor(CudaQSusyStatus status) {
        return cudaQSusyAdjudicationReasons(status);
    }

    AdjudicationReasons reasons_ = 0;
};

class CudaOdeScheduler {
public:
    CudaOdeScheduler(int device, std::size_t launchLimit, bool doubleDouble)
        : device_(device),
          launchLimit_(launchLimit),
          doubleDouble_(doubleDouble),
          worker_([this] { run(); }) {}

    ~CudaOdeScheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        worker_.join();
    }

    std::vector<double> solve(
            std::vector<double> state,
            double start,
            double end,
            double initialStep) {
        auto request = std::make_shared<Request>();
        request->state = std::move(state);
        request->start = start;
        request->end = end;
        request->initialStep = initialStep;
        request->enqueuedAt = std::chrono::steady_clock::now();
        boost::fibers::future<std::vector<double>> result =
            request->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("CUDA ODE scheduler is stopping");
            queue_.push_back(std::move(request));
        }
        ready_.notify_one();
        return result.get();
    }

    std::size_t maximumObservedBatchSize() const {
        return maximumObservedBatchSize_.load();
    }

    CudaStageProfile profile() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return profile_;
    }

private:
    struct Request {
        std::vector<double> state;
        double start = 0.0;
        double end = 0.0;
        double initialStep = 0.0;
        std::chrono::steady_clock::time_point enqueuedAt;
        boost::fibers::promise<std::vector<double>> completion;
    };

    void run() {
        while (true) {
            std::vector<std::shared_ptr<Request>> requests;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;

                // A short coalescing window lets independently progressing point workers
                // rendezvous without imposing a millisecond-scale latency on a lone request.
                const auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::microseconds(200);
                while (!stopping_ && queue_.size() < launchLimit_) {
                    if (ready_.wait_until(lock, deadline) == std::cv_status::timeout) break;
                }
                const std::size_t count = std::min(launchLimit_, queue_.size());
                std::size_t observed = maximumObservedBatchSize_.load();
                while (observed < count
                        && !maximumObservedBatchSize_.compare_exchange_weak(
                            observed, count)) {
                }
                requests.reserve(count);
                const auto dequeuedAt = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < count; ++i) {
                    requests.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                    profile_.cumulativeQueueWaitSeconds +=
                        std::chrono::duration<double>(
                            dequeuedAt - requests.back()->enqueuedAt).count();
                }
            }

            try {
                CudaOdeBatch batch;
                batch.states.reserve(requests.size() * kRgeStateSize);
                batch.startTimes.reserve(requests.size());
                batch.endTimes.reserve(requests.size());
                batch.initialSteps.reserve(requests.size());
                for (const auto& request : requests) {
                    batch.states.insert(
                        batch.states.end(), request->state.begin(), request->state.end());
                    batch.startTimes.push_back(request->start);
                    batch.endTimes.push_back(request->end);
                    batch.initialSteps.push_back(request->initialStep);
                }
                CudaOdeBatchResult output = doubleDouble_
                    ? solveCudaOdeBatchDoubleDouble(batch, device_, requests.size())
                    : solveCudaOdeBatchFp64(batch, device_, requests.size());
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    profile_.requests += output.profile.requests;
                    profile_.launches += output.profile.launches;
                    profile_.trajectories += output.profile.trajectories;
                    profile_.allocationSeconds += output.profile.allocationSeconds;
                    profile_.hostToDeviceSeconds += output.profile.hostToDeviceSeconds;
                    profile_.kernelAndSyncSeconds += output.profile.kernelAndSyncSeconds;
                    profile_.deviceToHostSeconds += output.profile.deviceToHostSeconds;
                }
                for (std::size_t i = 0; i < requests.size(); ++i) {
                    if (output.statuses[i] != CudaOdeStatus::Success) {
                        requests[i]->completion.set_exception(
                            std::make_exception_ptr(CudaOdeFailure(output.statuses[i])));
                        continue;
                    }
                    const auto begin = output.states.begin()
                        + static_cast<std::ptrdiff_t>(i * kRgeStateSize);
                    requests[i]->completion.set_value(
                        std::vector<double>(begin, begin + kRgeStateSize));
                }
            } catch (...) {
                const std::exception_ptr failure = std::current_exception();
                for (const auto& request : requests) {
                    try {
                        request->completion.set_exception(failure);
                    } catch (...) {
                        // A promise completed above must not hide the original batch failure
                        // from the requests that have not yet received a result.
                    }
                }
            }
        }
    }

    int device_ = 0;
    std::size_t launchLimit_ = 1;
    std::atomic<std::size_t> maximumObservedBatchSize_{0};
    bool doubleDouble_ = false;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Request>> queue_;
    bool stopping_ = false;
    std::thread worker_;
    CudaStageProfile profile_;
};

class CudaQSusyScheduler {
public:
    CudaQSusyScheduler(int device, std::size_t launchLimit, bool doubleDouble)
        : device_(device),
          launchLimit_(launchLimit),
          doubleDouble_(doubleDouble),
          worker_([this] { run(); }) {}

    ~CudaQSusyScheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        worker_.join();
    }

    QSusyResult solve(
            const std::vector<double>& state,
            double highLogScale,
            double initialStep,
            double maxDeltaLogQ) {
        auto request = std::make_shared<Request>();
        request->state = state;
        request->highLogScale = highLogScale;
        request->initialStep = initialStep;
        request->maxDeltaLogQ = maxDeltaLogQ;
        request->enqueuedAt = std::chrono::steady_clock::now();
        boost::fibers::future<QSusyResult> result = request->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("CUDA Q_SUSY scheduler is stopping");
            queue_.push_back(std::move(request));
        }
        ready_.notify_one();
        return result.get();
    }

    std::size_t maximumObservedBatchSize() const {
        return maximumObservedBatchSize_.load();
    }

    CudaStageProfile profile() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return profile_;
    }

private:
    struct Request {
        std::vector<double> state;
        double highLogScale = 0.0;
        double initialStep = 0.0;
        double maxDeltaLogQ = 0.0;
        std::chrono::steady_clock::time_point enqueuedAt;
        boost::fibers::promise<QSusyResult> completion;
    };

    void run() {
        while (true) {
            std::vector<std::shared_ptr<Request>> requests;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                const auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::microseconds(200);
                while (!stopping_ && queue_.size() < launchLimit_) {
                    if (ready_.wait_until(lock, deadline) == std::cv_status::timeout) break;
                }
                const std::size_t count = std::min(launchLimit_, queue_.size());
                std::size_t observed = maximumObservedBatchSize_.load();
                while (observed < count
                        && !maximumObservedBatchSize_.compare_exchange_weak(
                            observed, count)) {
                }
                requests.reserve(count);
                const auto dequeuedAt = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < count; ++i) {
                    requests.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                    profile_.cumulativeQueueWaitSeconds +=
                        std::chrono::duration<double>(
                            dequeuedAt - requests.back()->enqueuedAt).count();
                }
            }

            try {
                CudaQSusyBatch batch;
                batch.states.reserve(requests.size() * kRgeStateSize);
                batch.highLogScales.reserve(requests.size());
                batch.initialSteps.reserve(requests.size());
                batch.maxDeltaLogQ.reserve(requests.size());
                for (const auto& request : requests) {
                    batch.states.insert(
                        batch.states.end(), request->state.begin(), request->state.end());
                    batch.highLogScales.push_back(request->highLogScale);
                    batch.initialSteps.push_back(request->initialStep);
                    batch.maxDeltaLogQ.push_back(request->maxDeltaLogQ);
                }
                CudaQSusyBatchResult output = doubleDouble_
                    ? solveCudaQSusyBatchDoubleDouble(batch, device_, requests.size())
                    : solveCudaQSusyBatchFp64(batch, device_, requests.size());
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    profile_.requests += output.profile.requests;
                    profile_.launches += output.profile.launches;
                    profile_.trajectories += output.profile.trajectories;
                    profile_.allocationSeconds += output.profile.allocationSeconds;
                    profile_.hostToDeviceSeconds += output.profile.hostToDeviceSeconds;
                    profile_.kernelAndSyncSeconds += output.profile.kernelAndSyncSeconds;
                    profile_.deviceToHostSeconds += output.profile.deviceToHostSeconds;
                }
                for (std::size_t i = 0; i < requests.size(); ++i) {
                    if (output.statuses[i] != CudaQSusyStatus::Success) {
                        requests[i]->completion.set_exception(std::make_exception_ptr(
                            CudaQSusyFailure(output.statuses[i])));
                        continue;
                    }
                    QSusyResult result;
                    const auto begin = output.statesAtRoot.begin()
                        + static_cast<std::ptrdiff_t>(i * kRgeStateSize);
                    result.stateAtRoot.assign(begin, begin + kRgeStateSize);
                    result.logScale = output.logScales[i];
                    result.scale = std::exp(result.logScale);
                    result.residual = output.residuals[i];
                    result.stop1Squared = output.stop1Squared[i];
                    result.stop2Squared = output.stop2Squared[i];
                    result.refinedBracketWidth = output.refinedBracketWidths[i];
                    result.acceptedSteps = static_cast<std::size_t>(output.acceptedSteps[i]);
                    result.declaredMaxDeltaLogQ = requests[i]->maxDeltaLogQ;
                    result.scanSegments = static_cast<std::size_t>(output.scanSegments[i]);
                    result.maxObservedDeltaLogQ = output.maxObservedDeltaLogQ[i];
                    result.rootsFound = static_cast<std::size_t>(output.rootsFound[i]);
                    result.invalidBoundaries =
                        static_cast<std::size_t>(output.invalidBoundaries[i]);
                    result.refinementEvaluations =
                        static_cast<std::size_t>(output.refinementEvaluations[i]);
                    result.diagnostic =
                        "one positive-stop sign-changing or exact root at the declared "
                        "scan spacing (CUDA candidate)";
                    if (!std::isfinite(result.scale) || result.scale <= 0.0) {
                        requests[i]->completion.set_exception(std::make_exception_ptr(
                            CudaQSusyFailure(CudaQSusyStatus::NonFiniteBoundary)));
                        continue;
                    }
                    requests[i]->completion.set_value(std::move(result));
                }
            } catch (...) {
                const std::exception_ptr failure = std::current_exception();
                for (const auto& request : requests) {
                    try {
                        request->completion.set_exception(failure);
                    } catch (...) {
                    }
                }
            }
        }
    }

    int device_ = 0;
    std::size_t launchLimit_ = 1;
    std::atomic<std::size_t> maximumObservedBatchSize_{0};
    bool doubleDouble_ = false;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Request>> queue_;
    bool stopping_ = false;
    std::thread worker_;
    CudaStageProfile profile_;
};

bool closeValue(const high_prec_float& candidate, const high_prec_float& reference) {
    const high_prec_float scale = max(
        high_prec_float(1), max(abs(candidate), abs(reference)));
    // This cross-backend gate permits small accumulated host/device rounding. Success,
    // label, ordinal, and ordering comparisons remain exact and separate from it.
    constexpr const char* kCrossBackendRelativeTolerance = "2e-8";
    return abs(candidate - reference)
        <= high_prec_float(kCrossBackendRelativeTolerance) * scale;
}

bool closeDbgValue(
        const high_prec_float& candidate,
        const high_prec_float& reference,
        const high_prec_float& propagatedRootWidth = 0) {
    const high_prec_float scale = max(
        high_prec_float(1), max(abs(candidate), abs(reference)));
    // Compare DBG values at 0.05% of the local unit-floored scale unless propagated root
    // uncertainty is larger. Success, count, label, ordinal, and ordering comparisons
    // remain exact and separate from this numeric comparison.
    const high_prec_float allowed = max(
        high_prec_float("5e-4") * scale, propagatedRootWidth);
    return abs(candidate - reference) <= allowed;
}

template <typename Contribution>
std::string contributionMismatch(
        const char* measure,
        const std::vector<Contribution>& candidate,
        const std::vector<Contribution>& reference) {
    if (candidate.size() != reference.size()) {
        return std::string(measure) + " contribution count differs (candidate "
            + std::to_string(candidate.size()) + ", CPU "
            + std::to_string(reference.size()) + ")";
    }
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        if (candidate[i].label != reference[i].label) {
            return std::string(measure) + " label differs at index "
                + std::to_string(i) + " (candidate '" + candidate[i].label
                + "', CPU '" + reference[i].label + "')";
        }
        if (!closeValue(candidate[i].value, reference[i].value)) {
            return std::string(measure) + " value differs at index "
                + std::to_string(i) + " ('" + candidate[i].label
                + "', candidate " + candidate[i].value.str(17, std::ios_base::scientific)
                + ", CPU " + reference[i].value.str(17, std::ios_base::scientific)
                + ")";
        }
    }
    return {};
}

std::string dbgContributionMismatch(
        const std::vector<LabeledValueBG>& candidate,
        const std::vector<LabeledValueBG>& reference) {
    if (candidate.size() != reference.size()) {
        return "Delta_BG contribution count differs (candidate "
            + std::to_string(candidate.size()) + ", CPU "
            + std::to_string(reference.size()) + ")";
    }
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        if (candidate[i].label != reference[i].label
                || candidate[i].ordinal != reference[i].ordinal) {
            return "Delta_BG label/ordinal differs at index " + std::to_string(i)
                + " (candidate '" + candidate[i].label + "' ordinal "
                + std::to_string(candidate[i].ordinal) + ", CPU '"
                + reference[i].label + "' ordinal "
                + std::to_string(reference[i].ordinal) + ")";
        }
        const high_prec_float propagatedRootWidth =
            candidate[i].rootUncertainty + reference[i].rootUncertainty;
        if (!closeDbgValue(
                candidate[i].value, reference[i].value, propagatedRootWidth)) {
            const high_prec_float scale = max(
                high_prec_float(1),
                max(abs(candidate[i].value), abs(reference[i].value)));
            const high_prec_float allowed = max(
                high_prec_float("5e-4") * scale, propagatedRootWidth);
            return "Delta_BG value differs at index " + std::to_string(i)
                + " ('" + candidate[i].label + "', candidate "
                + candidate[i].value.str(17, std::ios_base::scientific)
                + ", CPU " + reference[i].value.str(17, std::ios_base::scientific)
                + ", allowed " + allowed.str(17, std::ios_base::scientific) + ")";
        }
    }
    return {};
}

std::string labelResultMismatch(const Result& candidate, const Result& reference) {
    if (candidate.ok != reference.ok
            || candidate.haveDEW != reference.haveDEW
            || candidate.haveDHS != reference.haveDHS
            || candidate.haveDBG != reference.haveDBG
            || candidate.haveDSN != reference.haveDSN) {
        return "result success or requested-measure branch differs";
    }
    if (!candidate.ok) {
        return candidate.error == reference.error ? std::string{}
                                                   : "failure diagnostic differs";
    }
    if (!closeValue(candidate.qSusy, reference.qSusy)) return "Q_SUSY differs";
    if (!closeValue(candidate.logQGut, reference.logQGut)) return "log(Q_GUT) differs";
    if (!closeValue(candidate.mZ2, reference.mZ2)) return "mZ^2 differs";
    if (!closeValue(candidate.deltaEW, reference.deltaEW)) return "Delta_EW differs";
    if (!closeValue(candidate.deltaHS, reference.deltaHS)) return "Delta_HS differs";
    if (!closeDbgValue(candidate.deltaBG, reference.deltaBG)) return "Delta_BG differs";
    if (!closeValue(candidate.deltaSN, reference.deltaSN)) return "delta_SN differs";
    for (const auto& mismatch : {
             contributionMismatch(
                 "Delta_EW", candidate.dewContributions, reference.dewContributions),
             contributionMismatch(
                 "Delta_HS", candidate.dhsContributions, reference.dhsContributions),
             dbgContributionMismatch(candidate.dbgContributions, reference.dbgContributions),
             contributionMismatch(
                 "delta_SN", candidate.dsnContributions, reference.dsnContributions)}) {
        if (!mismatch.empty()) return mismatch;
    }
    return {};
}

bool sameLabelResult(const Result& candidate, const Result& reference) {
    return labelResultMismatch(candidate, reference).empty();
}

bool nearQSusyBoundary(const Result& candidate) {
    if (!candidate.ok) return false;
    const double residualLimit =
        0.5 * std::max(odeTolerances().absolute, odeTolerances().relative);
    if (std::abs(static_cast<double>(candidate.qSusyResidual)) > residualLimit) return true;
    return !std::isfinite(candidate.qSusyRootBracketWidth)
        || candidate.qSusyRootBracketWidth > residualLimit;
}

bool nearBranchBoundary(const Result& candidate) {
    if (!candidate.ok) return true;
    if (candidate.haveDBG
            && (candidate.dbgHeadline.headlineSignFragileRootUncertainty
                || (candidate.dbgHeadline.headlineMagnitudeGap >= 0
                    && candidate.dbgHeadline.headlineMagnitudeGap
                        <= candidate.dbgHeadline.topRootUncertainty
                            + candidate.dbgHeadline.secondRootUncertainty))) {
        return true;
    }
    for (std::size_t index = 1; index < candidate.dbgContributions.size(); ++index) {
        const LabeledValueBG& previous = candidate.dbgContributions[index - 1];
        const LabeledValueBG& current = candidate.dbgContributions[index];
        const high_prec_float scale = max(
            high_prec_float(1), max(abs(previous.value), abs(current.value)));
        const high_prec_float magnitudeGap =
            abs(abs(previous.value) - abs(current.value));
        const high_prec_float orderingRisk = max(
            previous.rootUncertainty + current.rootUncertainty,
            high_prec_float("1e-4") * scale);
        // Exact contribution order is part of the CPU semantic contract. Retry any adjacent
        // pair whose magnitude gap is small enough that GPU arithmetic or root uncertainty
        // could reverse it, regardless of which Delta_BG model produced the pair.
        if (magnitudeGap <= orderingRisk) return true;
    }
    for (const auto& direction : candidate.dbgDiagnostics) {
        if (!direction.accepted || !direction.failure.empty()) return true;
        for (const auto& window : direction.windows) {
            if (!window.accepted && window.failure.empty()) return true;
            if (window.accepted && window.agreementTolerance > 0
                    && (window.contributionSpan
                            > high_prec_float("0.5") * window.agreementTolerance
                        || (!window.rootUncertainties.empty()
                            && *std::max_element(
                                window.rootUncertainties.begin(),
                                window.rootUncertainties.end())
                                > high_prec_float("0.5")
                                    * window.agreementTolerance))) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

CudaDeviceInfo queryCudaDevice(int device) {
    CudaDeviceInfo info;
    info.compiled = true;
    info.device = device;
    int count = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&count);
    if (countStatus != cudaSuccess) {
        info.diagnostic = std::string("cudaGetDeviceCount: ")
            + cudaGetErrorString(countStatus);
        cudaGetLastError();
        return info;
    }
    if (device < 0 || device >= count) {
        info.diagnostic = "CUDA device ordinal " + std::to_string(device)
            + " is outside the available range [0, " + std::to_string(count) + ")";
        return info;
    }

    cudaDeviceProp properties{};
    const cudaError_t propertyStatus = cudaGetDeviceProperties(&properties, device);
    if (propertyStatus != cudaSuccess) {
        info.diagnostic = std::string("cudaGetDeviceProperties: ")
            + cudaGetErrorString(propertyStatus);
        return info;
    }
    info.available = true;
    info.computeCapabilityMajor = properties.major;
    info.computeCapabilityMinor = properties.minor;
    info.multiprocessorCount = properties.multiProcessorCount;
    info.totalMemoryBytes = properties.totalGlobalMem;
    info.name = properties.name;
    info.diagnostic = "available";
    return info;
}

template <typename Real>
CudaOdeBatchResult solveCudaOdeBatchTyped(
        const CudaOdeBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    const std::size_t points = batch.startTimes.size();
    if (batch.endTimes.size() != points || batch.initialSteps.size() != points
            || batch.states.size() != points * kRgeStateSize) {
        throw std::invalid_argument(
            "CUDA ODE batch arrays do not describe the same number of 44-state points");
    }

    CudaOdeBatchResult result;
    result.states = batch.states;
    result.statuses.resize(points);
    result.acceptedSteps.resize(points);
    result.rejectedSteps.resize(points);
    if (points == 0) return result;

    requireCuda(cudaSetDevice(device), "cudaSetDevice");
    const ODETolerances& tolerances = odeTolerances();
    const std::size_t chunkLimit = requestedBatchSize == 0
        ? automaticBatchSize<Real>(points, device) : requestedBatchSize;
    if (chunkLimit == 0) {
        throw std::invalid_argument("CUDA ODE batch size must be positive or zero for auto");
    }

    const std::size_t capacity = std::min(points, chunkLimit);
    const auto allocationStarted = std::chrono::steady_clock::now();
    DeviceBuffer<double> deviceStates(capacity * kRgeStateSize);
    DeviceBuffer<double> deviceStarts(capacity);
    DeviceBuffer<double> deviceEnds(capacity);
    DeviceBuffer<double> deviceInitialSteps(capacity);
    DeviceBuffer<int> deviceStatuses(capacity);
    DeviceBuffer<unsigned long long> deviceAccepted(capacity);
    DeviceBuffer<unsigned long long> deviceRejected(capacity);
    result.profile.requests = points;
    result.profile.trajectories = points;
    result.profile.allocationSeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - allocationStarted).count();
    for (std::size_t offset = 0; offset < points; offset += chunkLimit) {
        const std::size_t count = std::min(chunkLimit, points - offset);

        const auto hostToDeviceStarted = std::chrono::steady_clock::now();
        requireCuda(cudaMemcpy(
            deviceStates.get(), result.states.data() + offset * kRgeStateSize,
            count * kRgeStateSize * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA ODE states to device");
        requireCuda(cudaMemcpy(
            deviceStarts.get(), batch.startTimes.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA ODE start times to device");
        requireCuda(cudaMemcpy(
            deviceEnds.get(), batch.endTimes.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA ODE end times to device");
        requireCuda(cudaMemcpy(
            deviceInitialSteps.get(), batch.initialSteps.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA ODE initial steps to device");
        result.profile.hostToDeviceSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - hostToDeviceStarted).count();

        const auto kernelStarted = std::chrono::steady_clock::now();
        const int blocks = static_cast<int>((count + kThreadsPerBlock - 1)
                                            / kThreadsPerBlock);
        integrateKernel<Real><<<blocks, kThreadsPerBlock>>>(
            deviceStates.get(), deviceStarts.get(), deviceEnds.get(),
            deviceInitialSteps.get(), deviceStatuses.get(), deviceAccepted.get(),
            deviceRejected.get(), count, tolerances.absolute, tolerances.relative);
        const char* tierName = std::is_same<Real, DoubleDouble>::value
            ? "double-double" : "FP64";
        requireCuda(
            cudaGetLastError(), std::string("launch ") + tierName + " CUDA ODE kernel");
        requireCuda(
            cudaDeviceSynchronize(),
            std::string("synchronize ") + tierName + " CUDA ODE kernel");
        ++result.profile.launches;
        result.profile.kernelAndSyncSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - kernelStarted).count();

        const auto deviceToHostStarted = std::chrono::steady_clock::now();
        requireCuda(cudaMemcpy(
            result.states.data() + offset * kRgeStateSize, deviceStates.get(),
            count * kRgeStateSize * sizeof(double), cudaMemcpyDeviceToHost),
            "copy CUDA ODE states to host");
        requireCuda(cudaMemcpy(
            result.statuses.data() + offset, deviceStatuses.get(),
            count * sizeof(CudaOdeStatus), cudaMemcpyDeviceToHost),
            "copy CUDA ODE statuses to host");
        requireCuda(cudaMemcpy(
            result.acceptedSteps.data() + offset, deviceAccepted.get(),
            count * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
            "copy CUDA ODE accepted-step counts to host");
        requireCuda(cudaMemcpy(
            result.rejectedSteps.data() + offset, deviceRejected.get(),
            count * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
            "copy CUDA ODE rejected-step counts to host");
        result.profile.deviceToHostSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - deviceToHostStarted).count();
    }
    return result;
}

CudaOdeBatchResult solveCudaOdeBatchFp64(
        const CudaOdeBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    return solveCudaOdeBatchTyped<double>(batch, device, requestedBatchSize);
}

CudaOdeBatchResult solveCudaOdeBatchDoubleDouble(
        const CudaOdeBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    return solveCudaOdeBatchTyped<DoubleDouble>(batch, device, requestedBatchSize);
}

std::vector<CudaQSusyHelperResult> evaluateCudaQSusyHelpers(
        const std::vector<CudaQSusyHelperInput>& inputs,
        int device) {
    requireCuda(cudaSetDevice(device), "cudaSetDevice for Q_SUSY helper validation");
    std::vector<QSusyHelperDeviceInput> hostInputs(inputs.size());
    for (std::size_t point = 0; point < inputs.size(); ++point) {
        if (inputs[point].state.size() != kRgeStateSize) {
            throw std::invalid_argument(
                "CUDA Q_SUSY helper input must contain exactly 44 state values");
        }
        QSusyHelperDeviceInput& output = hostInputs[point];
        std::copy(inputs[point].state.begin(), inputs[point].state.end(), output.state);
        output.logScale = inputs[point].logScale;
        output.currentLogScale = inputs[point].currentLogScale;
        output.lowerLogScale = inputs[point].lowerLogScale;
        output.maxDeltaLogQ = inputs[point].maxDeltaLogQ;
        const auto copyPoint = [](const StopScalePoint& source) {
            natlha::qsusy_numeric::StopPoint target;
            target.numericallyValid = source.numericallyValid;
            target.physical = source.physical;
            target.stop1Squared = source.stop1Squared;
            target.stop2Squared = source.stop2Squared;
            target.logResidual = source.logResidual;
            return target;
        };
        output.highPoint = copyPoint(inputs[point].highPoint);
        output.lowPoint = copyPoint(inputs[point].lowPoint);
        output.scanState.inInvalidDomain = inputs[point].inInvalidDomain;
        output.scanState.inNonFiniteDomain = inputs[point].inNonFiniteDomain;
        output.scanState.invalidBoundaries = inputs[point].invalidBoundaries;
        output.scanState.nonFiniteBoundaries = inputs[point].nonFiniteBoundaries;
    }

    DeviceBuffer<QSusyHelperDeviceInput> deviceInputs(inputs.size());
    DeviceBuffer<QSusyHelperDeviceResult> deviceResults(inputs.size());
    if (!inputs.empty()) {
        requireCuda(cudaMemcpy(
            deviceInputs.get(), hostInputs.data(),
            hostInputs.size() * sizeof(QSusyHelperDeviceInput),
            cudaMemcpyHostToDevice), "copy Q_SUSY helper inputs to CUDA");
        const int blocks = static_cast<int>(
            (inputs.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
        qSusyHelperKernel<<<blocks, kThreadsPerBlock>>>(
            deviceInputs.get(), deviceResults.get(), inputs.size());
        requireCuda(cudaGetLastError(), "launch Q_SUSY helper validation kernel");
        requireCuda(cudaDeviceSynchronize(), "synchronize Q_SUSY helper validation kernel");
    }
    std::vector<QSusyHelperDeviceResult> hostResults(inputs.size());
    if (!inputs.empty()) {
        requireCuda(cudaMemcpy(
            hostResults.data(), deviceResults.get(),
            hostResults.size() * sizeof(QSusyHelperDeviceResult),
            cudaMemcpyDeviceToHost), "copy Q_SUSY helper results from CUDA");
    }

    std::vector<CudaQSusyHelperResult> results(inputs.size());
    for (std::size_t point = 0; point < inputs.size(); ++point) {
        const QSusyHelperDeviceResult& source = hostResults[point];
        CudaQSusyHelperResult& target = results[point];
        target.evaluatedPoint.numericallyValid = source.evaluatedPoint.numericallyValid;
        target.evaluatedPoint.physical = source.evaluatedPoint.physical;
        target.evaluatedPoint.stop1Squared = source.evaluatedPoint.stop1Squared;
        target.evaluatedPoint.stop2Squared = source.evaluatedPoint.stop2Squared;
        target.evaluatedPoint.logResidual = source.evaluatedPoint.logResidual;
        target.nextLogScale = source.nextLogScale;
        target.scanStepStatus = source.scanStepStatus;
        target.scanEvents = source.scanEvents;
        target.classificationOk = source.classificationOk;
        target.inInvalidDomain = source.scanState.inInvalidDomain;
        target.inNonFiniteDomain = source.scanState.inNonFiniteDomain;
        target.invalidBoundaries = source.scanState.invalidBoundaries;
        target.nonFiniteBoundaries = source.scanState.nonFiniteBoundaries;
    }
    return results;
}

template <typename Real>
CudaQSusyBatchResult solveCudaQSusyBatchTyped(
        const CudaQSusyBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    const std::size_t points = batch.highLogScales.size();
    if (batch.states.size() != points * kRgeStateSize
            || batch.initialSteps.size() != points
            || batch.maxDeltaLogQ.size() != points) {
        throw std::invalid_argument("CUDA Q_SUSY batch arrays are not aligned");
    }
    requireCuda(cudaSetDevice(device), "cudaSetDevice for CUDA Q_SUSY batch");
    CudaQSusyBatchResult result;
    result.statesAtRoot.resize(points * kRgeStateSize);
    result.logScales.resize(points);
    result.residuals.resize(points);
    result.stop1Squared.resize(points);
    result.stop2Squared.resize(points);
    result.refinedBracketWidths.resize(points);
    result.statuses.resize(points);
    result.acceptedSteps.resize(points);
    result.rejectedSteps.resize(points);
    result.scanSegments.resize(points);
    result.maxObservedDeltaLogQ.resize(points);
    result.rootsFound.resize(points);
    result.invalidBoundaries.resize(points);
    result.nonFiniteBoundaries.resize(points);
    result.refinementEvaluations.resize(points);
    if (points == 0) return result;

    const std::size_t launchLimit = requestedBatchSize == 0
        ? automaticQSusyBatchSize<Real>(points, device) : requestedBatchSize;
    if (launchLimit == 0) throw std::invalid_argument("CUDA Q_SUSY launch limit is zero");
    const ODETolerances& tolerances = odeTolerances();
    const std::size_t capacity = std::min(points, launchLimit);
    const auto allocationStarted = std::chrono::steady_clock::now();
    DeviceBuffer<double> deviceStates(capacity * kRgeStateSize);
    DeviceBuffer<double> deviceHigh(capacity);
    DeviceBuffer<double> deviceSteps(capacity);
    DeviceBuffer<double> deviceSpacing(capacity);
    DeviceBuffer<QSusyDeviceResult> deviceResults(capacity);
    result.profile.requests = points;
    result.profile.trajectories = points;
    result.profile.allocationSeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - allocationStarted).count();
    for (std::size_t offset = 0; offset < points; offset += launchLimit) {
        const std::size_t count = std::min(launchLimit, points - offset);
        const auto hostToDeviceStarted = std::chrono::steady_clock::now();
        requireCuda(cudaMemcpy(
            deviceStates.get(), batch.states.data() + offset * kRgeStateSize,
            count * kRgeStateSize * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA Q_SUSY states to device");
        requireCuda(cudaMemcpy(
            deviceHigh.get(), batch.highLogScales.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA Q_SUSY high scales to device");
        requireCuda(cudaMemcpy(
            deviceSteps.get(), batch.initialSteps.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA Q_SUSY steps to device");
        requireCuda(cudaMemcpy(
            deviceSpacing.get(), batch.maxDeltaLogQ.data() + offset,
            count * sizeof(double), cudaMemcpyHostToDevice),
            "copy CUDA Q_SUSY scan spacing to device");
        result.profile.hostToDeviceSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - hostToDeviceStarted).count();
        const auto kernelStarted = std::chrono::steady_clock::now();
        const int blocks = static_cast<int>(
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        qSusyKernel<Real><<<blocks, kThreadsPerBlock>>>(
            deviceStates.get(), deviceHigh.get(), deviceSteps.get(),
            deviceSpacing.get(), deviceResults.get(), count,
            tolerances.absolute, tolerances.relative);
        const char* tier = std::is_same<Real, DoubleDouble>::value
            ? "double-double" : "FP64";
        requireCuda(
            cudaGetLastError(), std::string("launch CUDA Q_SUSY ") + tier + " kernel");
        requireCuda(
            cudaDeviceSynchronize(),
            std::string("synchronize CUDA Q_SUSY ") + tier + " kernel");
        ++result.profile.launches;
        result.profile.kernelAndSyncSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - kernelStarted).count();
        const auto deviceToHostStarted = std::chrono::steady_clock::now();
        std::vector<QSusyDeviceResult> hostResults(count);
        requireCuda(cudaMemcpy(
            hostResults.data(), deviceResults.get(),
            count * sizeof(QSusyDeviceResult), cudaMemcpyDeviceToHost),
            "copy CUDA Q_SUSY results to host");
        result.profile.deviceToHostSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - deviceToHostStarted).count();
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t output = offset + local;
            const QSusyDeviceResult& source = hostResults[local];
            std::copy(
                source.stateAtRoot, source.stateAtRoot + kRgeStateSize,
                result.statesAtRoot.begin()
                    + static_cast<std::ptrdiff_t>(output * kRgeStateSize));
            result.logScales[output] = source.logScale;
            result.residuals[output] = source.residual;
            result.stop1Squared[output] = source.stop1Squared;
            result.stop2Squared[output] = source.stop2Squared;
            result.refinedBracketWidths[output] = source.refinedBracketWidth;
            result.statuses[output] = static_cast<CudaQSusyStatus>(source.status);
            result.acceptedSteps[output] = source.acceptedSteps;
            result.rejectedSteps[output] = source.rejectedSteps;
            result.scanSegments[output] = source.scanSegments;
            result.maxObservedDeltaLogQ[output] = source.maxObservedDeltaLogQ;
            result.rootsFound[output] = source.rootsFound;
            result.invalidBoundaries[output] = source.invalidBoundaries;
            result.nonFiniteBoundaries[output] = source.nonFiniteBoundaries;
            result.refinementEvaluations[output] = source.refinementEvaluations;
        }
    }
    return result;
}

CudaQSusyBatchResult solveCudaQSusyBatchFp64(
        const CudaQSusyBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    return solveCudaQSusyBatchTyped<double>(batch, device, requestedBatchSize);
}

CudaQSusyBatchResult solveCudaQSusyBatchDoubleDouble(
        const CudaQSusyBatch& batch,
        int device,
        std::size_t requestedBatchSize) {
    return solveCudaQSusyBatchTyped<DoubleDouble>(batch, device, requestedBatchSize);
}

BatchRun evaluateCudaBatch(
        const std::vector<Config>& configs,
        const BatchOptions& options,
        const CudaDeviceInfo& device) {
    BatchRun run;
    run.summary.points = configs.size();
    run.results.resize(configs.size());
    run.diagnostics.resize(configs.size());
    if (configs.empty()) return run;

    // Point state machines run as fibers. A blocked numerical submission yields its host
    // thread, so thousands of trajectories can rendezvous at a device stage without creating
    // thousands of OS threads. Waves bound live fibers and their stacks independently of the
    // input population size.
    const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    constexpr std::size_t kMaximumLivePointFibers = 4096;
    const std::size_t automaticWorkers =
        std::min<std::size_t>(configs.size(), kMaximumLivePointFibers);
    const std::size_t availableWorkers =
        options.cudaWorkers == 0
            ? automaticWorkers
            : std::min(options.cudaWorkers, kMaximumLivePointFibers);
    struct EvaluationPassProfile {
        std::size_t launchLimit = 0;
        std::size_t maximumLaunch = 0;
        CudaStageProfile rge;
        CudaStageProfile qSusy;
    };
    const auto addStageProfile = [](CudaStageProfile& destination,
                                    const CudaStageProfile& source) {
        destination.requests += source.requests;
        destination.launches += source.launches;
        destination.trajectories += source.trajectories;
        destination.cumulativeQueueWaitSeconds += source.cumulativeQueueWaitSeconds;
        destination.allocationSeconds += source.allocationSeconds;
        destination.hostToDeviceSeconds += source.hostToDeviceSeconds;
        destination.kernelAndSyncSeconds += source.kernelAndSyncSeconds;
        destination.deviceToHostSeconds += source.deviceToHostSeconds;
    };
    const auto evaluateIndices = [&](const std::vector<std::size_t>& indices,
                                     bool doubleDouble,
                                     std::vector<Result>& destination) {
        if (indices.empty()) return EvaluationPassProfile{};
        const std::size_t launchLimit = options.cudaBatchSize == 0
            ? (doubleDouble
                ? automaticBatchSize<DoubleDouble>(indices.size(), device.device)
                : automaticBatchSize<double>(indices.size(), device.device))
            : options.cudaBatchSize;
        const std::size_t qSusyLaunchLimit = options.cudaBatchSize == 0
            ? (doubleDouble
                ? automaticQSusyBatchSize<DoubleDouble>(indices.size(), device.device)
                : automaticQSusyBatchSize<double>(indices.size(), device.device))
            : options.cudaBatchSize;
        CudaOdeScheduler scheduler(device.device, launchLimit, doubleDouble);
        CudaQSusyScheduler qSusyScheduler(
            device.device, qSusyLaunchLimit, doubleDouble);
        const std::size_t liveFiberLimit =
            std::min<std::size_t>(indices.size(), availableWorkers);
        const std::size_t hostWorkerCount = std::min<std::size_t>(
            liveFiberLimit, static_cast<std::size_t>(hardwareThreads));
        const auto evaluatePoint = [&](std::size_t point) {
                    AdjudicationReasons cudaFailureReasons = 0;
                    const OdeSubmitFunction solveOnCuda = [&](std::vector<double> state,
                                                               double start,
                                                               double end,
                                                               double initialStep) {
                        try {
                            return scheduler.solve(
                                std::move(state), start, end, initialStep);
                        } catch (const CudaOdeFailure& failure) {
                            cudaFailureReasons |= failure.reasons();
                            throw;
                        } catch (...) {
                            cudaFailureReasons |= adjudicationReason(
                                AdjudicationReason::InfrastructureFailure);
                            throw;
                        }
                    };
                    const QSusySubmitFunction findOnCuda = [&] (
                            const std::vector<double>& state,
                            double highLogScale,
                            double initialStep,
                            double maxDeltaLogQ) {
                        try {
                            return qSusyScheduler.solve(
                                state, highLogScale, initialStep, maxDeltaLogQ);
                        } catch (const CudaQSusyFailure& failure) {
                            cudaFailureReasons |= failure.reasons();
                            throw;
                        } catch (...) {
                            cudaFailureReasons |= adjudicationReason(
                                AdjudicationReason::InfrastructureFailure);
                            throw;
                        }
                    };
                    const CudaExecutionContext executionContext{
                        &solveOnCuda, &findOnCuda};
                    const ScopedCudaExecutionContext contextScope(executionContext);
                    try {
                        destination[point] = evaluate(configs[point]);
                        recordCudaPointResult(run.diagnostics[point], doubleDouble);
                    } catch (const std::exception& failure) {
                        // evaluate() has a public never-throw contract and its own catch-all.
                        destination[point] = Result{};
                        destination[point].error =
                            std::string("exception escaped natlha::evaluate: ")
                            + failure.what();
                        PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
                        recordCudaPointEscape(diagnostic, doubleDouble);
                        diagnostic.adjudicationReasons |= adjudicationReason(
                            AdjudicationReason::InfrastructureFailure);
                        diagnostic.detail += "; " + destination[point].error;
                    } catch (...) {
                        destination[point] = Result{};
                        destination[point].error =
                            "unknown exception escaped natlha::evaluate";
                        PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
                        recordCudaPointEscape(diagnostic, doubleDouble);
                        diagnostic.adjudicationReasons |= adjudicationReason(
                            AdjudicationReason::InfrastructureFailure);
                        diagnostic.detail += "; " + destination[point].error;
                    }
                    run.diagnostics[point].adjudicationReasons |= cudaFailureReasons;
        };

        for (std::size_t wave = 0; wave < indices.size(); wave += liveFiberLimit) {
            const std::size_t waveSize = std::min(liveFiberLimit, indices.size() - wave);
            const std::size_t waveHostWorkers = std::min(hostWorkerCount, waveSize);
            std::mutex failureMutex;
            std::exception_ptr fiberInfrastructureFailure;
            std::vector<std::thread> workers;
            workers.reserve(waveHostWorkers);
            try {
                for (std::size_t worker = 0; worker < waveHostWorkers; ++worker) {
                    workers.emplace_back([&, worker] {
                        std::vector<boost::fibers::fiber> fibers;
                        fibers.reserve((waveSize + waveHostWorkers - 1) / waveHostWorkers);
                        try {
                            for (std::size_t local = worker; local < waveSize;
                                 local += waveHostWorkers) {
                                const std::size_t point = indices[wave + local];
                                fibers.emplace_back([&, point] { evaluatePoint(point); });
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(failureMutex);
                            if (!fiberInfrastructureFailure) {
                                fiberInfrastructureFailure = std::current_exception();
                            }
                        }
                        for (boost::fibers::fiber& fiber : fibers) {
                            if (fiber.joinable()) fiber.join();
                        }
                    });
                }
            } catch (...) {
                for (std::thread& worker : workers) worker.join();
                throw;
            }
            for (std::thread& worker : workers) worker.join();
            if (fiberInfrastructureFailure) {
                std::rethrow_exception(fiberInfrastructureFailure);
            }
        }
        return EvaluationPassProfile{
            std::max(launchLimit, qSusyLaunchLimit),
            std::max(
                scheduler.maximumObservedBatchSize(),
                qSusyScheduler.maximumObservedBatchSize()),
            scheduler.profile(),
            qSusyScheduler.profile()};
    };

    std::vector<std::size_t> allPoints(configs.size());
    for (std::size_t point = 0; point < configs.size(); ++point) {
        allPoints[point] = point;
        PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
        diagnostic.requestedBackend = options.backend;
        diagnostic.selectedBackend = Backend::Cuda;
        diagnostic.detail = "CUDA FP64 RGE/Q_SUSY candidate requested";
    }
    try {
        const auto launch = evaluateIndices(allPoints, false, run.results);
        run.summary.cudaFp64LaunchLimit = launch.launchLimit;
        run.summary.maximumCudaLaunchSize = launch.maximumLaunch;
        addStageProfile(run.summary.rgeProfile, launch.rge);
        addStageProfile(run.summary.qSusyProfile, launch.qSusy);
    } catch (const std::exception& failure) {
        const std::string detail =
            std::string("CUDA batch infrastructure failure: ") + failure.what();
        for (std::size_t point = 0; point < configs.size(); ++point) {
            run.results[point] = Result{};
            run.results[point].error = detail;
            PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
            diagnostic.candidateTier = ExecutionTier::None;
            diagnostic.finalTier = ExecutionTier::None;
            diagnostic.executed = false;
            diagnostic.adjudicationReasons |=
                adjudicationReason(AdjudicationReason::InfrastructureFailure);
            diagnostic.detail = detail;
        }
        run.summary.failed = configs.size();
        return run;
    } catch (...) {
        const std::string detail = "unknown CUDA batch infrastructure failure";
        for (std::size_t point = 0; point < configs.size(); ++point) {
            run.results[point] = Result{};
            run.results[point].error = detail;
            PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
            diagnostic.candidateTier = ExecutionTier::None;
            diagnostic.finalTier = ExecutionTier::None;
            diagnostic.executed = false;
            diagnostic.adjudicationReasons |=
                adjudicationReason(AdjudicationReason::InfrastructureFailure);
            diagnostic.detail = detail;
        }
        run.summary.failed = configs.size();
        return run;
    }
    std::vector<std::size_t> retryPoints;
    for (std::size_t point = 0; point < configs.size(); ++point) {
        const bool rootBoundary = nearQSusyBoundary(run.results[point]);
        const bool branchBoundary = nearBranchBoundary(run.results[point]);
        if (rootBoundary || branchBoundary) {
            retryPoints.push_back(point);
            if (rootBoundary) {
                run.diagnostics[point].adjudicationReasons |=
                    adjudicationReason(AdjudicationReason::RootBoundary);
            }
            if (branchBoundary) {
                run.diagnostics[point].adjudicationReasons |=
                    adjudicationReason(AdjudicationReason::BranchBoundary);
            }
        }
    }

    std::vector<Result> doubleDoubleResults(configs.size());
    run.summary.doubleDoubleRetries = retryPoints.size();
    std::string doubleDoubleInfrastructureFailure;
    try {
        const auto launch = evaluateIndices(retryPoints, true, doubleDoubleResults);
        run.summary.maximumCudaLaunchSize = std::max(
            run.summary.maximumCudaLaunchSize, launch.maximumLaunch);
        addStageProfile(run.summary.rgeProfile, launch.rge);
        addStageProfile(run.summary.qSusyProfile, launch.qSusy);
    } catch (const std::exception& failure) {
        doubleDoubleInfrastructureFailure = failure.what();
    } catch (...) {
        doubleDoubleInfrastructureFailure = "unknown infrastructure failure";
    }
    for (const std::size_t point : retryPoints) {
        PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
        if (!doubleDoubleInfrastructureFailure.empty()) {
            run.results[point] = evaluate(configs[point]);
            diagnostic.finalTier = ExecutionTier::CpuMpfr;
            diagnostic.cpuAdjudicated = true;
            diagnostic.adjudicationReasons |=
                adjudicationReason(AdjudicationReason::InfrastructureFailure);
            diagnostic.detail += "; CUDA double-double infrastructure failure: "
                + doubleDoubleInfrastructureFailure + "; CPU/MPFR adjudicated";
            ++run.summary.cpuAdjudications;
            continue;
        }

        const Result& fp64 = run.results[point];
        const Result& doubleDouble = doubleDoubleResults[point];
        if (doubleDouble.ok && sameLabelResult(fp64, doubleDouble)
                && !nearQSusyBoundary(doubleDouble)
                && !nearBranchBoundary(doubleDouble)) {
            run.results[point] = doubleDouble;
            diagnostic.finalTier = ExecutionTier::CudaDoubleDouble;
            diagnostic.detail += "; CUDA double-double accepted";
            continue;
        }
        if (!sameLabelResult(fp64, doubleDouble)) {
            diagnostic.adjudicationReasons |=
                adjudicationReason(AdjudicationReason::TierDisagreement);
        }
        run.results[point] = evaluate(configs[point]);
        diagnostic.finalTier = ExecutionTier::CpuMpfr;
        diagnostic.cpuAdjudicated = true;
        diagnostic.detail += "; CPU/MPFR adjudicated";
        ++run.summary.cpuAdjudications;
    }

    run.summary.cudaCandidates = static_cast<std::size_t>(std::count_if(
        run.diagnostics.begin(), run.diagnostics.end(),
        [](const PointExecutionDiagnostic& diagnostic) {
            return diagnostic.executed
                && diagnostic.candidateTier != ExecutionTier::None;
        }));

    for (std::size_t point = 0; point < configs.size(); ++point) {
        Result& candidate = run.results[point];
        PointExecutionDiagnostic& diagnostic = run.diagnostics[point];
        if (options.backendAudit) {
            const Result reference = evaluate(configs[point]);
            const std::string mismatch = labelResultMismatch(candidate, reference);
            const bool matched = mismatch.empty();
            diagnostic.auditCompared = true;
            diagnostic.auditMatched = matched;
            if (!matched) {
                diagnostic.adjudicationReasons |=
                    adjudicationReason(AdjudicationReason::AuditMismatch);
                ++run.summary.auditMismatches;
                diagnostic.detail += "; audit mismatch: " + mismatch;
                candidate = reference;
                if (!diagnostic.cpuAdjudicated) {
                    diagnostic.finalTier = ExecutionTier::CpuMpfr;
                    diagnostic.cpuAdjudicated = true;
                    diagnostic.detail += "; CPU/MPFR adjudicated after audit mismatch";
                    ++run.summary.cpuAdjudications;
                }
            }
        }
        if (candidate.ok) {
            ++run.summary.succeeded;
        } else {
            ++run.summary.failed;
        }
    }
    return run;
}

}  // namespace natlha::detail
