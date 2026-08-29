#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_derivatives.inl"
#include "natlha_execution_context.hpp"

#include <boost/numeric/odeint.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

void MSSMRGESolver(
        const std::vector<double>& x,
        std::vector<double>& dxdt,
        const double /* t */) {
    MSSMRGEDerivatives(x, dxdt);
}

typedef boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>> stepper_type;

const ODETolerances& odeTolerances() {
    struct TolEnv {
        static double read(const char * name, double fallback) {
            const char * e = std::getenv(name);
            if (e == nullptr || *e == '\0') return fallback;
            char * end = nullptr;
            const double value = std::strtod(e, &end);
            if (end == e || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
                std::cerr << "natLHA: " << name << "='" << e
                          << "' is not a finite positive number; refusing to guess.\n";
                std::exit(4);
            }
            return value;
        }
    };
    static const ODETolerances tolerances = {
        TolEnv::read("NATLHA_ODE_ABS_ERR", 1.0E-12),
        TolEnv::read("NATLHA_ODE_REL_ERR", 1.0E-12)
    };
    return tolerances;
}

static std::vector<double> solveODEsCpu(
        std::vector<double> initialConditions,
        double startTime,
        double endTime,
        double timeStep) {
    using state_type = std::vector<double>;
    state_type x = initialConditions;

    // Step statistics, for telling a genuinely stiff integration apart from one that is
    // grinding through a huge number of tiny steps in a narrow window.
    //
    // `steps` counts ACCEPTED steps only. Read from Boost's own source: in the
    // controlled_stepper_tag overload of integrate_adaptive.hpp the observer is invoked at
    // lines 94 and 111, outside the do/while at lines 100-106 that repeats `st.try_step`
    // until it stops returning `fail`. Rejected attempts therefore never reach the observer
    // and are invisible here.
    //
    // So a large `steps` shows the integrator took many small successful steps. A SMALL
    // `steps` together with a long wall time does NOT identify a cause: repeated rejections,
    // or simply expensive right-hand-side evaluations, would both look the same from here.
    struct StepStats {
        long steps = 0;
        double prevT = 0.0;
        double minDt = 0.0;      // smallest |t_{k+1} - t_k| seen, 0 until two points exist
        double minDtAt = 0.0;    // the t where that smallest step ended
    };
    StepStats stats;

    struct MyObserver {
        StepStats * s;
        void operator()(const state_type& /* x */, const double t) const {
            if (s->steps > 0) {
                const double dt = std::abs(t - s->prevT);
                if (s->steps == 1 || dt < s->minDt) {
                    s->minDt = dt;
                    s->minDtAt = t;
                }
            }
            ++(s->steps);
            s->prevT = t;
        }
    };

    MyObserver myObserver;
    myObserver.s = &stats;

    // Error control for the adaptive stepper.
    //
    // make_controlled takes (eps_abs, eps_rel). Read from Boost's own source rather than
    // described from memory: default_operations.hpp:443 divides each component's error by
    //     m_eps_abs + m_eps_rel * ( m_a_x * |x| + m_a_dxdt * |dt * dxdt| )
    // and controlled_runge_kutta.hpp:67-68 defaults both weight factors a_x and a_dxdt to 1.
    // So eps_abs sets a floor on the denominator that matters most where |x| and |dt*dxdt|
    // are small, while eps_rel governs the bound where they are large. The two are not
    // interchangeable, and the derivative term means the scaling is not a function of |x|
    // alone.
    //
    // Both default to 1.0E-12, the value this call has always passed, so shipped numerical
    // behaviour is unchanged unless one of the environment variables below is set. They exist
    // so the tolerances can be varied WITHOUT rebuilding, which is what makes it practical to
    // test whether stepper control explains the pathologically slow points seen during
    // labelling. The shared `odeTolerances()` accessor reads each once into a
    // function-local static so the trajectory and joint-convergence gate use one contract.
    //
    // NOT ESTABLISHED: that these tolerances cause any slow point. They are made adjustable so
    // the question can be answered, not because it has been.
    //
    // PARSING IS VALIDATED, and that is not pedantry here. strtod reports failure only through
    // its end pointer: on unparseable text it returns 0.0 and leaves end == the input, which a
    // caller that ignores end cannot distinguish from a genuine "0". A tolerance of ZERO makes
    // the denominator above vanish where the other terms are also small, so the error ratio
    // diverges and the stepper shrinks dt without limit -- a typo in an environment variable
    // would then look exactly like the pathology being investigated. Anything that is not a
    // fully consumed, finite, strictly positive number is therefore refused loudly.
    const ODETolerances& tolerances = odeTolerances();

    boost::numeric::odeint::integrate_adaptive(
        make_controlled(tolerances.absolute, tolerances.relative, stepper_type()),
        MSSMRGESolver, x, startTime, endTime, timeStep, myObserver
    );

    // Emit the step statistics when NATLHA_ODE_TRACE is set to anything non-empty. Off by
    // default and written to stderr, so a traced run's stdout stays byte-identical to an
    // untraced one and remains safe to pipe into a parser.
    static const bool trace = [] {
        const char * e = std::getenv("NATLHA_ODE_TRACE");
        return e != nullptr && *e != '\0';
    }();
    if (trace) {
        std::cerr << "# ode_trace steps " << stats.steps
                  << "  t_from " << startTime << "  t_to " << endTime
                  << "  span " << (endTime - startTime)
                  << "  min_dt " << stats.minDt
                  << "  min_dt_at " << stats.minDtAt
                  << "  eps_abs " << tolerances.absolute
                  << "  eps_rel " << tolerances.relative << "\n";
    }

    return x;
}

std::vector<double> solveODEs(
        std::vector<double> initialConditions,
        double startTime,
        double endTime,
        double timeStep) {
    const natlha::detail::CudaExecutionContext* context =
        natlha::detail::currentCudaExecutionContext();
    if (context != nullptr && context->submitOde != nullptr) {
        natlha::detail::ScopedCudaDispatchCall dispatch("ODE solve");
        return (*context->submitOde)(
            std::move(initialConditions), startTime, endTime, timeStep);
    }
    return solveODEsCpu(std::move(initialConditions), startTime, endTime, timeStep);
}
