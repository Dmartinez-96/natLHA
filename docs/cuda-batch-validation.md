# CUDA batch backend: design and validation report

## Scope and evidence boundary

This report describes the optional CUDA batch extension. The SoftSUSY population
was generated on 2026-08-28; the final population audit, performance
measurements, and build/test matrix were collected on 2026-08-29 after the
review-driven source changes.
The tested natLHA worktree is based on commit
`223600d005eb73759a2bb062ec8ff16768bf81e2`; the CUDA changes described here are
the uncommitted patch relative to that base until the project owner chooses to
commit them. The generated SoftSUSY population is validation evidence and is not
part of the repository's permanent test corpus.

The measurements establish correctness on the named tests and generated
population, and throughput on the named laptop. They do not establish parity for
all physically possible spectra, a universal speedup, or RTX 5090 performance.
The CPU implementation remains authoritative whenever numerical tiers disagree.

## Execution design

The extension separates four concerns:

1. The public batch API keeps one `Result` and one execution diagnostic aligned
   with each input `Config`.
2. A hardware-bounded host-thread pool advances lightweight, fiber-backed point
   state machines. A numerical wait yields the host thread instead of consuming
   one blocked OS thread per scan point.
3. Separate coalescing schedulers collect concurrent `solveODEs` and `findQSusy`
   requests. They submit the shared 44-state, two-loop MSSM RGE algebra and a
   fused Q_SUSY evolution/scan/root-refinement kernel to CUDA.
4. FP64 candidates near a numerical or branch boundary are recomputed with
   device double-double arithmetic. A persistent boundary, tier disagreement,
   or audit mismatch is resolved by the CPU/MPFR implementation.

The CUDA integrator is an adaptive Dormand-Prince 5(4) solver with explicit
non-finite input/state, step-limit, and step-underflow statuses. The Q_SUSY
kernel uses the same stepper, Boost-derived dense-output interpolation and
TOMS748 control flow, shared CPU/device stop and domain classifiers, and
explicit scan-spacing, boundary, root-count, refinement, and residual statuses.
The final bracket width is retained as telemetry and participates in the
precision-risk gate. API validity rejects non-finite or negative widths; exact
sampled roots report zero, while a large finite width remains valid telemetry
and triggers precision escalation rather than being misclassified as malformed.
One trajectory is assigned to one GPU thread. Requested populations can be
larger than device memory because `cudaBatchSize` bounds each launch and the
backend processes successive chunks.

CUDA compilation explicitly enables fused multiply-add and precise division and
square root (`--fmad=true --prec-div=true --prec-sqrt=true`). Device parity tests
therefore validate the arithmetic policy actually used by production kernels.

`cudaWorkers` and `cudaBatchSize` are intentionally independent. The former
bounds live logical point fibers in one wave; the latter bounds a scheduler
launch. Automatic logical concurrency is the smaller of the population and
4096, while the OS-thread pool never exceeds
`std::thread::hardware_concurrency()`. The CLI allows an explicit logical-point
limit from 0 (automatic) through 4096. Fiber, thread, or CUDA infrastructure
failure is reported distinctly rather than returning a partial batch.

## Backend and failure semantics

The CPU backend is the default, so an existing single-point or batch invocation
does not silently change numerical implementation. Explicit CUDA selection
fails every aligned row before numerical execution when the build, device, or
ordinal is unavailable. Automatic selection records the unavailable-backend
reason and runs CPU instead.

Every executed CUDA point diagnostic records requested and selected backend,
candidate and final tier, adjudication-reason bits, whether CPU adjudicated it,
and optional audit status. CUDA infrastructure failure, non-finite state, ODE limits,
Q_SUSY root/domain boundaries, branch-boundary retry, tier disagreement, and
audit mismatch are distinct conditions. Failed result rows retain their input
position. Device strings never participate in semantic decisions: device
statuses map to structured host reasons, while final failure text and all labels
are constructed on the CPU.
The batch summary records the selected FP64 launch limit and the largest launch
actually observed, making automatic sizing and underfilled workloads visible to
API and CLI consumers.

The CLI preserves input order under every backend but has two output-timing
contracts. Explicit CPU batches evaluate, emit, and flush one row at a time.
CUDA and auto batches retain aligned results until the complete batch call
returns, then emit all rows in input order. Interruption during CUDA/auto
evaluation may leave the already-written header, but it cannot leave a prefix
of completed data rows.

For Delta_BG, CPU/CUDA comparison applies exact hard gates to success state,
requested-measure presence, contribution count, label, ordinal, and ordering.
Values must agree within the larger of propagated root-width uncertainty and
0.05% of `max(1, |candidate|, |CPU|)`. That scale-proportional 0.05% term is one
tenth of the CPU adaptive-window agreement term's 0.5% scale component;
propagated root-width uncertainty can make the total allowed difference larger.
Neither value allowance can hide a label or branch-selection change. Other
measure values use a relative/absolute gate of `2e-8` against the same unit
floor.

Exact Delta_BG contribution order is part of the CPU semantic contract. The
generic boundary detector therefore escalates any adjacent contributions whose
magnitude gap is no larger than their combined root uncertainty or 0.01% of a
unit-floored local scale. Persistent FP64/double-double disagreement causes CPU
adjudication even when the signed headline is stable.

## Build and test identity

Validation machine:

- CPU: Intel Core Ultra 9 275HX, 24 online logical CPUs
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0,
  12,227 MiB reported memory
- NVIDIA driver: 577.13
- CUDA compiler: 12.8.93
- CMake: 3.28.3
- operating surface: x86-64 Linux 6.6.87.2-microsoft-standard-WSL2

The CPU-only and CUDA-enabled builds can be reproduced from the repository root
with the following equivalent clean-tree recipe. These commands use portable
output directories; they are not a claim that the retained test artifacts used
those literal paths or generators.

```bash
cmake -S natLHA -B build/cpu -G Ninja \
  -DNATLHA_STATIC_LINK=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure

cmake -S natLHA -B build/cuda -G Ninja \
  -DNATLHA_STATIC_LINK=OFF \
  -DNATLHA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.8-or-newer/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

The placeholder CUDA compiler path must name a verified 12.8-or-newer `nvcc`;
check it with `/path/to/cuda-12.8-or-newer/bin/nvcc --version` before configuring.
This matters on machines where an older toolkit appears first on `PATH`.

The two additional CPU matrices can be reproduced with these equivalent
configurations:

```bash
cmake -S natLHA -B build/cpu-static -G Ninja \
  -DNATLHA_STATIC_LINK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-static
ctest --test-dir build/cpu-static --output-on-failure

cmake -S natLHA -B build/cpu-sanitize -G Ninja \
  -DNATLHA_STATIC_LINK=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build/cpu-sanitize
ctest --test-dir build/cpu-sanitize --output-on-failure
```

The retained runs were actually in `/tmp/natlha-cpu-build`,
`/tmp/natlha-cpu-static`, `/tmp/natlha-cpu-sanitize`, and
`/tmp/natlha-cuda-build`. Their `CMakeCache.txt` files recorded, respectively:
Release/Ninja/dynamic CPU; Release/Unix Makefiles/static CPU;
Debug/Unix Makefiles/dynamic CPU with the displayed address/undefined sanitizer
flags; and Release/Ninja/dynamic CUDA with
`/usr/local/cuda-12.8/bin/nvcc`. All used `/usr/bin/c++`, identified as GCC
13.3.0. The portable recipes above may use a different build directory or
generator while preserving the named natLHA options.

Final 2026-08-29 result: the dynamic CPU, static CPU, and GCC sanitizer matrices
each passed 14 of 14 tests; the CUDA-enabled matrix passed 15 of 15 tests.
The CUDA test includes 257 FP64 RGE trajectories chunked at 17, five RGE
double-double trajectories chunked at three, 33 FP64 Q_SUSY trajectories
chunked at seven, three Q_SUSY double-double trajectories chunked at two, CPU
comparison of every 44-component endpoint/root state, shared helper parity,
non-finite rejection, status-to-adjudication mapping, and an audited high-level
batch. After the shared CPU Q_SUSY helper extraction, the retained command
`/tmp/natlha-cpu-build/natlha-cli --slha natLHA/test_data/joint_qsusy.slha | sha256sum`
produced stdout SHA-256
`baa7948f1fb90eee5ca0caf2599d28832c7f4ee222eb3257a2823d21837f14a9`;
the extraction review separately compared the original definitions with the
shared definitions and their CPU wrappers.

The checked-in `tools/validate_rge_extraction.py` compares the shared RGE
equation body mechanically against the original CPU function at the base commit
named by this report. It normalizes only `double`/generic-scalar, `pow` adapter,
and loop-factor identifier substitutions, then removes comments and whitespace.
It separately compares the generic loop-factor definitions with their original
definitions in `constants.hpp`. That comparison maps `Real(N)` casts to the
corresponding floating-point `N.0` literals, maps `Real(M_PI)` to `M_PI`, maps
the remaining scalar type name `Real` to `double`, maps the `pow` adapter and
loop-factor identifiers, and removes comments and whitespace. It does not erase
`.0` from bare literals, so accidental bare integers remain different. The
validator also requires an exponent match for every generic power call. Built-in
mutation probes must reject a changed loop-factor denominator, bare-integer
drift, an unparsed exponent, and an unexpected exponent before the source
comparison runs. On the final worktree,
`python3 tools/validate_rge_extraction.py` reported equal 96,054-byte equation
streams, equal loop factors, 2,778 power calls in each, 2,778 matched generic
exponents, and exponent values 2, 3, and 4 only. The generic power adapter
statically checks the return type of the underlying ADL-selected `pow` overload,
while `nm -C` on the retained CUDA object reported concrete
`integrateKernel<DoubleDouble>` and `qSusyKernel<DoubleDouble>` symbols, as well
as their `double` counterparts.

A separate GCC AddressSanitizer plus UndefinedBehaviorSanitizer CPU build passed
all 14 tests. NVIDIA Compute Sanitizer 2025.1 could launch the production CUDA
binary but could not attach under this WSL/WDDM surface (`Failed to initialize
WDDM debugger interface`, `Device not supported`), so no device-memcheck pass is
claimed. The run itself completed its row, the tool emitted both environment
errors and an error summary, and the wrapper process exited zero.

## SoftSUSY validation population

The 16 SLHA spectra were generated with SOFTSUSY 4.1.23 from source commit
`f816a0bb29c4ac8308dec5fa157b8c1c3674de26` (`v4.1.22-6-gf816a0b`), using the
SUGRA interface, positive mu sign, and tolerance `1e-4`. At that commit,
`configure.ac` declares version 4.1.23 and the retained binary's banner agrees;
the `git describe` string records its position relative to the prior 4.1.22 tag.
The parameter tuples `(m0, m12, A0, tan(beta))`, in GeV where applicable, were:

```text
(500, 500, -1000, 10)       (500, 800, -1500, 20)
(750, 600, 0, 30)           (1000, 700, -2000, 10)
(1000, 1000, -2500, 30)     (1500, 800, -3000, 40)
(2000, 1000, -4000, 10)     (2000, 1500, -3500, 30)
(3000, 800, -5000, 20)      (3000, 1200, -6000, 50)
(4000, 1000, -6500, 15)     (5000, 1200, -8000, 10)
(5000, 1600, -7500, 30)     (7000, 1500, -10000, 20)
(8000, 2000, -12000, 40)    (10000, 2500, -15000, 50)
```

A single spectrum can be regenerated with the corresponding values using:

```bash
softpoint.x sugra \
  --m0=VALUE --m12=VALUE --a0=VALUE --tanBeta=VALUE \
  --sgnMu=1 --tol=1e-4 > point.slha
```

The retained `/tmp/softsusy-build/softpoint.x` binary's own usage output
documents those SUGRA parameter and common-option spellings. Re-running the
first tuple through that command on 2026-08-29 produced SHA-256
`4aed17c55779d7a619889f816d1032b160ed85f370d1a7b76eabcb9e3d2ee87b`,
identical to the retained first population file.

Observed audit results:

- DEW + Delta_HS + differential Delta_SN (`nF=5`, `nD=3`): 16/16 rows
  succeeded and matched; zero retries, adjudications, or mismatches.
- adaptive Delta_BG, model 1: 16/16 rows succeeded and matched; zero retries,
  adjudications, or mismatches.
- adaptive Delta_BG models 1 through 5: the selected cross-model spectrum
  `(1000, 1000, -2500, 30)` retained success, complete label/ordinal/order
  parity, and value agreement.
- adaptive Delta_BG model 6: the FP64 candidate triggered the aggregate
  branch-boundary detector. The retained diagnostic does not identify which
  underlying boundary sub-condition fired. The mandatory double-double retry
  did not clear the acceptance boundary gates, so the final row was
  CPU-adjudicated and matched the CPU audit. Its final diagnostic reported one
  retry, one CPU adjudication, adjudication-reason value 32
  (`BranchBoundary`), and zero audit mismatches.
- Delta_BG fixed diagnostic precisions 1 (eight-point) and 2 (four-point), model
  1: both succeeded and matched the CPU audit.

These checks cover every Delta_BG model label family and all three exposed
numerical modes on at least one spectrum. Only model 1 and the non-Delta_BG
measures were exercised over all 16 varied spectra, so the evidence does not
support an all-model, all-population claim.

## Performance observations

Timing used repeated copies of `natLHA/test_data/joint_qsusy.slha` for the DEW
scaling sweep and the 16 varied spectra above for Delta_BG. Audit was disabled,
both controls were automatic, and `/usr/bin/time` measured three end-to-end DEW
runs at each size after the binaries had been built and exercised. Wall times
include process startup, SLHA reading, host calculations, allocation, transfer,
and kernel time. The table reports the median and observed minimum-maximum range;
three samples characterize run-to-run variation but are not confidence intervals.

For reproduction, each `repeated-N.txt` consists of exactly `N` lines containing
the repository-relative path `natLHA/test_data/joint_qsusy.slha`. Each timing run
used the following invocation shape, with `BACKEND` and its matching build
substituted. Standard output was redirected to `/dev/null`; standard error
retained the timing and backend summaries:

```bash
/usr/bin/time -f 'elapsed=%e user=%U sys=%S maxrss_kb=%M' \
  build/BACKEND/natlha-cli \
  --batch repeated-N.txt --backend BACKEND > /dev/null
```

The individual elapsed-time observations underlying the DEW aggregates were,
in seconds and sorted within each three-run set:

| Points | CPU observations | CUDA observations |
|---:|---:|---:|
| 64 | 1.96, 1.96, 1.97 | 2.92, 2.94, 3.20 |
| 256 | 7.85, 7.87, 7.89 | 3.49, 3.66, 3.70 |
| 1,024 | 31.43, 31.52, 31.70 | 4.24, 4.39, 4.72 |
| 4,096 | 125.85, 126.06, 126.45 | 7.47, 7.62, 7.98 |

| Workload | CPU median [range] | CUDA median [range] | Median speedup |
|---|---:|---:|---:|
| 64 DEW points | 1.96 s [1.96-1.97] | 2.94 s [2.92-3.20] | 0.67x |
| 256 DEW points | 7.87 s [7.85-7.89] | 3.66 s [3.49-3.70] | 2.15x |
| 1,024 DEW points | 31.52 s [31.43-31.70] | 4.39 s [4.24-4.72] | 7.18x |
| 4,096 DEW points | 126.06 s [125.85-126.45] | 7.62 s [7.47-7.98] | 16.54x |
| 16 varied adaptive Delta_BG model-1 points (single run) | 37.11 s | 17.75 s | 2.09x |

The final 16-point adaptive model-1 CUDA run issued 489 RGE requests and 56
Q_SUSY requests. Small or underfilled batches can be slower on CUDA because
launch, synchronization, and host scheduling overhead dominate.

The table shows the intended scaling shape on the tested laptop: startup and
launch overhead make 64 points slower on CUDA, while 4,096 repeated DEW points
reach about 538 points/s and a 16.5x median end-to-end speedup. The retained
diagnostic for the third 4,096-point timing sample reported a maximum launch of
3,037 trajectories and 4,096 successful rows. This is evidence for the named
repeated fixture, not a claim of 16.5x on arbitrary spectra or a prediction of
RTX 5090 performance. On another machine, sweep `--cuda-workers` while keeping
the input population and all physics options fixed, and report points per second
together with CPU/GPU identity and backend diagnostics.

The batch summary and CLI expose separate RGE and Q_SUSY profiles. For the
profiled 1,024-point run, RGE recorded 6,144 requests in 15 launches and Q_SUSY
recorded 3,072 requests in 12 launches. Host-observed synchronized kernel time
was 2.32 s for RGE and 1.75 s for Q_SUSY; allocation time was 0.022 s and 0.011 s
respectively. Cumulative queue wait is deliberately reported as a sum across
logical points (772.5 s RGE, 393.3 s Q_SUSY in that run), so it diagnoses
contention but must not be read as wall time. Transfer and allocation fields
likewise describe serialized scheduler work rather than a device-only profiler
trace.

## Remaining optimization boundary

The scheduler now exposes thousands of numerical trajectories without thousands
of OS threads, and Q_SUSY root work is fused on the device. EWSB retuning,
high-precision contribution construction, branch ordering, final labels, and
formatting remain on the CPU. Fibers can also reach different numerical stages
at slightly different times. The retained 4,096-point diagnostic peaked at
3,037 rather than one monolithic 4,096-trajectory launch, so a still more
explicit data-oriented stage graph could improve synchronization and buffer
reuse.

That future optimization should be justified by profiling on the target GPU.
It must retain the same ordered-label, failure-semantic, tiered-precision, and
CPU-audit contracts; moving host-only strings or semantic adjudication onto the
device is neither required nor desirable for throughput.
