# Boost-derived numerical routines

The CUDA Q_SUSY dense-output and TOMS Algorithm 748 implementations adapt formulas and
control flow from the installed Boost 1.83 headers:

- `boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp`
  - Copyright 2010-2013 Karsten Ahnert
  - Copyright 2010-2013 Mario Mulansky
  - Copyright 2012 Christoph Koke
- `boost/math/tools/toms748_solve.hpp`
  - Copyright John Maddock 2006

Those adapted portions are distributed under the Boost Software License, Version 1.0.
The complete license statement is in `LICENSE_1_0.txt` beside this notice.
