# -*- coding: utf-8 -*-
# options.py
# Global options file
# Copyright 2006 Giuseppe Venturini

# This file is part of the ahkab simulator.
#
# Ahkab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2 of the License.
#
# Ahkab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License v2
# along with ahkab.  If not, see <http://www.gnu.org/licenses/>.

"""This module contains options and configuration switches the user
may tune to meet his needs.

The default values are sensible options for the general case.
"""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import os

import numpy as np

try:
    from matplotlib import rcParams as plotting_rcParams
except ImportError:
    plotting_rcParams = {}

#: Encoding of the netlist files.
encoding = 'utf8'

#: Cache size to be used in :func:`ahkab.utilities.memoize`, defaults to 512MB
cache_len = 67108864 # 512MB

#: A boolean to differentiate command line execution from module import
#: When cli is False, no printing and no weird stdout stuff.
cli = False

#: A boolean to specify if the netlist is in Xyce format
Xyce_netlist = False

# testcase name
testcase = None

# global: errors
#: Voltage absolute tolerance.
vea = 1e-6
#: Voltage relative tolerance.
ver = 1e-2
#: Current absolute tolerance.
iea = 1e-6
#: Current relative tolerance.
ier = 1e-2
# delta tolerance
deltaTol = 0.33

# global: circuit
#: Minimum conductance to ground.
gmin = 1e-12 * 1e0
#: Should we show to the user results pertaining to nodes introduced by
#: components or by the simulator?
print_int_nodes = True
#: whether reparse a circuit netlist 
#: 0: no reparse, 1: reparse only the directives and options part, 2: full reparse
reparse = 0

# global: solving
#: Dense matrix limit: if the dimensions of the square MNA matrix are bigger,
#: use sparse matrices.
dense_matrix_limit = 400
# Use sparse matrix code to build and solve the matrices (Alex)
# Preferred for all real cases. The dense matrix code is useful only for toy models 
use_sparse = True
#: linear solution method (scipy, pardiso, GMRES)
lin_sol_method = 'pardiso'
# relative tolerance of GMRES
lin_GMRES_rtol = 1e-6
# absolute tolerance of GMRES
lin_GMRES_atol = 1e-9
# max GMRES iterations
lin_GMRES_maxiter = 100
# Whether restart is allowed for GMRES
lin_GMRES_restart = True
#: Should we damp artificially the first NR iterations? See also
#: :func:`ahkab.dc_analysis.get_td`.
nr_damp_first_iters = False
#: In all NR iterations, lock the nodes controlling non-linear elements. See
#: also :func:`ahkab.dc_analysis.get_td`.
nl_voltages_lock = True     # Apply damping - slows down solution.
#: Non-linear nodes lock factor:
#: if we allow the voltage on controlling ports to change too much, we may
#: have current/voltage overflows. Think about the diode characteristic.
#: So we allow them to change of ``nl_voltages_lock_factor``
#: :math:`\cdot V_{th}` at most and damp all variables accordingly.
nl_voltages_lock_factor = 4
# use BSIM3 model or not
useBsim3 = False

#: Whether the standard solving method can be used.
use_standard_solve_method = True
#: Whether the gmin-settping homothopy can be used.
use_gmin_stepping = True
#: Whether the source-stepping homothopy can be used.
use_source_stepping = True

#: When printing out to the user, whether we can suppress trailing zeros.
print_suppress = False
#: When printing out to the user, how many decimal digits to show at maximum.
print_precision = np.get_printoptions()['precision']
# Whether the element-wise op information is computed and stored (only when you want this info)
get_op_info = False

# dc
#: Maximum allowed NR iterations during a DC analysis.
dc_max_nr_iter = 200
#: Enable guessing to init the NR solver during a DC analysis.
dc_use_guess = True
#: Do not perform an init DC guess if its effort is higher than
#: this value.
dc_max_guess_effort = 250000
# shorthand to set logarithmic stepping in DC analyses.
dc_log_step = 'LOG'
# shorthand to set linear stepping in DC analyses.
dc_lin_step = 'LIN'
#: Can we skip troublesome points during DC sweeps?
dc_sweep_skip_allowed = True


# transient
#: The default differentiation method for transient analyses.
default_tran_method = "TRAP"
#: Minimum allowed discretization step for time.
hmin = 1e-16
#: Maximum number of time iterations for transient analyses
hmax = 1e-6
#: Notice the default (0) means no limit is enforced.
transient_max_time_iter = 0  # disabled
#: Maximum number of NR iterations for transient analyses.
transient_max_nr_iter = 200
#: In a transisent analysis, if a prediction value is avalilable,
#: use it as first guess for ``x(n+1)``, otherwise ``x(n)`` is used.
transient_prediction_as_x0 = True
#: Use aposteriori step control?
transient_use_aposteriori_step_control = True
#: Step change threshold:
#: we do not want to redo the iteraction if the aposteriori check suggests a step
#: that is very close to the one we already used. A value of 0.9 seems to be a
#: good idea.
transient_aposteriori_step_threshold = 0.9
#: Disable all step control in transient analyses.
transient_no_step_control = True
#: Minimum capacitance to ground.
cmin = 1e-16
# minimum number of time-steps to use between breakpoints. 
#This enforces a maximum time-step between breakpoints equal to the distance 
#between the last breakpoint and the next breakpoint divided by MINTIMESTEPSBP
mintimestepsbp = 2

# exponential integrator
#: max dimension of Krylov subspace
ei_max_m = 40
# Default gamma for shift-and-invert Krylov
ei_gamma = 1e-10
# Krylov approximation absolute tolerance
krylovTola = 1e-12
# Krylov approximation relative tolerance
krylovTolr = 1e-9
ei_reg = True
ei_newton = False

# pss
use_pss = False
BFPSS = 'brute-force'
SHOOTINGPSS = 'shooting'

# shooting
#: Default number of points for a shooting analysis.
shooting_default_points = 100
#: Maximum number of NR iterations for shooting analyses.
shooting_max_nr_iter = 10000

# bfpss
#: Default number of points for a BFPSS analysis.
bfpss_default_points = 100
#: Maximum number of NR iterations for BFPSS analyses.
bfpss_max_nr_iter = 10000

# symbolic
#: Enable the manual solver:
#: solve the circuit equations one at a time as you might do "manually".
symb_sympy_manual_solver = False
#: Formulate the equations with conductances and at the last moment
#: swap resistor symbols back in. It seems to make sympy play nicer.
#: Sometimes.
symb_formulate_with_gs = False

# ac
ac_log_step = 'LOG'
ac_lin_step = 'LIN'
#: Maximum number of NR iterations for AC analyses.
ac_max_nr_iter = 20
#: Use degrees instead of rads in AC phase results.
ac_phase_in_deg = False

#pz
#: Maximum considered angular frequency in rad/s for PZ analyses.
pz_max = 1e12

# plotting
# Set to None to disable writing plots to disk
#: Should plots be shown to the user? This variable is set to ``True``
#: automatically if a screen is detected in Unix systems.
#:
#: Notice that by default ahkab both shows plots *and* saves them to disk.
plotting_show_plots = ('DISPLAY' in os.environ)
#: Wait for the user to close the plot? If set to ``False``, plots are created
#: and immediately destroyed.
plotting_wait_after_plot = True
#: Format to be used when writing plots to disk.
plotting_outtype = "png"
#: Matplotlib line plot style: see matplotlib's doc.
plotting_style = "-o"
#: Plotting line width.
plotting_lw = 2
#: Default size for plots showed to the user, in inches.
plotting_display_figsize = (12.94, 8)
#: Default size for plots saved to disk.
plotting_save_figsize = (20, 10)
# plotting_rcParams['font.family'] = 'Baskerville'
plotting_rcParams['axes.labelsize'] = 12
plotting_rcParams['xtick.labelsize'] = 12
plotting_rcParams['ytick.labelsize'] = 12
plotting_rcParams['legend.fontsize'] = 14

# Windows
RECT_WINDOW = 'RECT'
BART_WINDOW = 'BART'
HANN_WINDOW = 'HANN'
HAMM_WINDOW = 'HAMM'
BLACK_WINDOW = 'BLACK'
HARRIS_WINDOW = 'HARRIS'
GAUSS_WINDOW = 'GAUSS'
KAISER_WINDOW = 'KAISER'
WINDOWS_NAMES = dict(RECT='RECTANGULAR', BART='BARTLETT',
                     HANN='HANNING', HAMM='HAMMING',
                     BLACK='BLACKMAN-HARRIS', HARRIS='HARRIS',
                     GAUSS='GAUSSIAN', KAISER='KAISER')

