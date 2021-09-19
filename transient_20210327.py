# -*- coding: iso-8859-1 -*-
# transient.py
# Transient analysis
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

""" This module provides the methods required to perform a transient analysis.

Our problem can be written as:

.. math::

    D \\cdot dx/dt + MNA \\cdot x + T_v(x) + T_t(t) + N = 0

We need:

    1. :math:`MNA`, the static Modified Nodal Analysis matrix,
    2. :math:`N`, constant DC term,
    3. :math:`T_v(x)`, the non-linear DC term
    4. :math:`T_t(t)`, the time variant term, time dependent-sources,
       to be evaluated at each time step,
    5. The dynamic :math:`D` matrix,
    6. a differentiation method to approximate :math:`dx/dt`.

"""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import sys
import imp
import time as timeit

import numpy as np
import scipy as sp
import scipy.io as spio

from . import dc_analysis
#from . import dc_analysis
from . import implicit_euler
from . import ticker
from . import options
from . import circuit
from . import printing
from . import utilities
from . import devices
from . import memristor
from . import results
from . import expint

from pypardiso import PyPardisoSolver
from pypardiso import spsolve
from collections import deque
import copy

#pardiso.set_iparm(52, 1) 

# differentiation methods, add them here
IMPLICIT_EULER = "IMPLICIT_EULER"
TRAP = "TRAP"
GEAR1 = "GEAR1"
GEAR2 = "GEAR2"
GEAR3 = "GEAR3"
GEAR4 = "GEAR4"
GEAR5 = "GEAR5"
GEAR6 = "GEAR6"
EI = "EI"

specs = {'tran':{'tokens':({
                          'label':'tstep',
                          'pos':0,
                          'type':float,
                          'needed':True,
                          'dest':'tstep',
                          'default':None
                         },
                         {
                          'label':'tstop',
                          'pos':1,
                          'type':float,
                          'needed':True,
                          'dest':'tstop',
                          'default':None
                         },
                         {
                          'label':'tstart',
                          'pos':None,
                          'type':float,
                          'needed':False,
                          'dest':'tstart',
                          'default':0
                         },
                         {
                          'label':'uic',
                          'pos':2,
                          'type':float,
                          'needed':False,
                          'dest':'uic',
                          'default':0
                         },
                         {
                          'label':'ic_label',
                          'pos':None,
                          'type':str,
                          'needed':False,
                          'dest':'x0',
                          'default':0
                         },
                         {
                          'label':'method',
                          'pos':None,
                          'type':str,
                          'needed':False,
                          'dest':'method',
                          'default':None
                         },
                         {
                          'label':'breakpoints',
                          'pos':None,
                          'type':list,
                          'needed':False,
                          'dest':'breakpoints',
                          'default':None
                         }        
                        )
               }
           }


def transient_analysis(circ, tstart, tstep, tstop, method=options.default_tran_method, use_step_control=True, x0=None,
                       mna=None, N=None, D=None, outfile="stdout", return_req_dict=None, printvar=None, breakpoints=None, verbose=2):
    """Performs a transient analysis of the circuit described by circ.

    Parameters:
    circ: circuit instance to be simulated.
    tstart: start value. Better leave this to zero.
    tstep: the maximum step to be allowed during simulation or
    tstop: stop value for simulation
    method: differentiation method: 'TRAP' (default) or 'IMPLICIT_EULER' or 'GEARx' with x=1..6
    use_step_control: the LTE will be calculated and the step adjusted. default: True
    x0: the starting point, the solution at t=tstart (defaults to None, will be set to the OP)
    mna, N, D: MNA matrices, defaulting to None, for big circuits, reusing matrices saves time
    outfile: filename, the results will be written to this file. "stdout" means print out.
    return_req_dict:  to be documented
    verbose: verbosity level from 0 (silent) to 6 (very verbose).

    """
    if outfile == "stdout":
        verbose = verbose
    _debug = True
    if options.transient_no_step_control:
        use_step_control = False
    if _debug:
        print_step_and_lte = True
    else:
        print_step_and_lte = False

    method = method.upper() if method is not None else options.default_tran_method
    
    if options.lin_sol_method == 'pardiso':
        pardiso = PyPardisoSolver()
        pardiso.set_iparm(1, 1)
        pardiso.set_iparm(2, 0)
        #for i in range(5):
        #pardiso.set_iparm(3, 4)
        pardiso.set_iparm(10, 13)
        pardiso.set_iparm(11, 1)
        pardiso.set_iparm(13, 1)
        pardiso.set_iparm(21, 1)
        pardiso.set_iparm(25, 1)
        pardiso.set_iparm(34, 1)
        pardiso1 = PyPardisoSolver()
        pardiso1.set_iparm(1, 1)
        pardiso1.set_iparm(2, 0)
        #for i in range(5):
        #pardiso.set_iparm(3, 4)
        pardiso1.set_iparm(10, 13)
        pardiso1.set_iparm(11, 1)
        pardiso1.set_iparm(13, 1)
        pardiso1.set_iparm(21, 1)
        pardiso1.set_iparm(25, 1)
        pardiso1.set_iparm(34, 1)
        linsolver = {'name': 'pardiso', 'param': pardiso, 'param1': pardiso1}
    elif options.lin_sol_method == 'GMRES':
        linsolver = {'name': 'GMRES', 'tol': options.lin_GMRES_tol, 'maxiter': options.lin_GMRES_maxiter}
    else:
        pardiso1 = PyPardisoSolver()
        pardiso1.set_iparm(1, 1)
        pardiso1.set_iparm(2, 0)
        #for i in range(5):
        #pardiso.set_iparm(3, 4)
        pardiso1.set_iparm(10, 13)
        pardiso1.set_iparm(11, 1)
        pardiso1.set_iparm(13, 1)
        pardiso1.set_iparm(21, 1)
        pardiso1.set_iparm(25, 1)
        pardiso1.set_iparm(34, 1)
        # linsolver = {'name': 'pardiso', 'param': pardiso, 'param1': pardiso1}
        # linsolver = {'name': 'splu', 'param1': pardiso1}
        linsolver = {'name': 'splu', 'lu': None, 'pardiso1': pardiso1}
    
    tstep0 = tstep
    HMAX = min(options.hmax, max(((tstop - tstart) / 20.0, tstep))) # revised by Alex

    #check parameters
    if tstart > tstop:
        printing.print_general_error("tstart > tstop")
        sys.exit(1)
    if tstep < 0:
        printing.print_general_error("tstep < 0")
        sys.exit(1)
    if tstep == 0.0:
        tstep = 5e-11  # set a minimum nonzero step if step = 0  
    
    if verbose > 4:
        tmpstr = "Vea = %g Ver = %g Iea = %g Ier = %g max_time_iter = %g HMIN = %g" % \
        (options.vea, options.ver, options.iea, options.ier, options.transient_max_time_iter, options.hmin)
        printing.print_info_line((tmpstr, 5), verbose)

    locked_nodes = circ.get_locked_nodes()

    if print_step_and_lte:
        fname = "step_and_lte_" + method + "_" + str(tstep) + '.graph'
        flte = open(fname, "w")
        flte.write("#T\tStep\tLTE\n")

    printing.print_info_line(("Starting transient analysis: ", 3), verbose)
    printing.print_info_line(("Selected method: %s" % (method,), 3), verbose)
    #It's a good idea to call transient with prebuilt MNA and N matrix
    #the analysis will be slightly faster (long netlists).
    if mna is None or N is None:
        (mna, N) = dc_analysis.generate_mna_and_N(circ, verbose=verbose) #build MNA matrices. G, b
        
#        (mna, N) = dc_analysis.generate_mna_and_N(circ, verbose=verbose)
        mna = utilities.remove_row_and_col(mna)
        N = utilities.remove_row(N, rrow=0)
    elif not mna.shape[0] == N.shape[0]:
        printing.print_general_error("mna matrix and N vector have different number of columns.")
        sys.exit(0)
        
    N_mna = mna.shape[0]    
    if D is None:
        # if you do more than one tran analysis, output streams should be changed...
        # this needs to be fixed
        D = generate_D(circ, (N_mna, N_mna))
        D = utilities.remove_row_and_col(D)
        D = D + sp.sparse.spdiags(np.ones(circ.nv)*1e-16, 0, N_mna, N_mna, format='csr')
        if options.testcase == 'io':
            row, col, val = sp.sparse.find(D)
            val[(val < 1e-16) & (val > 0)] = 1e-16
            D = sp.sparse.csc_matrix((val, (row, col)), shape=(N_mna, N_mna))
    
#    spio.savemat('crossbar.mat',{'mna': mna, 'D': D, 'N': N})    
    # setup x0
    if x0 is None or type(x0) == int:
        printing.print_info_line(("Generating x(t=%g) = 0" % (tstart,), 5), verbose)
        # x0 = np.zeros((mna.shape[0], 1))
        # x0 = utilities.load_x0(options.testcase, circ)
        if options.testcase == '1k1': # for 1k_cell case only
            x0 = spio.loadmat('1k_ic.mat')['x']
            print('use 1k_ic.mat as initial guess')
        elif options.testcase == 'io': # for io case only
            if method == EI:
                x0_file = 'x0_io_ei_4e-10.mat'
                x0 = spio.loadmat(x0_file)['x']
            else:
                # x0_file = 'io_ic_2.mat'
                x0_file = 'x0_io_ei_4e-10.mat'
                x0 = spio.loadmat(x0_file)['x']
            print('use {} as initial guess'.format(x0_file))
        elif options.testcase == 'ram': # for ram case only
            x0 = spio.loadmat('x0_ram.mat')['x']
            print('use x0_ram.mat as initial guess')   
        elif options.testcase == 'clk': # for ram case only
            filepath = 'D:\\Research\\Circuit simulation\\Python\\oea\\clock_ahkab.op.raw' 
            icdict = utilities.load_LTSpice_raw_file(filepath)
            x0 = dc_analysis.build_x0_from_user_supplied_ic(circ, icdict)
            print('use clock_ahkab.op.raw as initial guess')
            # x0 = spio.loadmat('x0_clk.mat')['x']
            # print('use x0_clk.mat as initial guess')    
        else:
            x0 = np.zeros((N_mna, 1))
#        nvc = circ.get_nodes_number() -1 + circ.get_current_number()
#        x0[nvc:] = 0.01 # initial guess of x for memristors
        t_opsol = timeit.time()
        opsol =  results.op_solution(x=x0, error=x0, circ=circ, outfile=None)
        t_opsol = timeit.time() - t_opsol
        printing.print_info_line(("Generating x(t=%g) finished. Time used: %g s" % (tstart, t_opsol), 5), verbose)
    else:
        if isinstance(x0, results.op_solution):
            opsol = x0
            x0 = x0.asarray()
        else:
            opsol =  results.op_solution(x=x0, error=np.zeros((mna.shape[0], 1)), circ=circ, outfile=None)
        printing.print_info_line(("Using the supplied op as x(t=%g)." % (tstart,), 5), verbose)
    # spio.savemat('io_ic.mat',{'x': x0})   
    if verbose > 4:
        print("x0:")
#        opsol.print_short()

    # setup the df method
    printing.print_info_line(("Selecting the appropriate DF ("+method+")... ", 5), verbose, print_nl=False)
    if method == IMPLICIT_EULER:
        from . import implicit_euler as df
    elif method == TRAP:
        from . import trap as df
    elif method == GEAR1:
        from . import gear as df
        df.order = 1
    elif method == GEAR2:
        from . import gear as df
        df.order = 2
    elif method == GEAR3:
        from . import gear as df
        df.order = 3
    elif method == GEAR4:
        from . import gear as df
        df.order = 4
    elif method == GEAR5:
        from . import gear as df
        df.order = 5
    elif method == GEAR6:
        from . import gear as df
        df.order = 6
    elif method == EI:
        from . import expint as df
        df.order = 9
        df.m_max = options.ei_max_m
        df.gamma = (tstop - tstart) / 2000
    else:
        df = import_custom_df_module(method, print_out=(outfile != "stdout"))

    if df is None:
        sys.exit(23)

    if not df.has_ff() and use_step_control:
        printing.print_warning("The chosen DF does not support step control. Turning off the feature.")
        use_step_control = False

    printing.print_info_line(("done.", 5), verbose)

    # the step is generated automatically, but never exceed the one provided.
    if use_step_control:
        #tstep = min((tstop-tstart)/9999.0, HMAX, 100.0 * options.hmin)
        tstep = min(tstep, (tstop-tstart)/1000, HMAX)
    else:
        tstep = check_step(tstep, tstart, tstart, tstop, HMAX, breakpoints)
    printing.print_info_line(("Initial step: %g"% (tstep,), 5), verbose)

    if method != EI:
        # setup the data buffer
        # if you use the step control, the buffer has to be one point longer.
        # That's because the excess point is used by a FF in the df module to predict the next value.
        printing.print_info_line(("Setting up the buffer... ", 5), verbose, print_nl=False)
        ((max_x, max_dx), (pmax_x, pmax_dx)) = df.get_required_values()
        if max_x is None and max_dx is None:
            printing.print_general_error("df doesn't need any value?")
            sys.exit(1)
        if use_step_control:
            buffer_len = 0
            for mx in (max_x, max_dx, pmax_x, pmax_dx):
                if mx is not None:
                    buffer_len = max(buffer_len, mx)
            buffer_len += 1
            thebuffer = dfbuffer(length=buffer_len, width=3)
        else:
            if max_dx is not None:
                thebuffer = dfbuffer(length=max(max_x, max_dx) + 1, width=3)
            else:
                thebuffer = dfbuffer(length=max_x + 1, width=3)
        thebuffer.add((tstart, x0, None)) #setup the first values
        printing.print_info_line(("done.", 5), verbose) #FIXME
    
        #setup the output buffer
        if return_req_dict:
            # output_buffer = dfbuffer(length=return_req_dict["points"], width=2)
            # output_buffer.add((tstart, x0,))
            output_buffer = {'t': [tstart], 'x': [x0], 'lu': []}
        else:
            output_buffer = None

        # import implicit_euler to be used in the first iterations
        # this is because we don't have any dx when we start, nor any past point value
        if (max_x is not None and max_x > 0) or max_dx is not None:
            from . import implicit_euler
            first_iterations_number = max_x if max_x is not None else 1
            first_iterations_number = max( first_iterations_number, max_dx+1) \
                                      if max_dx is not None else first_iterations_number
        else:
            first_iterations_number = 0
            
        if max_dx is None:
            max_dx_plus_1 = None
        else:
            max_dx_plus_1 = max_dx +1
        if pmax_dx is None:
            pmax_dx_plus_1 = None
        else:
            pmax_dx_plus_1 = pmax_dx +1    
    elif method == 'EI':
         buffer_len = 2
         thebuffer = dfbuffer(length=buffer_len, width=3)
         thebuffer.add((tstep, 5, 0.0)) # (h_init, m_init, error_init)
         #setup the output buffer
         

    # setup the initial values to start the iteration:
    x1 = None
    Tx1 = None
    time = tstart
    nv = circ.get_nodes_number()

    Gmin_matrix = dc_analysis.build_gmin_matrix(circ, options.gmin, mna.shape[0], verbose)    
 
    # setup error vectors
    aerror = np.zeros((x0.shape[0], 1))
    aerror[:nv-1, 0] = options.vea
    aerror[nv-1:, 0] = options.iea
    rerror = np.zeros((x0.shape[0], 1))
    rerror[:nv-1, 0] = options.ver
    rerror[nv-1:, 0] = options.ier
    err_abs, err_rel = [], []
    
    if method == EI: 
        if options.use_standard_solve_method:
           mna = mna + Gmin_matrix
        if options.ei_reg: 
            D = utilities.remove_floating_cap(D, circ)
            
            kvec = np.diff(D.indptr) != 0 # a trick to find out the non-empty rows/cols in a csr/csc matrix
            df.kvec = kvec.reshape(-1, 1)
            df.idx1 = np.nonzero(kvec)[0]
            df.idx2 = np.nonzero(~kvec)[0]
            if options.ei_use_jac22:
                mna22 = mna[df.idx2, :][:, df.idx2].tocsc()
                lu22 = sp.sparse.linalg.splu(mna22)
                mna12 = mna[df.idx1, :][:, df.idx2].tocsc()
        else:
            df.kvec = None
            df.idx1 = np.arange(D.shape[0])[:,None]
            df.idx2 = []
    else:
        kvec = np.diff(D.indptr) != 0 # a trick to find out the non-empty rows/cols in a csr/csc matrix
        df.kvec = kvec.reshape(-1, 1)
        df.idx1 = np.nonzero(kvec)[0]
        df.idx2 = np.nonzero(~kvec)[0]    

    # tstep0 = tstep # initial step size (stored for fixed time step simulation)   
    iter_n = 0  # contatore d'iterazione
    if method != EI:
        # when to start predicting the next point
        start_pred_iter = max(*[i for i in (0, pmax_x, pmax_dx_plus_1) if i is not None])
    lte = None
    sol = results.tran_solution(circ, tstart, tstop, op=x0, method=method, outfile=outfile, printvar=printvar)
    sol.add_line(time, x0)
    printing.print_info_line(("Solving... ", 3), verbose, print_nl=False)
    total_soltime, total_iter, total_gmres_iter, n_sai, total_sai_iter, n_lu = 0, 0, 0, 0, 0, 1
    n_reverse, total_iter_waste, total_sai_waste = 0, 0, 0
    n_iter_hist = []
    reverse = False
    tick = ticker.ticker(increments_for_step=1)
    tick.display(verbose > 1)
#    
    bsimOpt = {}
    if options.useBsim3:
        bsimOpt = {'mode': 'dc', 'first_time': 0, 'iter': -1, 'outFlag': 1}
    
    if method == EI:  
        df.gamma = tstep / 2 
        # se0 = np.sort(sp.linalg.eigvals(mna.todense(), -D.todense()))*(tstep)
        D = D.dot(1/df.gamma)
        print("Scaled D by gamma = {}".format(df.gamma))
        if len(df.idx1) != 0:
            Ds = D[df.idx1, :][:, df.idx1]
            norm_Ds = sp.sparse.linalg.norm(Ds) * df.gamma
            # df.luDs = sp.sparse.linalg.splu(Ds * df.gamma)
            dsinv = 1 / (Ds.diagonal() * df.gamma)
            diagDsinv = sp.sparse.diags(dsinv, 0, format="csc")
        A = (mna + D).tocsc()
        linsolver['lu'] = sp.sparse.linalg.splu(A, permc_spec='COLAMD')
        # linsolver['luG'] = sp.sparse.linalg.splu(mna, permc_spec='COLAMD')
        if circ.isnonlinear:
            Tx1 = None
            x1 = x0
            Fbuffer = deque(maxlen=2)
            _, Tx1 = dc_analysis.generate_J_and_Tx(circ, x0, time, nojac=True)
            Fbuffer.append((time, Tx1))
        else:
            Tx1 = Tx0 = None
        if return_req_dict:
            output_buffer = {'t': [tstart], 'x': [x0], 'int_F': [], 'F': [Tx1], 'gamma': [],
                             'idx1': [], 'idx2': [], 'kvec': [], 'lu': []}
            output_buffer['idx1'] = df.idx1
            output_buffer['idx2'] = df.idx2
            output_buffer['kvec'] = df.kvec
            output_buffer['gamma'] = df.gamma
        else:
            output_buffer = None                
    else:
        if options.useBsim3:
            tmp = 1
        #     Cnl, J, Tx0, _, _ = dc_analysis.generate_J_and_Tx_bsim(circ, x0, tstep, bsimOpt) 
        else:
            _, Tx1 = dc_analysis.generate_J_and_Tx(circ, x0, time, nojac=True)
        tmp = 1
   
    # start transient simulation    
    while time < tstop - options.hmin:
        time_sol = timeit.time() 
        if method != EI:
            if iter_n < first_iterations_number:
                x_coeff, const, x_lte_coeff, prediction, pred_lte_coeff = \
                implicit_euler.get_df((thebuffer.get_df_vector()[0],), tstep, \
                predict=(use_step_control and iter_n >= start_pred_iter))
            else:
                x_coeff, const, x_lte_coeff, prediction, pred_lte_coeff = \
                    df.get_df(thebuffer.get_df_vector(), tstep, predict=(use_step_control and iter_n >= start_pred_iter))
                tmp = 1  
        else:
            if iter_n > 0:
                (tstep_old, m_old, error_old) = thebuffer.get_df_vector()[0]                
            prediction = None
            
                
        if options.transient_prediction_as_x0 and use_step_control and prediction is not None:
            x0 = prediction
        elif x1 is not None:
            x0 = x1.copy()
        
        if circ.isnonlinear and (Tx1 is not None):
            Tx0 = Tx1.copy()            
            
        printing.print_info_line(("Time step %d started. time = %g, tstep = %g" % (iter_n, time, tstep), 5), verbose)  
        # if circ.isnonlinear:
        #     if options.useBsim3:
        #         bsimOpt['mode'] = 'tran'
        #         bsimOpt['first_time'] = 1
        #         bsimOpt['iter'] = 1
        #         if method != EI:
        #             bsimOpt.update({'x_coeff': x_coeff, 'const': const})
        #         Cnl, J, Tx0, Tx0_mexp, _ = dc_analysis.generate_J_and_Tx_bsim(circ, x0, tstep, bsimOpt)  
        #         if method == EI and reverse == False:
        #             Tx0 = Cnl.dot(x0/tstep) + J.dot(x0) - Tx0 #revise Tx for EI
        #             if len(Fbuffer) >= 2:  
        #                 Fbuffer.popleft()
        #             Fbuffer.append((time, Tx0))
        #         bsimOpt['first_time'] = 0     
        #     else:
        #         _, Tx0 = dc_analysis.generate_J_and_Tx(circ, x0, time, nojac=True)
        #     if method == EI and reverse == False:
        #             Tx0 = Cnl.dot(x0/tstep) + J.dot(x0) - Tx0 #revise Tx for EI
        #             if len(Fbuffer) >= 2:  
        #                 Fbuffer.popleft()
        #             Fbuffer.append((time, Tx0))    
        # else:
        #     Tx0_mexp = 0
                       
        if method != EI:
            # df_coeff = np.round(x_coeff * tstep - 1) 
            # Ntran = mna.dot(x0 * df_coeff) + Tx0 * df_coeff + D.dot(const)
            Ndc = N   
            Ntran = D.dot(const)
            x1, Tx1, error, solved, n_iter, n_gmres, pss_data = dc_analysis.dc_solve(
                                                     mna=(mna + D.multiply(x_coeff)),
                                                     Ndc=Ndc,  Ntran=Ntran, circ=circ,
                                                     Gmin=Gmin_matrix, x0=x0.copy(), Tx0=Tx0.copy(),
                                                     time=(time + tstep), tstep=tstep,
                                                     locked_nodes=locked_nodes,
                                                     MAXIT=options.transient_max_nr_iter,
                                                     bsimOpt=bsimOpt,
                                                     linsolver=linsolver,
                                                     verbose=verbose, D=D.multiply(x_coeff)
                                                     )
                           
        else: 
            if ((tstep > 10 * df.gamma) or (tstep < 0.1 * df.gamma)) and True:
                gamma0 = copy.copy(df.gamma)
                df.gamma = tstep / 2
                D = D.dot(gamma0 / df.gamma)
                A = (mna + D).tocsc() 
                linsolver['lu'] = sp.sparse.linalg.splu(A, permc_spec='COLAMD')
                n_lu += 1
                printing.print_info_line(("Changed gamma from %g to %g. A new LU is performed" % (gamma0, df.gamma), 3), verbose)
            x1, Tx1, error, residual, solved, n_iter, n_gmres, n_sai, pss_data = expint.ei_solve(
                                                         A=A, mna=mna, D=D, Ndc=N, circ=circ,
                                                         Gmin=Gmin_matrix, x0=x0.copy(), Tx0=Tx0.copy(), lu=None,
                                                         time=(time + tstep), tstep=tstep,
                                                         locked_nodes=locked_nodes,
                                                         MAXIT=options.transient_max_nr_iter,
                                                         bsimOpt=bsimOpt, linsolver=linsolver,
                                                         verbose=verbose
                                                         )
            m = n_sai
            
        total_iter += n_iter   
        n_iter_hist.append(n_iter)
        if method == 'EI' and options.ei_newton:
            total_gmres_iter += n_gmres
            total_sai_iter += n_sai
        if solved:
            old_step = tstep #we will modify it, if we're using step control otherwise it's the same
            # step control (yeah)
            time = time + old_step
            if use_step_control and method != EI:  
                lte = None
                if x_lte_coeff is not None and pred_lte_coeff is not None and prediction is not None:
                    # this is the Local Truncation Error :)
                    lte = abs((x_lte_coeff / (pred_lte_coeff - x_lte_coeff)) * (prediction - x1))
                    tol = abs(aerror + rerror*abs(x1))
                    new_step_coeff_max = 2
                    # xx = thebuffer.get_df_vector()
                    # DD21 = ((x1 - xx[0][1]) / tstep - (x1 - xx[1][1]) / (time - xx[1][0])) / (0.5*(xx[0][0] - xx[1][0]))
                    # DD22 = ((x1 - xx[1][1]) / (time - xx[1][0]) - (x1 - xx[2][1]) / (time - xx[2][0])) / (0.5*(xx[1][0] - xx[2][0]))
                    # DD3 = (DD21 - DD22) / (0.25*(xx[0][0] - xx[2][0]))
                    # lte1 = abs(x_lte_coeff * DD3)
                    # DD21 = ((x1 - xx[0][1]) / tstep - (xx[0][1] - xx[1][1]) / (xx[0][0] - xx[1][0])) / (0.5*(time - xx[1][0]))
                    # DD22 = ((xx[0][1] - xx[1][1]) / (xx[0][0] - xx[1][0]) - (xx[1][1] - xx[2][1]) / (xx[1][0] - xx[2][0])) / (0.5*(xx[0][0] - xx[2][0]))
                    # DD3 = (DD21 - DD22) / (0.25*(time + xx[0][0] - xx[1][0] - xx[2][0]))
                    # lte2 = abs(x_lte_coeff * DD3)
                    # it should NEVER happen that new_step > 2*tstep, for stability
                    new_step_coeff = 2
                    idnz = (lte != 0)
                    step_coeff = min((tol[idnz] / lte[idnz]) ** (1.0 / (df.order + 1)))
                    
                    if (options.transient_use_aposteriori_step_control and
                        step_coeff < options.transient_aposteriori_step_threshold): #don't recalculate a x for a small change
                        time = time - old_step
                        tstep = tstep * step_coeff * 0.9
                        tstep = check_step(tstep, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                        x1, Tx1 = x0, Tx0
                        n_reverse += 1
                        total_iter_waste += n_iter
                        printing.print_info_line(("lte violation. Reverse and reduce step size to %g" % (tstep), 3), verbose)
                        continue
                    
                    new_step_coeff = max(1e-2, min((0.8 * step_coeff, new_step_coeff_max)))
                    new_step = tstep * new_step_coeff
                    tstep = check_step(new_step, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                    printing.print_info_line(("time=%g, step size = %g, lte = %g, new step = %g" % (time, old_step, lte.max(), tstep), 5), verbose)
                        # used in the next iteration
                    #print "Apriori tstep = "+str(tstep)
                else:
                    #print "LTE not calculated."
                    lte = None                   
            elif use_step_control and method == EI:
                thebuffer.add((tstep, m, np.linalg.norm(error)))
                if circ.isnonlinear == False:                  
                    new_step_coeff = tstep / tstep
                    tmp = 1
                else:                                       
                    new_step_coeff = 1.0
                    new_step_coeff_max = 4.0
                    ltex1 = None
                    if len(Fbuffer) == 2:
                        time1, time2 = Fbuffer[0][0], Fbuffer[1][0]
                        dt1, dt2 = time2 - time1, time - time2
                        DD2 = ((Tx1 - Fbuffer[1][1]) / 
                                     dt2 - (Fbuffer[1][1] - Fbuffer[0][1]) / dt1) / (0.5*(dt1 + dt2))
                        DD2s = DD2[df.idx1] - mna12.dot(lu22.solve(DD2[df.idx2]))
                        
                        CDD2s = diagDsinv.dot(DD2s) / 1000
                        # CDD2s = df.luDs.solve(DD2s) / 10
                        # CDD2s = (1 / norm_Ds) * DD2s                       
                        ltex1 = (1/12 * tstep ** 3) * abs(CDD2s)
                        
                        # ltex2 = (1/2 * tstep ** 2) * abs(lu22.solve(DD2[df.idx2]))
                        # lte = np.zeros((N_mna,1))
                        # lte[df.idx1] = ltex1
                        # lte[df.idx2] = ltex2
                        idnz1 = (ltex1 != 0)
                        # idnz2 = (ltex2 != 0)
#                        lteFs = lteFs[:, None]                    
                        tol = abs(aerror + rerror*abs(x1))
                        tol1, tol2 = tol[df.idx1], tol[df.idx2]
                        
                        step_coeff = min((tol1[idnz1] / ltex1[idnz1]) ** (1.0 / (2 + 1)))
                        # step_coeff2 = min((tol2[idnz2] / ltex2[idnz2]) ** (1.0 / (1 + 1)))
                        # step_coeff = min(step_coeff1, step_coeff2 * 10)
                        if (options.transient_use_aposteriori_step_control and
                            step_coeff < options.transient_aposteriori_step_threshold): #don't recalculate a x for a small change
                            time = time - old_step
                            tstep = tstep * step_coeff * 0.9
                            tstep = check_step(tstep, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                            x1 = x0.copy()
                            Tx1 = Tx0.copy()
                            n_reverse += 1
                            total_iter_waste += n_iter
                            total_sai_waste += n_sai
                            printing.print_info_line(("lte violation. Reverse and reduce step size to %g" % (tstep), 3), verbose)
                            continue
                        new_step_coeff = max(1e-2, min(0.8 * step_coeff, new_step_coeff_max))
                        print(new_step_coeff)
                        tmp = 1

                new_step = tstep * new_step_coeff
                lte = np.linalg.norm(error)                
                tstep = check_step(new_step, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                printing.print_info_line(("time=%g, step size = %g, lte = %g, new step = %g" % (time, old_step, lte, tstep), 5), verbose)
            else:
                # step size changes only due to breakpoints, and otherwise keeps the initial time step
                tstep = check_step(tstep0, time, tstart, tstop, HMAX, breakpoints=breakpoints)    
                
            if print_step_and_lte and lte is not None:
                #if you wish to look at the step. We print just a lte
                flte.write(str(time)+"\t"+str(old_step)+"\t"+str(np.max(lte))+"\n")
                
            # if we get here, either aposteriori_step_control is
            # disabled, or it's enabled and the error is small
            # enough. Anyway, the result is GOOD, STORE IT.     
            
            # x = x1
            # Tx = Tx1
            if use_step_control and method == EI and ltex1 is not None:
                id_max = np.argmax(ltex1[idnz1] / tol[idnz1])
                # err_abs.append((abs(ltex1[idnz1][id_max])))
                err_rel.append(np.linalg.norm(ltex1[idnz1]) / np.linalg.norm(x1[idnz1]))
                err_abs.append(np.linalg.norm(ltex1[idnz1]))
                # err_rel.append((abs(ltex1[idnz1][id_max])/abs(x1[df.idx1][idnz1][id_max])))
            elif use_step_control and method != EI and lte is not None:
                id_max = np.argmax(lte[idnz] / tol[idnz])
                # err_abs.append((abs(lte[idnz][id_max])))
                err_rel.append(np.linalg.norm(lte[idnz]) / np.linalg.norm(x1[idnz]))
                err_abs.append(np.linalg.norm(lte[idnz]))
                # err_rel.append((abs(lte[idnz][id_max])/abs(x1[idnz][id_max])))
            else:
                pass
            spio.savemat('err.mat', {'err_abs': np.array(err_abs), 'err_rel': np.array(err_rel)})
            iter_n = iter_n + 1
            reverse = False
            
            sol.add_line(time, x1) # add new solution to the result and print it to a file
            
            time_sol = timeit.time() - time_sol
            total_soltime = total_soltime + time_sol
            if linsolver['name'] == 'GMRES':
                printing.print_info_line(("Time step %d finished. NR steps: %d. GMRES iter: %d. Time used: %g" % (iter_n-1, n_iter, n_gmres, time_sol), 5), verbose)
            else:
                printing.print_info_line(("Time step %d finished. NR steps: %d. GMRES iter: %d. SAI iter: %d. Time used: %g" % (iter_n-1, n_iter, n_gmres, n_sai, time_sol), 5), verbose)

            if method != EI: # BDF
                dxdt = np.multiply(x_coeff, x1) + const
                thebuffer.add((time, x1, dxdt))
                if output_buffer is not None:
                    # output_buffer.add((x1, ))
                    # output_buffer.add((time, x1, ))
                    output_buffer['t'].append(time)
                    output_buffer['x'].append(x1)                    
                    output_buffer['lu'].append(pss_data['lu'])
            else: # EI
                if len(Fbuffer) >= 2:  
                    Fbuffer.popleft()
                Fbuffer.append((time, Tx1))
                if output_buffer is not None:
                    # output_buffer.add((x1, ))
                    # output_buffer.add((time, x1, ))
                    output_buffer['t'].append(time)
                    output_buffer['x'].append(x1)
                    output_buffer['int_F'].append(pss_data['int_F'])
                    output_buffer['F'].append(pss_data['F'])
                    if output_buffer['lu'] is None:
                        output_buffer['lu'] = df.lu
            tick.step()
#            if iter_n > 500:
#                sys.exit()
        else:
            # If we get here, Newton failed to converge. We need to reduce the step...
            if use_step_control and method != EI:
                tstep = tstep/4.0
                tstep = check_step(tstep, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                x1 = x0.copy()
                Tx1 = Tx0.copy()
                n_reverse += 1
                total_iter_waste += n_iter
                printing.print_info_line(("At %g s reducing step: %g s (convergence failed)" % (time, tstep), 5), verbose)
            elif method == EI:
                reverse = True
                tstep = tstep/4.0
                tstep = check_step(tstep, time, tstart, tstop, HMAX, breakpoints=breakpoints)
                x1 = x0.copy()
                Tx1 = Tx0.copy()
                n_reverse += 1
                total_iter_waste += n_iter
                total_sai_waste += n_sai
                printing.print_info_line(("At %g s reducing step: %g s (expint Newton failed)" % (time, tstep), 5), verbose)
            else: #we can't reduce the step
                printing.print_general_error("Can't converge with step "+str(tstep)+".")
                printing.print_general_error("Try setting --t-max-nr to a higher value or set step to a lower one.")
                solved = False
                break
        if options.transient_max_time_iter and iter_n == options.transient_max_time_iter:
            printing.print_general_error("MAX_TIME_ITER exceeded ("+str(options.transient_max_time_iter)+"), iteration halted.")
            solved = False
            break
#        if iter_n > 20:
#            sys.exit(1)

    if print_step_and_lte:
        flte.close()

    tick.hide(verbose > 1)

    if solved:
        printing.print_info_line(("done.", 3), verbose)
        printing.print_info_line(("Total system size: %d (nv=%d, ni=%d, nx=%d)" % (circ.nv + circ.ni + circ. nx, circ.nv, circ.ni, circ.nx), 3), verbose)
        printing.print_info_line(("Total solution time: %g s" % (total_soltime), 3), verbose)
        printing.print_info_line(("Total Netwon iteration: %g" % (total_iter), 3), verbose)
        printing.print_info_line(("Total GMRES iteration: %g" % (total_gmres_iter), 3), verbose)
        printing.print_info_line(("Total SAI iteration: %g" % (total_sai_iter), 3), verbose)
        printing.print_info_line(("Total time step reversal: %g" % (n_reverse), 3), verbose)
        printing.print_info_line(("Total Newton iteration wasted: %g" % (total_iter_waste), 3), verbose)
        if method == EI:
            printing.print_info_line(("Total SAI iteration wasted: %g" % (total_sai_waste), 3), verbose)
            printing.print_info_line(("Total number of LU: %g" % (n_lu), 3), verbose)
        if linsolver['name'] == 'GMRES':
            printing.print_info_line(("Total GMRES iteration: %g" % (total_gmres_iter), 3), verbose)
        printing.print_info_line(("Average time step: %g" % ((tstop - tstart)/iter_n), 3), verbose)
        # spio.savemat('n_iter_hist_EI.mat', {'n_iter_hist': n_iter_hist})
        if method != EI:
            if output_buffer:
                # ret_value = output_buffer.get_as_matrix()
                ret_value = output_buffer
            else:
                ret_value = sol
        else:
            if output_buffer:
                # ret_value = output_buffer.get_as_matrix()
                ret_value = output_buffer
            else:
                ret_value = sol  
    else:
        print("failed.")
        ret_value =  None

    return ret_value

def check_step(tstep, time, tstart, tstop, HMAX, breakpoints=None):
    """Checks the step for several common issues and corrects them.

    The following problems are checked:

    - the step must be shorter than ``HMAX``. In the context of a transient
      analysis, that usually is the time step provided by the user,
    - the step must be equal or shorter than the simulation time left (ie stop
      time minus current time),
    - the step must be longer than ``options.hmin``, the minimum allowable time
      step. If the step goes below this value, convergence problems due to
      machine precision will occur. Typically when this happens, we halt the
      simulation.

    **Parameters:**

    tstep : float
        The time step, in second, that needs to be checked.
    time : float
        The current simulation time.
    tstop : float
        The time at which the simulation ends.
    HMAX : float
        The maximum allowable time step.

    **Returns:**

    tstep : float
        The step provided if it passes the tests, a *shortened* step otherwise.

    :raises ValueError: When the step is shorter than ``option.hmin``.

    """
    
    if tstep > HMAX:
        tstep = HMAX
    if tstop - time < tstep:
        tstep = tstop - time
    elif tstep < options.hmin:
        printing.print_general_error("Step size too small: "+str(tstep))
        raise ValueError("Step size too small")
    elif len(breakpoints):
        tdiff = breakpoints - time
        idx0 = np.nonzero(np.abs(tdiff) <= options.hmin)[0] # check if time nearly overlaps with a bp
        if len(idx0) != 0:
            idx1 = idx0
            bp1 = breakpoints[idx1[0]]
        else:
            idx1 = np.nonzero(tdiff <= 0)[0]
            bp1 = breakpoints[idx1[-1]] if len(idx1) != 0 else []
            
        if len(idx1) == 0:
            bp2 = breakpoints[0]
        elif idx1[-1] == len(breakpoints) - 1:
            bp2 = []
        else:
            bp2 = breakpoints[idx1[-1] + 1]
        # bp2 = breakpoints[tdiff > options.hmin][0] if any(tdiff > 0) else []
        
        if bp1 and bp2:
            tstep_minbp = (bp2 - bp1)/options.mintimestepsbp
        elif bp2: # comment out to neglect the period before first bp or after last bp
            tstep_minbp = (bp2 - tstart) / options.mintimestepsbp
        else:
            tstep_minbp = tstep 
            
        if bp2:
            tstep_to_next_bp = bp2 - time
        else:
            tstep_to_next_bp = tstep
        # if tstep > tstep_minbp:
        #     # printing.print_info_line(('time step limited by breakpoints. Reduce from {0}s to {1}s'.format(tstep, tstep_minbp), 3), verbose, print_nl=True)
        #     print('time step limited by breakpoints. Reduce from {0}s to {1}s'.format(tstep, tstep_minbp))
        tstep = min(tstep, tstep_minbp, tstep_to_next_bp)
        if np.abs(bp2 - (tstep + time)) < options.hmin:
            tstep = tstep_to_next_bp
    tstep = max(tstep, options.hmin)
    return tstep

def generate_D(circ, shape):
    """Generates the D matrix

    For every time t, the D matrix is used (elsewhere) to solve the following system:

    .. math::

        D dx/dt + MNA x + N + T(x) = 0

    It's easy to set up the KCL law for the voltage unknowns, capacitors
    introduce stamps just like resistors do in the MNA and we know that row 1
    refers to node 1, row 2 refers to node 2, and so on

    Inductors generate, together with voltage sources, ccvs, vcvs, a additional
    line in the MNA matrix, and hence in D too.

    The current flowing through the device gets added to the x vector.

    In the case of an inductors, we have:

    .. math::

        V(n_1) - V(n_2) - V_L = 0

    Where:

    .. math::

        V_L = L dI/dt

    That's 0 (zero) in DC analysis, but not in transient analysis, where it
    needs to be differentiated.

    To understand on which line does the inductor's L*dI/dt go, we use the order
    of the elements in `circuit`: first are all voltage lines, then the current
    ones in the same order of the elements that introduce
    them. Therefore, we need to access the circuit (`circ`).

    **Parameters:**

    circ : circuit instance
        The circuit instance for which the :math:`D` matrix is computed.

    shape : tuple of ints
        The shape of the *reduced* :math:`MNA` matrix, D will be of the same
        shape.

    **Returns:**

    D : ndarray
        The *unreduced* D matrix.
    """
    if options.use_sparse:
        row, col, val = [], [], []
        nv = circ.nv# - 1
        nc = circ.ni
        i_eq = 0 #each time we find a vsource or vcvs or ccvs, we'll add one to this.
        
        for elem in circ:
            if circuit.is_elem_voltage_defined(elem) and not isinstance(elem, devices.Inductor):
                i_eq = i_eq + 1
            elif isinstance(elem, devices.Capacitor):
                row += [elem.n1, elem.n1, elem.n2, elem.n2]
                col += [elem.n1, elem.n2, elem.n1, elem.n2]
                val += [elem.value, -elem.value, -elem.value, elem.value]
            elif isinstance(elem, devices.Inductor):
                row += [nv + i_eq]
                col += [nv + i_eq]
                val += [-1 * elem.value]
#                D[ nv + i_eq, nv + i_eq ] = -1 * elem.value
                # Mutual inductors (coupled inductors)
                # need to add a -M dI/dt where I is the current in the OTHER inductor.
                if len(elem.coupling_devices):
                    for cd in elem.coupling_devices:
                        # get id+descr of the other inductor (eg. "L32")
                        other_id_wdescr = cd.get_other_inductor(elem.part_id)
                        # find its index to know which column corresponds to its current
                        other_index = circ.find_vde_index(other_id_wdescr, verbose=0)
                        # add the term.
                        row += [nv + i_eq]
                        col += [nv + other_index]
                        val += [-1 * cd.M]
#                        D[ nv + i_eq, nv + other_index ] += -1 * cd.M
                # carry on as usual
                i_eq = i_eq + 1
            elif isinstance(elem, memristor.memristor):
                row += [nv + nc + elem.mem_index]
                col += [nv + nc + elem.mem_index]
                val += [-1]
                
          # turn off cmin right now 
#        if options.cmin > 0:
#            n = shape[0] + 1 - i_eq
#            rowC, colC, valC = [], [], []
#            rowC += list(range(n))
#            colC += list(range(n))
#            valC += [options.cmin] * n
#            rowC += [0] * (n - 1)
#            colC += list(range(1, n))
#            valC += [options.cmin] * (n - 1)
#            rowC += list(range(1, n))
#            colC += [0] * (n - 1)
#            valC += [options.cmin] * (n - 1)
#            valC[0] = (n-1)*options.cmin
#            row += rowC
#            col += colC
#            val += valC
            
        D = sp.sparse.csr_matrix((val, (row, col)), shape=(shape[0]+1, shape[1]+1))
#            cmin_mat1 = sp.sparse.csr_matrix((valC, (rowC, colC)), shape=(n, n))
#            cmin_mat = sp.sparse.eye(shape[0]+1-i_eq).tolil()
#            cmin_mat[0, 1:] = 1
#            cmin_mat[1:, 0] = 1
#            cmin_mat[0, 0] = cmin_mat.shape[0]-1
#            if i_eq:
#                D[:-i_eq, :-i_eq] += options.cmin*cmin_mat
#            else:
#                D += options.cmin*cmin_mat
    else:
        D = np.zeros((shape[0]+1, shape[1]+1))
        nv = circ.get_nodes_number()# - 1
        i_eq = 0 #each time we find a vsource or vcvs or ccvs, we'll add one to this.
        for elem in circ:
            if circuit.is_elem_voltage_defined(elem) and not isinstance(elem, devices.Inductor):
                i_eq = i_eq + 1
            elif isinstance(elem, devices.Capacitor):
                n1 = elem.n1
                n2 = elem.n2
                D[n1, n1] = D[n1, n1] + elem.value
                D[n1, n2] = D[n1, n2] - elem.value
                D[n2, n2] = D[n2, n2] + elem.value
                D[n2, n1] = D[n2, n1] - elem.value
            elif isinstance(elem, devices.Inductor):
                D[ nv + i_eq, nv + i_eq ] = -1 * elem.value
                # Mutual inductors (coupled inductors)
                # need to add a -M dI/dt where I is the current in the OTHER inductor.
                if len(elem.coupling_devices):
                    for cd in elem.coupling_devices:
                        # get id+descr of the other inductor (eg. "L32")
                        other_id_wdescr = cd.get_other_inductor(elem.part_id)
                        # find its index to know which column corresponds to its current
                        other_index = circ.find_vde_index(other_id_wdescr, verbose=0)
                        # add the term.
                        D[ nv + i_eq, nv + other_index ] += -1 * cd.M
                # carry on as usual
                i_eq = i_eq + 1
    
        if options.cmin > 0:
            cmin_mat = np.eye(shape[0]+1-i_eq)
            cmin_mat[0, 1:] = 1
            cmin_mat[1:, 0] = 1
            cmin_mat[0, 0] = cmin_mat.shape[0]-1
            if i_eq:
                D[:-i_eq, :-i_eq] += options.cmin*cmin_mat
            else:
                D += options.cmin*cmin_mat
    return D

class dfbuffer:
    """This is a LIFO buffer with a method to read it all without deleting the elements.

    Newer entries are added on top of the buffer. It checks the size of the
    added elements, to be sure they are of the same size.

    **Parameters:**

    length : int
        The length of the buffer. Samples are added at index ``0``, shifting all
        the previous samples back to higher indices. Samples at an index equal
        to ``length`` (or higher) are discarded without notice.
    width : int
        The width of the buffer, every time :func:`add` is called, it must be to
        add a tuple of the same length as this parameter.
    """
    _the_real_buffer = None
    _length = 0
    _width  = 0

    def __init__(self, length, width):
        self._the_real_buffer = []
        self._length = length
        self._width = width

    def add(self, atuple):
        """Add a new data point to the buffer.

        **Parameters:**

        atuple : tuple of floats
            The data point to be added. Notice that the length of the tuple must
            agree with the width of the buffer.

        :raises ValueError: if the provided tuple and the buffer width do not
        match.

        """
        if not len(atuple) == self._width:
            raise ValueError("Attempted to add a element of wrong size to the" +
                             "LIFO buffer.")
        self._the_real_buffer.insert(0, atuple)
        if len(self._the_real_buffer) > self._length:
            self._the_real_buffer = self._the_real_buffer[:self._length]

    def get_df_vector(self):
        """Read out the contents of the buffer, without any modification

        This method, in the context of a transient analysis, returns a vector
        suitable for a differentiation formula.

        **Returns:**

        vec : list of tuples
            a list of tuples, each tuple being composed of ``width`` floats. In
            the context of a transient analysis, the list (or vector) conforms
            to the specification of the differentiation formulae.
            That is, the simulator stores in the buffer a list similar to::

                [[time(n), x(n), dx(n)], [time(n-1), x(n-1), dx(n-1)], ...]

        """
        return self._the_real_buffer

    def isready(self):
        """This shouldn't be used to determine if the buffer has enough points to
        use the df _if_ you use the step control.
        In that case, it holds even the points required for the FF.
        """
        if len(self._the_real_buffer) == self._length:
            return True
        else:
            return False

    def get_as_matrix(self):
        for vindex in range(self._width):
            for index in range(len(self._the_real_buffer)):
                if index == 0:
                    mat = self._the_real_buffer[index][vindex] # by Alex
                    if len(mat.shape) == 1:
                        single_matrix = [mat]
                    else:
                        single_matrix = [mat]
                else:
                    # single_matrix = np.concatenate((self._the_real_buffer[index][vindex], single_matrix), axis=0)
                    mat = self._the_real_buffer[index][vindex] # by Alex
                    if len(mat.shape) == 1:
                        single_matrix.append(self._the_real_buffer[index][vindex])
                        # single_matrix = np.concatenate((self._the_real_buffer[index][vindex][:, None], single_matrix), axis=1)
                    else:
                        single_matrix.append(self._the_real_buffer[index][vindex])
                        # single_matrix = np.concatenate((self._the_real_buffer[index][vindex], single_matrix), axis=1)
            if vindex == 0:
                complete_matrix = [single_matrix]
            else:
                # complete_matrix = np.concatenate((complete_matrix, single_matrix), axis=1)
                complete_matrix.append(single_matrix)
                
        return complete_matrix

def import_custom_df_module(method, print_out):
    """Imports a module that implements differentiation formula through imp.load_module
    Parameters:
    method: a string, the name of the df method module
    print_out: print to stdout some verbose messages

    Returns:
    The df module or None if the module is not found.
    """
    try:
        df = imp.load_module(imp.find_module(method.lower()))
        if print_out:
            print("Custom df module "+method.lower()+" loaded.")
    except:
        printing.print_general_error("Unrecognized method: "+method.lower()+".")
        df = None

    return df
