# -*- coding: iso-8859-1 -*-
# shooting.py
# Shooting analysis module
# Copyright 2009 Giuseppe Venturini

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

"""Periodic steady state analysis based on the shooting method."""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import numpy as np
import numpy.linalg
import scipy.io as spio
import scipy as sp
import sys
import time as timeit
import pyamg
from pypardiso import PyPardisoSolver
import pickle 


import transient
import implicit_euler
import dc_analysis
import ticker
import options
import circuit
import printing
import utilities
import results
import devices
import expint

MAXIT = 10
def shooting_analysis(circ, period, step=None, x0=None, points=None, autonomous=False,
             matrices=None, outfile='stdout', vector_norm=lambda v: max(abs(v)), printvar=None, breakpoints=None, verbose=3):
    """Performs a periodic steady state analysis based on the algorithm described in:

        Brambilla, A.; D'Amore, D., "Method for steady-state simulation of
        strongly nonlinear circuits in the time domain," *Circuits and
        Systems I: Fundamental Theory and Applications, IEEE Transactions on*,
        vol.48, no.7, pp.885-889, Jul 2001.

        http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933329&isnumber=20194

    The results have been computed again by me, the formulation is not exactly the
    same, but the idea behind the shooting algorithm is.

    This method allows us to have a period with many points without having to
    invert a huge matrix (and being limited to the maximum matrix size).

    A transient analysis is performed to initialize the solver.

    We compute the change in the last point, calculating several matrices in
    the process.
    From that, with the same matrices we calculate the changes in all points,
    starting from 0 (which is the same as the last one), then 1, ...

    Key points:

    - Only non-autonomous circuits are supported.
    - The time step is constant.
    - Implicit Euler is used as DF.

    **Parameters:**

    circ : Circuit instance
        The circuit description class.
    period : float
        The period of the solution.
    step : float, optional
        The time step between consecutive points.
        If not set, it will be computed from ``period`` and ``points``.
    points : int, optional
        The number of points to be used. If not set, it will be computed from
        ``period`` and ``step``.
    autonomous : bool, optional
        This parameter has to be ``False``, autonomous circuits are not
        currently supported.
    matrices : dict, optional
        A dictionary that may have as keys 'MNA', 'N' and 'D', with entries set
        to the corresponding MNA-formulation matrices, in case they have been
        already computed and the user wishes to save time by reusing them.
        Defaults to ``None`` (recompute).
    outfile : string, optional
        The output filename. Please use ``stdout`` (the default) to print to the
        standard output.
    verbose : boolean, optional
        Verbosity switch (0-6). It is set to zero (print errors only)
        if ``outfile`` == 'stdout'``, as not to corrupt the data.

    Notice that ``step`` and ``points`` are mutually exclusive options:

    - if ``step`` is specified, the number of points will be automatically determined.
    - if ``points`` is set, the step will be automatically determined.
    - if none of them is set, ``options.shooting_default_points`` will be used as points.

    **Returns:**

    sol : PSS solution object or ``None``
        The solution. If the circuit can't be solve, ``None`` is returned instead.
    """

    if options.default_tran_method == 'EI':    
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
        linsolver = {'name': 'splu', 'lu': None, 'pardiso1': pardiso1}
        linsolver1 = {'name': 'splu', 'lu': None, 'pardiso1': pardiso1}
   

    if outfile == "stdout":
        verbose = 0

    options.pss = True
    printing.print_info_line(("Starting periodic steady state analysis:", 3),
                             verbose)
    printing.print_info_line(("Method: shooting", 3), verbose)

    if isinstance(x0, results.op_solution):
        x0 = x0.asarray()

    # Do we have MNA T D or do we need to build them
    # MNA & T
    if (matrices is None or type(matrices) != dict or 'MNA' not in matrices or
        'Tf' not in matrices):
        # recalculate
        mna, Tf = dc_analysis.generate_mna_and_N(circ, verbose=verbose)
        mna = utilities.remove_row_and_col(mna)
        Tf = utilities.remove_row(Tf, rrow=0)
    elif not matrices['MNA'].shape[0] == matrices['Tf'].shape[0]:
        raise ValueError("MNA matrix and N vector have different number of" +
                         " rows.")
    else:
        mna, Tf = matrices['MNA'], matrices['Tf']
    # D
    if matrices is None or 'D' not in matrices or matrices['D'] is None:
        D = transient.generate_D(circ, [mna.shape[0], mna.shape[0]])
        D = utilities.remove_row_and_col(D)
        kvec = np.diff(D.indptr) != 0 # a trick to find out the non-empty rows/cols in a csr/csc matrix
        kvec = kvec.reshape(-1, 1)
        idx1 = np.nonzero(kvec)[0]
        idx2 = np.nonzero(~kvec)[0]
        idx = [idx1, idx2]
        Ds = D[idx1, :][:, idx1]
        tmp = 1
    elif not mna.shape == matrices['D'].shape:
        raise ValueError("MNA matrix and D matrix have different sizes.")
    else:
        D = matrices['D']

    points, step = utilities.check_step_and_points(step, points, period,
                                                   options.shooting_default_points)

    n_of_var = mna.shape[0]
    locked_nodes = circ.get_locked_nodes()
    
    nperiod_init = 10
    printing.print_info_line(("Starting TRAN analysis for algorithm init: " +
                              ("stop=%g, step=%g... " % (nperiod_init*points*step, step)),
                              3), verbose, print_nl=False)
    
    # load_xtran = True
    load_xtran = False
    if not load_xtran:
        x0 = None
        tstart = 0
        tstop = period
        for i in range(nperiod_init):       
            xtran = transient.transient_analysis(circ=circ, tstart=tstart, tstep=step,
                                                  tstop=tstop, method=options.default_tran_method,
                                                  x0=x0, mna=mna, N=Tf, D=D,
                                                  use_step_control=False,
                                                  outfile=outfile+".tran",
                                                  return_req_dict={"points": points},
                                                  printvar=printvar, breakpoints=breakpoints,
                                                  verbose=6)
            x0 = xtran['x'][-1]
            tstart += period
            tstop += period
            breakpoints += period
            xtran['t'] = np.array(xtran['t']) - i * period
            
        npoints = len(xtran['t'])
        # filename = 'xtran_buck_' + str(npoints) + '_' + str(options.default_tran_method) + '.mat'
        # spio.savemat(filename, xtran)
        
        filename = 'xtran_buck_' + str(npoints) + '_' + str(options.default_tran_method) + '.pkl'
        with open(filename, "wb") as output:
            pickle.dump(xtran, output)
        sys.exit(2)    
    else:
        if options.default_tran_method == 'EI':
            filename = 'xtran_buck_1010_EI.pkl'
            with open(filename, "rb") as data:
                xtran = pickle.load(data)
            # xtran = spio.loadmat('xtran_buck_1010_EI.mat')
        else:
            # xtran = spio.loadmat('xtran_buck_411.mat')
            filename = 'xtran_buck_1010_IMPLICIT_EULER.pkl'
            with open(filename, "rb") as data:
                xtran = pickle.load(data)
     # printvar=None, breakpoints=None, verbose=3
    if xtran is None:
        print("failed.")
        return None
    printing.print_info_line(("done.", 3), verbose)

    # x = []
    # for index in range(points):
    #     x.append(xtran[index * n_of_var:(index + 1) * n_of_var, 0])
    tpoints = xtran['t']
    npoints = len(tpoints)
    x = list(xtran['x'])
    tick = ticker.ticker(increments_for_step=1)

    if options.default_tran_method == 'EI':
        expint.idx1 = xtran['idx1'] 
        expint.idx2 = xtran['idx2']
        expint.kvec = np.ravel(xtran['kvec']).astype(bool)    
        # gamma = period / 2
        gamma = period/npoints/4
        Dg = D.dot(1/gamma)
        A = (mna + Dg).tocsc()
        linsolver['lu'] = sp.sparse.linalg.splu(A, permc_spec='COLAMD')
        
        gamma1 = period/4
        Dg1 = D.dot(1/gamma1)
        A1 = (mna + Dg1).tocsc()
        linsolver1['lu'] = sp.sparse.linalg.splu(A1, permc_spec='COLAMD')
        
        mna22 = mna[expint.idx2, :][:, expint.idx2].tocsc()
        lu_mna22 = sp.sparse.linalg.splu(mna22)
        mna2 = mna[expint.idx2].tocsc()
        mna21 = mna[expint.idx2, :][:, expint.idx1]
        matrices = {'A': A, 'D': Dg, 'mna21': mna21, 'lu_mna22': lu_mna22, 'gamma': gamma}
        T = dc_analysis.generate_Tt(circ, 0, n_of_var)
        T2 = T[expint.idx2]
        mna_size = A.shape[0]
        # expint.lu = xtran['lu']
        # expint.gamma = xtran['gamma']
        
##################################################################
    
    
    
##################################################################
    
    MAXIT = 20    
    converged = False
    printing.print_info_line(("Solving... ", 3), verbose, print_nl=False)
    tick.reset()
    tick.display(verbose > 2)

    iteration = 0  # newton iteration counter
    conv_counter = 0
    
    tsim_total, tjacob_total, tgmres_tot = 0.0, 0.0, 0.0
    tot_gmres = 0
    tstart = 0
    tstop = period
    x0 = xtran['x'][-1]
    step0 = step
    # D = D.toarray()
    while True:   
        tsim = timeit.time()
        if iteration == 0 and options.default_tran_method == 'EI' and True:
            # xtran = spio.loadmat('temp.mat')
            with open("temp.pkl", "rb") as data:
                xtran = pickle.load(data)
        else:
            xtran = transient.transient_analysis(circ=circ, tstart=tstart, tstep=step0,
                                                      tstop=tstop, method=options.default_tran_method,
                                                      x0=x0, mna=mna, N=Tf, D=D,
                                                      use_step_control=False,
                                                      outfile=outfile+".tran",
                                                      return_req_dict={"points": points},
                                                      printvar=printvar, breakpoints=breakpoints,
                                                      verbose=5)
        # sys.exit(3)
        tsim = timeit.time() - tsim
        # with open("temp.pkl", "wb") as output:
        #     pickle.dump(xtran, output)
        tsim_total += tsim
        tpoints = xtran['t']
        npoints = len(tpoints)
        x = list(xtran['x'])
        x0 = x[0]
        
                     
        # tjacob = timeit.time()
        # Jphi = np.eye(n_of_var)
        # for index in range(npoints - 1):
        #     step = tpoints[index + 1] - tpoints[index]
        #     Ai = lu[index].solve(D.toarray() * (1 / step))
        #     Jphi = Ai.dot(Jphi)
        # tjacob = timeit.time() - tjacob
        # tjacob_total += tjacob        
        # dx0 = np.linalg.solve((np.eye(n_of_var) - Jphi), rhs)
        if options.default_tran_method != 'EI':
            lu = xtran['lu']
            CG = xtran['CG']
        
            rhs = x[-1] - x[0]
            Jv = sp.sparse.linalg.LinearOperator((n_of_var, n_of_var), 
                                                 matvec=lambda v: jacobvec(lu, D, v, tpoints))
            Jv_inv = sp.sparse.linalg.LinearOperator((n_of_var, n_of_var), 
                                                 matvec=lambda v: jacobvec_inv(CG, D, v, tpoints))
            dx00 = sp.zeros((n_of_var, 1))
        else:           
            n_x1 = len(expint.idx1)           
            n_of_var = n_x1
            # n_x1 = n_of_var
            int_F = xtran['int_F']
            Tx = xtran['F']
            rhs = np.zeros((len(expint.idx1), 1))
            rhsn1_vector = []
            for index in range(npoints - 1):
                if index == npoints - 2:
                    rhsn1 = int_F[index][expint.idx1]
                else:
                    dtn = period - tpoints[index + 1]
                    v1 = np.zeros((mna_size, 1))   
                    v1[expint.idx1] = int_F[index][expint.idx1]
                    v1 = np.vstack((v1, [[0],[1]]))
                    W = np.zeros((mna_size, 2)) 
                    tol_expGMRES = (options.lin_GMRES_atol + options.lin_GMRES_rtol  * np.abs(v1))         
                    rhsn, error, res, solved, m1, _ = expint.expm_ArnoldiSAI(
                        A, Dg, W, v1, dtn, gamma, expint.m_max, tol_expGMRES, expint.kvec, 
                        linsolver=linsolver, ordest=False)
                    rhsn1 = rhsn[expint.idx1]
                rhsn1_vector.append(rhsn1)
                rhs += rhsn1
                # rhs += int_F[index][expint.idx1]
            epsilon = 1e-5
            # Jv = sp.sparse.linalg.LinearOperator((n_x1, n_x1), 
            #                                      matvec=lambda v: jacobvec_EI_JFNK(v, tpoints, x, Tx, 
                                                                                  # matrices, circ, epsilon, linsolver=linsolver))    
            Jv = sp.sparse.linalg.LinearOperator((n_x1, n_x1), 
                                                  matvec=lambda v: jacobvec_EI(A1, Dg1, v, period, gamma1, linsolver=linsolver1))
            dx00 = x0[expint.idx1]
            # rhs = x[-1] - x[0]
            # dx00 = sp.zeros((n_of_var, 1))
            
            
        res_Krylov = []
        
        tgmres = timeit.time()
        if np.max(abs(rhs)) == 0:
            dx0 = sp.zeros((n_of_var, 1))
            niter= 0
            print('    Zero rhs, no GMRES needed')
        else:
            GMREStol = options.lin_GMRES_rtol
            GMRESmaxit = np.min((options.lin_GMRES_maxiter, n_of_var))
            (dx0, info) = pyamg.krylov.gmres(
                    Jv, rhs, x0=dx00, tol=GMREStol, maxiter=GMRESmaxit, 
                    residuals=res_Krylov, orthog='mgs')
            dx0 = dx0.reshape((-1, 1))
            
            
            niter = len(res_Krylov) - 1
            if info == 0 and niter <= GMRESmaxit:
                print('    GMRES converged to {0} in {1} iterations. '.format(GMREStol, niter))
            else:
                print('    GMRES does not converge to {0} in {1} iterations. Min residual={2}. Break'.format(GMREStol, niter, min(abs(np.array(res_Krylov)))))
                break
        tgmres = timeit.time() - tgmres
        tgmres_tot += tgmres
        
        tot_gmres += niter
        
        td = dc_analysis.get_td(dx0.reshape(-1, 1), locked_nodes, n=-1)
        # x0 = x0 + td * dx0
        
        if options.default_tran_method == 'EI':
            x0_1 = dx0
            dx = np.zeros((A.shape[0], 1))
            dx[expint.idx1] = x0_1 - x0[expint.idx1]
            x0[expint.idx1] = x0_1
            residual = np.zeros((A.shape[0], 1))
            for it in range(MAXIT):
                J, Tx = dc_analysis.generate_J_and_Tx(circ, x0, 0, nojac=False)
                J21 = J[expint.idx2, :][:, expint.idx1]
                J22 = J[expint.idx2, :][:, expint.idx2]
                M22 = mna22 + J22
                rhs2 = mna2.dot(x0) + Tx[expint.idx2] + T2
                dx2 = sp.sparse.linalg.spsolve(M22, -rhs2)[:, None]
                dx[expint.idx2] = dx2
                residual[expint.idx2] = rhs2
                conv2, conv_data2 = utilities.convergence_check(x0, dx, residual, circ.nv - 1, circ.ni)
                x0[expint.idx2] += dx2
                
            dx0_norm = _vector_norm_wrapper(dx, vector_norm)
            print('  PSS: iter={0},  dx0_norm={1}'.format(iteration, dx0_norm)) 
            if np.all(conv2):
               converged = True
               break
            else:
               tick.step()
        else:
            x0 = x0 + td * dx0
            dx0_norm = _vector_norm_wrapper(dx0, vector_norm)
            x0_norm =_vector_norm_wrapper(x0, vector_norm)  
            print('  PSS: iter={0},  dx0_norm={1}'.format(iteration, dx0_norm))  
                                    
            if dx0_norm < (min(options.ver, options.ier) * x0_norm + min(options.vea, options.iea)):
                # and (dc_analysis.vector_norm(residuo) <
                # options.er*dc_analysis.vector_norm(x) + options.ea):
                converged = True
                
                
                break
            elif vector_norm(dx0) is np.nan:  # needs work fixme
                raise OverflowError
                # break
            else:
                conv_counter = 0
                tick.step()

        if options.shooting_max_nr_iter and iteration == options.shooting_max_nr_iter:
            printing.print_general_error("Hitted SHOOTING_MAX_NR_ITER (" +
                                         str(options.shooting_max_nr_iter) +
                                         "), iteration halted.")
            converged = False
            break
        else:
            iteration = iteration + 1

    tick.hide(verbose > 2)
    if converged:
        printing.print_info_line(("done.", 3), verbose)
        # t = np.arange(points) * step
        # t = t.reshape((1, points))
        t = tpoints
        xmat = x[0].reshape(-1, 1)
        for index in range(1, points):
            xmat = numpy.concatenate((xmat, x[index].reshape(-1, 1)), axis=1)
        sol = results.pss_solution(circ=circ, method="shooting", period=period,
                                   outfile=outfile)
        sol.set_results(t, xmat)
        # print_results(circ, x, fdata, points, step)
    else:
        print("failed.")
        sol = None
    return sol

def jacobvec(lu, D, v, tpoints):
    npoints = len(tpoints)
    v1 = v.copy()
    for index in range(npoints - 1):
        step = tpoints[index + 1] - tpoints[index]
        v1 = lu[index].solve(D.dot((1 / step)).dot(v1))
    y = v - v1
    return y

def jacobvec_inv(M, D, v, tpoints):
    npoints = len(tpoints)
    v1 = v.copy()
    for index in range(npoints - 1):
        step = tpoints[index + 1] - tpoints[index]
        v1 = M[index].solve(D.dot((1 / step)).dot(v1))
    y = v - v1
    return y

def matvec(M, D, v):
    
    return M.dot(v)

def jacobvec_EI(A, D, v, dt, gamma, linsolver=None):
    # nonlocal tot_SAI, alpha
    # nonlocal M22#, lu22
#        nonlocal GMRES_tol
    mna_size = A.shape[0]       
    v = v.reshape(-1, 1)
    # alpha1 = 1 / np.sqrt(dt)
    # alpha1 = 1 / np.sqrt(dt)
    # alpha2 = (1 / dt) / alpha1

    v1 = np.zeros((mna_size, 1))   
    v1[expint.idx1] = v
    v1 = np.vstack((v1, [[0],[1]]))
    
    W = np.zeros((mna_size, 2)) #phi_2 used in Jacobian
 
    if np.all(v == 0):
        x1 = np.zeros((mna_size, 1))
    else:
        tol_expGMRES = (options.lin_GMRES_atol + options.lin_GMRES_rtol  * np.abs(v1))         
        x1, error, res, solved, m1, _ = expint.expm_ArnoldiSAI(
                A, D, W, v1, dt, gamma, expint.m_max, tol_expGMRES, expint.kvec, 
                linsolver=linsolver, ordest=False)
        # tot_SAI += m1
#        print(m1)
        if not solved:
            raise ValueError('expm_SAI in jacobvec_EI does not converge')
        
    y = v - x1[expint.idx1]
    return y

def jacobvec_EI_JFNK(v, tpoints, x, Tx, matrices, circ, epsilon, linsolver=None):
     
    if np.all(v == 0.0):
        y = np.zeros((len(v),1))
    else:
        v = v.reshape(-1, 1)
        # vv = np.zeros((len(x[0]),1))
        # vv[expint.idx1] = v
        xN = x[-1]
        y1 = PHI(epsilon * v, tpoints, x, Tx, matrices, circ, linsolver) - xN
        y1 = y1.dot(1 / epsilon)           
        y = v - y1
    return y

def PHI(dx0, tpoints, x, Tx, matrices, circ, linsolver):
    npoints = len(tpoints)
    nx = len(dx0)
    dx_n = dx0
    _, Tx0_new = dc_analysis.generate_J_and_Tx(circ, x[0] + dx0, 0, nojac=True)
    dF_n = Tx0_new - Tx[0]
    x[0] += dx0
    Tx[0] = Tx0_new
    # MAXIT = 1
    for index in range(1, npoints):
        dt = tpoints[index] - tpoints[index - 1]
        x_nplus1 = x[index]
        for k in range(MAXIT):
            if k == 0:
                dx_nplus1 = dx_n
               
            _, Txk = dc_analysis.generate_J_and_Tx(circ, x[index] + dx_nplus1, tpoints[index], nojac=True)           
            dF_nplus1 = Txk - Tx[index]  
            
            # Jxk, Txk = dc_analysis.generate_J_and_Tx(circ, x[index], tpoints[index], nojac=False)           
            # dF_nplus1 = Jxk.dot(dx_nplus1)   
            tol_exp = options.ver * abs(dx_n)                        
            v1 = np.vstack((dx_n, [[0],[1]]))
            W = np.hstack((-(dF_nplus1 - dF_n).dot(1/dt), -dF_n))           
            dx_nplus1_temp, error_k, res_k, solved_k, m1_k, _ = expint.expm_ArnoldiSAI(
                matrices['A'], matrices['D'], W, v1, dt, matrices['gamma'], expint.m_max, tol_exp, expint.kvec, 
                linsolver=linsolver, ordest=False)
            
            # v1 = np.vstack((np.zeros((nx, 1)), [[0],[1]]))
            # W = np.hstack((-(dF_nplus1).dot(1/dt), np.zeros((nx, 1))))           
            # dx_nplus1_temp, error_k, res_k, solved_k, m1_k, _ = expint.expm_ArnoldiSAI(
            #     matrices['A'], matrices['D'], W, v1, dt, matrices['gamma'], expint.m_max, tol_exp, expint.kvec, 
            #     linsolver=linsolver, ordest=False)
            
            dx_nplus1_new = dx_nplus1.copy()
            dx_nplus1_new_1 = dx_nplus1_temp[expint.idx1]
            dx_nplus1_new[expint.idx1] = dx_nplus1_new_1
            _, Txk1 = dc_analysis.generate_J_and_Tx(circ, x[index] + dx_nplus1_new, tpoints[index], nojac=True)              
            rhs20 = -(matrices['mna21'].dot(dx_nplus1_new[expint.idx1]) + (Txk1 - Tx[index])[expint.idx2])
            
            Jxk1, Txk1 = dc_analysis.generate_J_and_Tx(circ, x[index], tpoints[index], nojac=False)   
            J2 = Jxk1[expint.idx2]
            rhs2 = -(matrices['mna21'].dot(dx_nplus1_new[expint.idx1]) + J2.dot(dx_nplus1_new))
            dx_nplus1_new_2 = matrices['lu_mna22'].solve(rhs2)
            dx_nplus1_new[expint.idx2] = dx_nplus1_new_2
            delta_dx_nplus1 = dx_nplus1_new - dx_nplus1
            norm_ddx = max(abs(delta_dx_nplus1))
            norm_dx = max(abs(dx_nplus1))
            
            if norm_ddx < (min(options.ver, options.ier) * norm_dx) or False:
                converged = True
                dx_nplus1 = dx_nplus1_new
                _, Tx_new = dc_analysis.generate_J_and_Tx(circ, x[index] + dx_nplus1, tpoints[index], nojac=True)
                break
            else:                
                converged = False
                dx_nplus1 = 1*dx_nplus1_new
                
        dx_n = dx_nplus1
        dF_n = dF_nplus1
        x[index] = x[index] + dx_nplus1
        Tx[index] = Tx_new
        
    return x[-1]
        


def _vector_norm_wrapper(vector, norm_fun):
    _max = 0
    for elem in vector:
        new_max = norm_fun(elem)
        if _max < new_max:
            _max = new_max
    return _max


def _build_static_MAass_and_MBass(mna, D, step):
    (C1, C0) = implicit_euler.get_df_coeff(step)
    MAass = mna + D * C1
    MBass = D.dot(C0)
    return (MAass, MBass)


def _build_Tass_static_vector(circ, Tf, points, step, tick, n_of_var, verbose=3):
    Tass_vector = []
    nv = circ.get_nodes_number()
    printing.print_info_line(("Building Tass...", 5), verbose, print_nl=False)

    tick.reset()
    tick.display(verbose > 2)
    for index in range(0, points):
        Tt = numpy.zeros((n_of_var,1))
        v_eq = 0
        time = index * step
        for elem in circ:
            if (isinstance(elem, devices.VSource) or
                isinstance(elem, devices.ISource)) and elem.is_timedependent:
                # time dependent source
                if isinstance(elem, devices.VSource):
                    Tt[nv - 1 + v_eq] = -1.0 * elem.V(time)
                elif isinstance(elem, devices.ISource):
                    if elem.n1:
                        Tt[elem.n1 - 1] = Tt[elem.n1 - 1] + elem.I(time)
                    if elem.n2:
                        Tt[elem.n2 - 1] = Tt[elem.n2 - 1] - elem.I(time)
            if circuit.is_elem_voltage_defined(elem):
                v_eq = v_eq + 1
        tick.step()
        Tass_vector.append(Tf + Tt)
    tick.hide(verbose > 2)
    printing.print_info_line(("done.", 5), verbose)

    return Tass_vector

def _build_Tass_static_vector_adaptive(circ, Tf, tpoints, step, tick, n_of_var, verbose=3):
    Tass_vector = []
    nv = circ.get_nodes_number()
    printing.print_info_line(("Building Tass...", 5), verbose, print_nl=False)

    tick.reset()
    tick.display(verbose > 2)
    for index in range(0, len(tpoints)):
        Tt = numpy.zeros((n_of_var,1))
        v_eq = 0
        # time = index * step
        time = tpoints[index]
        for elem in circ:
            if (isinstance(elem, devices.VSource) or
                isinstance(elem, devices.ISource)) and elem.is_timedependent:
                # time dependent source
                if isinstance(elem, devices.VSource):
                    Tt[nv - 1 + v_eq] = -1.0 * elem.V(time)
                elif isinstance(elem, devices.ISource):
                    if elem.n1:
                        Tt[elem.n1 - 1] = Tt[elem.n1 - 1] + elem.I(time)
                    if elem.n2:
                        Tt[elem.n2 - 1] = Tt[elem.n2 - 1] - elem.I(time)
            if circuit.is_elem_voltage_defined(elem):
                v_eq = v_eq + 1
        tick.step()
        Tass_vector.append(Tf + Tt)
    tick.hide(verbose > 2)
    printing.print_info_line(("done.", 5), verbose)

    return Tass_vector


def _get_variable_MAass_and_Tass(circ, xi, xi_minus_1, M, D, step, n_of_var):
    Tass = np.zeros((n_of_var, 1))
    J = np.zeros((n_of_var, n_of_var))
    (C1, C0) = implicit_euler.get_df_coeff(step)

    for elem in circ:
        # build all dT(xn)/dxn (stored in J) and T(x)
        if elem.is_nonlinear:
            output_ports = elem.get_output_ports()
            for index in range(len(output_ports)):
                n1, n2 = output_ports[index]
                ports = elem.get_drive_ports(index)
                v_ports = []
                for port in ports:
                    v = 0  # build v: remember we trashed the 0 row and 0 col of mna -> -1
                    if port[0]:
                        v = v + xi[port[0] - 1, 0]
                    if port[1]:
                        v = v - xi[port[1] - 1, 0]
                    v_ports.append(v)
                if n1:
                    Tass[n1 - 1] = Tass[n1 - 1] + elem.i(index, v_ports)
                if n2:
                    Tass[n2 - 1] = Tass[n2 - 1] - elem.i(index, v_ports)
                for pindex in range(len(ports)):
                    if n1:
                        if ports[pindex][0]:
                            J[n1 - 1, ports[pindex][0] - 1] = \
                                J[n1 - 1, ports[pindex][0] - 1] + \
                                elem.g(index, v_ports, pindex)
                        if ports[pindex][1]:
                            J[n1 - 1, ports[pindex][1] - 1] = \
                                J[n1 - 1, ports[pindex][1] - 1] - \
                                elem.g(index, v_ports, pindex)
                    if n2:
                        if ports[pindex][0]:
                            J[n2 - 1, ports[pindex][0] - 1] = \
                                J[n2 - 1, ports[pindex][0] - 1] - \
                                elem.g(index, v_ports, pindex)
                        if ports[pindex][1]:
                            J[n2 - 1, ports[pindex][1] - 1] = \
                                J[n2 - 1, ports[pindex][1] - 1] + \
                                elem.g(index, v_ports, pindex)

    Tass = Tass + (D*C1).dot(xi) + M.dot(xi) + (D * C0).dot(xi_minus_1)

    return sp.sparse.csr_matrix(J), sp.sparse.csr_matrix(Tass)


def _compute_dxN(MAass_vector, MBass, Tass_vector, n_of_var, tpoints):
    npoints = len(tpoints)
    temp_mat1 = np.eye(n_of_var)
    for index in range(npoints):
        # temp_mat1 = -sp.sparse.linalg.spsolve(MAass_vector[index], MBass.dot(temp_mat1))
        temp_mat1 = -MAass_vector[index].solve(MBass.dot(temp_mat1))
    temp_mat2 = np.zeros((n_of_var, 1))
    for index in range(npoints):
        # temp_mat3 = -sp.sparse.linalg.spsolve(MAass_vector[index], Tass_vector[index])
        temp_mat3 = -MAass_vector[index].solve(Tass_vector[index])
        for index2 in range(index + 1, npoints):
            # temp_mat3 = -sp.sparse.linalg.spsolve(MAass_vector[index2], MBass.dot(temp_mat3))
            temp_mat3 = -MAass_vector[index2].solve(MBass.dot(temp_mat3))
        temp_mat2 = temp_mat2 + temp_mat3

    # dxN = np.dot(np.linalg.inv(np.eye(n_of_var) - temp_mat1), temp_mat2)
    dxN = sp.sparse.linalg.spsolve(sp.sparse.eye(n_of_var) - temp_mat1, temp_mat2)

    return dxN[:, None]


def _compute_dx(MAass, MBass, Tass, dxi_minus_1):
    # dxi = -sp.sparse.linalg.spsolve(MAass, MBass.dot(dxi_minus_1) + Tass)
    dxi = -MAass.solve(MBass.dot(dxi_minus_1) + Tass)
    # dxi = -1 * np.linalg.inv(MAass) * (MBass * dxi_minus_1 + Tass)
    return dxi
