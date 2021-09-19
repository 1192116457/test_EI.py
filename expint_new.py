# -*- coding: iso-8859-1 -*-
# trap.py
# Trap DF (with a second order prediction)
# Copyright 2006 Giuseppe Venturini

# This file is part of the ahkab simulator.
#
# Ahkab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2 of the License.
#
# Ahkab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License v2
# along with ahkab.  If not, see <http://www.gnu.org/licenses/>.

"""
This file implements the Trapezoidal (TRAP) Differentiation Formula (DF) and a
second order prediction formula.

Module reference
----------------
"""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import numpy as np
import scipy as sp
import scipy.io as spio
from numpy.linalg import inv
import time as timeit
import pyamg
import sys

from .py3compat import range_type
from . import options
from . import dc_analysis
from . import results
from . import ticker
from . import utilities
from . import printing

from pypardiso import PyPardisoSolver
from pypardiso import spsolve

global horder
order = 9
horder = 4
m_max = options.ei_max_m
gamma = 1e-8
kappa = 1000
kvec = None
expord = 2
idx1 = []
idx2 = []
lu = None


def is_implicit():
    """Is this Differentiation Formula (DF) implicit?

    **Returns:**

    isit : boolean
        In this case, that's ``True``.
    """
    return True

def has_ff():
    """Has the method a Forward Formula for prediction?

    **Returns:**

    doesit : bool
        In this particular case, this function always returns ``True``.
    """
    return True

def ei_linear_solver(x, A, mna, D, circ, T, dT, dt, lu, time=None,
               print_steps=False, linsolver=None, debug=True, verbose=3):
    
    mna_size = A.shape[0]
    
    nv = circ.nv
#    ni = circ.ni
    tick = ticker.ticker(increments_for_step=1)
    tick.display(print_steps)
    if x is None:
        # if no guess was specified, its all zeros
        x = np.zeros((mna_size, 1))
    else:
        if x.shape[0] != mna_size:
            raise ValueError("x0s size is different from expected: got "
                             "%d-elements x0 with an MNA of size %d" %
                             (x.shape[0], mna_size))
    if T is None:
        printing.print_warning(
            "expint.ei_linear_solver called with T==None, setting T=0. BUG or no sources in circuit?")
        T = np.zeros((mna_size, 1))
        
    if kvec is not None:
        mna21 = mna[idx2, :][:, idx1]
        T2 = T[idx2] + dT[idx2] * dt

    converged = False
    iteration = 0
    tot_SAI = 0
    
    tot_gmres = 0
    dx = np.zeros((mna_size, 1))
    residual = np.zeros(dx.shape)
    xtold = x.copy()
    v_rhs = xtold.copy()
    if expord == 1:
        v_rhs = np.vstack((v_rhs, [1]))
    elif expord == 2:
        v_rhs = np.vstack((v_rhs, [[0],[1]]))
        
    tnr = timeit.time()
                
    if expord == 1:
        W = -T
    elif expord == 2:
        W = np.hstack((-dT, -T))
  
    texp1 = timeit.time()
#    tol_exp = options.vea
    aerror = np.zeros((x.shape[0], 1))
    aerror[:nv-1, 0] = options.vea
    aerror[nv-1:, 0] = options.iea
    rerror = np.zeros((x.shape[0], 1))
    rerror[:nv-1, 0] = options.ver
    rerror[nv-1:, 0] = options.ier
    tol_exp0 = aerror + rerror*abs(x)
    tol_exp = abs(tol_exp0)
#    tol_exp = tol_exp[idx1]
    xpre, err_exp, res_exp, krylovcheck, m1_SAI, solvar = expm_ArnoldiSAI(
            A, D, W, v_rhs, dt, gamma, m_max, tol_exp, kvec, linsolver=linsolver)[0:6]
    tot_SAI += m1_SAI
    texp1 = timeit.time() - texp1   
    
    if not krylovcheck:
        raise ValueError('expm_SAI does not converge') 
    
    x1 = xpre[idx1]
    rhs2 = mna21.dot(x1) + T2
    x2 = lu.solve(-rhs2)
    x[idx1] = x1
    x[idx2] = x2    
    tnr = timeit.time() - tnr 
    tick.hide(print_steps)
    error = err_exp
    residual = res_exp
    converged = True
    conv_history = []
    pss_data = []
    
    return (x, error, residual, converged, iteration, conv_history, tot_gmres, tot_SAI, pss_data)   


def ei_solve(A, mna, D, Ndc, circ, Gmin=None, x0=None, Tx0=None, lu=None, time=None, tstep=None, 
             MAXIT=None, locked_nodes=None, skip_Tt=False, bsimOpt=None, linsolver=None, verbose=3):
    """Low-level method to perform a DC solution of the circuit

    .. note::
        Typically the user calls :func:`dc_analysis.op_analysis` or
        :func:`dc_analysis.dc_analysis`, which in turn will setup all
        matrices and call this method on their behalf.

    The system we want to solve is:

    .. math::

        (mna + G_{min}) \\cdot x + N(t) + T(x, t) = 0

    Where:

    * :math:`mna` is the reduced MNA matrix with the required KVL/KCL rows
    * :math:`N` is composed by a DC part, :math:`N_{dc}`, and a dynamic
      time-dependent part :math:`N_{tran}(t)` and a time-dependent part
      :math:`T_t(t)`.
    * :math:`T(x, t)` is both time-dependent and non-linear with respect to
      the circuit solution :math:`x`, and it will be built at each iteration
      over :math:`t` and :math:`x`.

    **Parameters:**

    mna : ndarray
        The MNA matrix described above. It can be built calling
        :func:`generate_mna_and_N`. This matrix will contain the dynamic
        component due to a Differetiation Formula (DF) when this method is
        called from a transient analysis.
    Ndc : ndarray
        The DC part of :math:`N`. Also this vector may be built calling
        :func:`generate_mna_and_N`.
    circ : Circuit instance
        The circuit instance from which ``mna`` and ``N`` were built.
    Ntran : ndarray, optional
        The linear time-dependent and *dynamic* part of :math:`N`, if available.
        Notice this is typically set when a DF being applied and the method is
        being called from a transient analysis.
    Gmin : ndarray, optional
        A matrix of the same size of ``mna``, containing the minimum
        transconductances to ground. It can be built with
        :func:`build_gmin_matrix`. If not set, no Gmin matrix is used.
    x0 : ndarray or results.op_solution instance, optional
        The initial guess for the Newthon-Rhapson algorithm. If not specified,
        the all-zeros vector will be used.
    time : float scalar, optional
        The time at which any matrix evaluation done by this method will be
        performed. Do not set for DC or OP analysis, must be set for a
        transisent analysis. Notice that :math:`t=0` is not the same as DC!
    MAXIT : int, optional
        The maximum number of Newton Rhapson iterations to be performed before
        giving up. If unset, ``options.dc_max_nr_iter`` is used.
    locked_nodes : list of tuples, optional
        The nodes that need to have a well behaved, slowly varying voltage
        applied. Typically they control non-linear elements. This is generated
        by :func:`ahkab.circuit.Circuit.get_locked_nodes` and it will be
        generated for you if left unset. However, if you are doing many
        simulations of the same circuit (as it happens in a transient
        analysis), it's a good idea to generate it only once.
    skip_Tt : boolean, optional
        Do not build the :math:`T_t(t)` vector. Defaults to ``False``.
    verbose : int, optional
        The verbosity level. From 0 (silent) to 6 (debug). Defaults to 3.

    **Returns:**

    x : ndarray
        The solution, if found.
    error : ndarray
        The error associated with each solution item, if it was found.
    converged : boolean
        A flag set to True when convergence was detected.
    tot_iterations : int
        Total number of NR iterations run.
    """
    if MAXIT == None:
        MAXIT = options.dc_max_nr_iter
    if locked_nodes is None:
        locked_nodes = circ.get_locked_nodes()
    if Gmin is None:
        Gmin = 0
    if linsolver is None:
        linsolver ='splu'
        
    mna_size = mna.shape[0]
    tot_iterations = 0
    tot_gmres = 0
    tot_sai = 0
    Ndc = Ndc.toarray()
    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    Tt = np.zeros((mna_size, 1))
    if not skip_Tt:
        Tt = dc_analysis.generate_Tt(circ, time, mna_size) 
        
    # update N to include the time variable sources
        if expord == 1:
            Ndc = Ndc + Tt
        elif expord == 2:
            Ttold = dc_analysis.generate_Tt(circ, time - tstep, mna_size) 
#           Ttold = Tt
            dTt = (Tt - Ttold) / tstep
            Ndc = Ndc + Ttold
        else:
            raise ValueError
    
    # initial guess, if specified, otherwise it's zero
    if x0 is not None:
        if isinstance(x0, results.op_solution):
            x = x0.asarray()
        else:
            x = x0
    else:
        x = np.zeros((mna_size, 1))
    
    if not circ.isnonlinear:
        W = Ndc
        N_to_pass = Ndc # no Ntran for EI
        dTt_to_pass = dTt
        (x1, error, res, converged, n_iter, conv_history, n_gmres, n_sai, pss_data) = ei_linear_solver(x, A, mna, D, circ, N_to_pass, dTt_to_pass,
                                                                                    tstep, lu, time=time, print_steps=False, linsolver=linsolver, debug=True, verbose=3)
        Tx1 = None
        tot_sai += n_sai
    elif circ.isnonlinear and (not options.ei_newton):
        Jold, Txold = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
        Dold = Txold - Jold.dot(x)
        W = Dold + Ndc       
        v = np.vstack((x,[1]))
        x1, error, solved, m1, solvar = expm_ArnoldiSAI(
                    A, D, -W, v, tstep, gamma, m_max, kvec, linsolver=linsolver)     
        if kvec is not None:
            idx1 = np.nonzero(kvec)[0]
            idx2 = np.nonzero(~kvec)[0]
    #            v1 = v.copy()
            x11 = x1[idx1]
            mna21 = mna[idx2, :][:, idx1]
            mna22 = mna[idx2, :][:, idx2]
            Txold2 = Txold[idx2]
            Ndc2 = Ndc[idx2]
            rhs = mna21.dot(x11) + Txold2 + Ndc2
            if linsolver['name'] == 'pardiso':
                x12 = spsolve(mna22, -rhs, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
            elif linsolver['name'] == 'splu':
                lu = sp.sparse.linalg.splu(mna22)
                x12 = lu.solve(-rhs)
            else:
                raise ValueError('undefined linear solver')
            
            x1[idx2] = x12
    elif circ.isnonlinear and options.ei_newton:
        converged = False
        standard_solving, gmin_stepping, source_stepping = dc_analysis.get_solve_methods()
        standard_solving, gmin_stepping, source_stepping = dc_analysis.set_next_solve_method(standard_solving, gmin_stepping,
                                                                                 source_stepping, verbose)
    
        convergence_by_node = None
        printing.print_info_line(("Solving... ", 3), verbose, print_nl=False)
        if Tx0 is not None:
            Tx = Tx0
        while(not converged):
            if standard_solving["enabled"]:
                mna_to_pass = mna
                N_to_pass = Ndc # no Ntran for EI
                dTt_to_pass = dTt
            elif gmin_stepping["enabled"]:
                # print "gmin index:", str(gmin_stepping["index"])+", gmin:", str(
                # 10**(gmin_stepping["factors"][gmin_stepping["index"]]))
                printing.print_info_line(
                    ("Setting Gmin to: " + str(10 ** gmin_stepping["factors"][gmin_stepping["index"]]), 6), verbose)
                mna_to_pass = dc_analysis.build_gmin_matrix(
                    circ, 10 ** (gmin_stepping["factors"][gmin_stepping["index"]]), mna_size, verbose) + mna
                N_to_pass = Ndc
                dTt_to_pass = dTt
            elif source_stepping["enabled"]:
                printing.print_info_line(
                    ("Setting sources to " + str(source_stepping["factors"][source_stepping["index"]] * 100) + "% of their actual value", 6), verbose)
                mna_to_pass = mna + Gmin
                N_to_pass = source_stepping["factors"][source_stepping["index"]]*Ndc
                dTt_to_pass = dTt
            else:
                mna_to_pass = mna
                N_to_pass = Ndc # no Ntran for EI
                dTt_to_pass = dTt
            try:
                (x1, Tx1, error, res, converged, n_iter, conv_history, n_gmres, n_sai, pss_data) = ei_mdn_solver(x, Tx, A, mna_to_pass, D, circ, N_to_pass, dTt_to_pass,
                                                                                    tstep, MAXIT, lu, locked_nodes, 
                                                                                    time=time, print_steps=False, 
                                                                                    vector_norm=lambda v: max(abs(v)), bsimOpt=bsimOpt, 
                                                                                    linsolver=linsolver, debug=True, verbose=3)
                tot_iterations += n_iter
                tot_gmres += n_gmres
                tot_sai += n_sai
            except np.linalg.linalg.LinAlgError:
                n_iter = 0
                converged = False
                print("failed.")
                printing.print_general_error("J Matrix is singular")
            except OverflowError:
                n_iter = 0
                converged = False
                print("failed.")
                printing.print_general_error("Overflow")
    
            if not converged:
                if verbose == 6 and convergence_by_node is not None:
                    print('Nonlinear step failed')

                if n_iter == MAXIT - 1:
                    printing.print_general_error(
                        "Error: MAXIT exceeded (" + str(MAXIT) + ")")
                break
                # if dc_analysis.more_solve_methods_available(standard_solving, gmin_stepping, source_stepping):
                #     standard_solving, gmin_stepping, source_stepping = dc_analysis.set_next_solve_method(
                #                                                         standard_solving, gmin_stepping, source_stepping, verbose)                   
                # else:
                #     # print "Giving up."
                #     # x = None
                #     # error = None
                #     break
            else:
                printing.print_info_line(
                    ("[%d iterations]" % (n_iter,), 6), verbose)
                if (source_stepping["enabled"] and source_stepping["index"] != 9):
                    converged = False
                    source_stepping["index"] = source_stepping["index"] + 1
                elif (gmin_stepping["enabled"] and gmin_stepping["index"] != 9):
                    gmin_stepping["index"] = gmin_stepping["index"] + 1
                    converged = False
                else:
                    printing.print_info_line((" done.", 3), verbose)
    else:
        raise ValueError('Undefined nonlinear solver for nonlinear circuits')

    return x1, Tx1, error, res, converged, tot_iterations, tot_gmres, tot_sai, pss_data

def ei_mdn_solver(x, Tx, A, mna, D, circ, T, dT, dt, MAXIT, lu, locked_nodes, time=None,
               print_steps=False, vector_norm=lambda v: max(abs(v)),
               bsimOpt=None, linsolver=None, debug=True, verbose =3):
    
    # OLD COMMENT: FIXME REWRITE: solve through newton
    # problem is F(x)= mna*x +H(x) = 0
    # H(x) = N + T(x)
    # lets say: J = dF/dx = mna + dT(x)/dx
    # J*dx = -1*(mna*x+N+T(x))
    # dT/dx e' lo jacobiano -> g_eq (o gm)
    # print_steps = False
    # locked_nodes = get_locked_nodes(element_list)
#    mna = mna + sp.sparse.csc_matrix(sp.eye(mna.shape[0]) * 1e-4)
#    A = D + mna
    mna_size = A.shape[0]
    
    nv = circ.nv
    ni = circ.ni
    nonlinear_circuit = circ.isnonlinear
    tick = ticker.ticker(increments_for_step=1)
    tick.display(print_steps)
    if x is None:
        # if no guess was specified, its all zeros
        x = np.zeros((mna_size, 1))
    else:
        if x.shape[0] != mna_size:
            raise ValueError("x0s size is different from expected: got "
                             "%d-elements x0 with an MNA of size %d" %
                             (x.shape[0], mna_size))
    if T is None:
        printing.print_warning(
            "dc_analysis.mdn_solver called with T==None, setting T=0. BUG or no sources in circuit?")
        T = np.zeros((mna_size, 1))
    if kvec is not None:
#        D11 = D[idx1, :][:, idx1]
        mna12 = mna[idx1, :][:, idx2]
        mna21 = mna[idx2, :][:, idx1]
        mna22 = mna[idx2, :][:, idx2].tocsc()
        mna2 = mna[idx2]
        T2 = T[idx2] + dT[idx2] * dt

    converged = False
    iteration = 0
    tot_SAI = 0
    alpha = 1
   
 #### user-defined matrix exponential vector production function ############   
    def jacobvec(A, D, J, J21, v, dt, kvec=None, linsolver=None):
        nonlocal tot_SAI, alpha
        nonlocal M22#, lu22
#        nonlocal GMRES_tol
        mna_size = A.shape[0]       
        v = v.reshape(-1, 1)
        # alpha1 = 1 / np.sqrt(dt)
        alpha1 = 1 / np.sqrt(dt)
        alpha2 = (1 / dt) / alpha1
        if (kvec is not None) and (J21 is not None):
            # v2 = -lu22.solve(J21.dot(v))
            # tpardiso1 = timeit.time()
            # v2 = -spsolve(M22,J21.dot(v), factorize=True, squeeze=False, solver=linsolver['pardiso1'])
            v2 = -dc_analysis.lin_solve(M22, J21.dot(v), linsolver, options.use_sparse)
            # tpardiso1 = timeit.time() - tpardiso1
            # print('pardiso time jv2: {}'.format(tpardiso1))
            vv = np.zeros((mna_size, 1))
            vv[idx1] = v
            vv[idx2] = v2
            w1 = J.dot(vv * alpha1)
        else:
            w1 = J.dot(v)  
            vv = v
#        v1 = np.vstack((sp.zeros((mna_size, 1)), [1]))
        v1 = np.vstack((sp.zeros((mna_size, 1)), [[0],[1]]))
        W = np.hstack((w1, np.zeros((mna_size, 1)))) #phi_2 used in Jacobian
        # W = np.hstack((np.zeros((mna_size, 1)), w1))
        
        # tol_expGMRES = options.lin_GMRES_atol + \
        #                     options.lin_GMRES_rtol * np.abs(vv)
        if np.all(w1 == 0):
            x1 = np.zeros((mna_size, 1))
        else:
            tol_expGMRES = (options.lin_GMRES_atol + options.lin_GMRES_rtol  * np.abs(vv)) / 10        
            x1, error, res, solved, m1, _ = expm_ArnoldiSAI(
                    A, D, W, v1, dt, gamma, m_max, tol_expGMRES, kvec, 
                    linsolver=linsolver, ordest=False)
            tot_SAI += m1
    #        print(m1)
            if not solved:
                raise ValueError('expm_SAI in jacobvec does not converge')
            
        if (kvec is not None) and (J21 is not None):      
            y = x1[idx1] * alpha2 + v
            # print(x1[idx1].flatten(), v.flatten())
        else:
            y = sp.zeros((mna_size, 1))
            y[idx1] = x1[idx1] + v[idx1]
            if kvec is not None: y[idx2] = M22.dot(v[idx2]) 
        return y
################################################################################    
   
    conv_history = []
    niter = 0
    tot_gmres = 0
    dx = np.zeros((mna_size, 1))
    residual = np.zeros(dx.shape)
    xtold = x.copy()
    Txtold = Tx.copy()
    v_rhs = xtold.copy()
    if expord == 1:
        v_rhs = np.vstack((v_rhs, [1]))
    elif expord == 2:
        v_rhs = np.vstack((v_rhs, [[0],[1]]))
    # MAXIT = 30
#    x = spio.loadmat('xref.mat')['xref']
    while iteration < MAXIT:  # newton iteration counter
        tnr = timeit.time()
        iteration += 1
        xold = x.copy()
        dx = np.zeros((mna_size, 1))
        residual = np.zeros(dx.shape)
        tick.step()
        if options.useBsim3:               
            bsimOpt['iter'] = iteration
            # bsimOpt['first_time'] = iteration
            Cnl, J, Tx0, Tx_mexp, ECnl = dc_analysis.generate_J_and_Tx_bsim(circ, x, dt, bsimOpt)
            bsimOpt['first_time'] = 0
            Tx = Cnl.dot(x/dt) + J.dot(x) - Tx0 
            # Tx = Tx_mexp
            J = J + Cnl.multiply(1/dt)
        else:
            tnl = timeit.time()
            J, Tx = dc_analysis.generate_J_and_Tx(circ, xold, time, nojac=False)
            tnl = timeit.time() - tnl
            # sys.exit(3)
        Tx = alpha * Tx + (1 - alpha) * Txtold
        dTx = (Tx - Txtold) / dt
        
        if kvec is not None:
#            J11 = J[idx1, :][:, idx1]
#            J12 = J[idx1, :][:, idx2]
            J21 = J[idx2, :][:, idx1]
            J22 = J[idx2, :][:, idx2]
            
            ### new nonlinear formula
            M22 = (J22 + mna22).tocsc()
            M21 = mna21 + J21
            # lu22 = sp.sparse.linalg.splu(M22)
#            J2 = J[idx2]
        else:
#            J11 = J
            J21 = None
            M21 = 0
        
        if expord == 1:
#            W = -(Tx + T) 
            W = -(Txtold + T)
        elif expord == 2:
            W = np.hstack((-(dT + dTx), -(Txtold + T)))
#            W = np.hstack((-(dT), -(Tx + T)))
      
        texp1 = timeit.time()
#        tol_exp = options.vea
        tol_exp = options.vea
        aerror = np.zeros((x.shape[0], 1))
        aerror[:nv-1, 0] = options.vea
        aerror[nv-1:, 0] = options.iea
        rerror = np.zeros((x.shape[0], 1))
        rerror[:nv-1, 0] = options.ver
        rerror[nv-1:, 0] = options.ier
        tol_exp0 = aerror + rerror*abs(x)
        tol_exp = abs(tol_exp0) / 10
        
        # xpre0, err_exp, res_exp, krylovcheck, m0_SAI, solvar = expm_ArnoldiSAI(
        #         A, D, W, v_rhs, dt, gamma, m_max, tol_exp/1000, kvec, 
        #         linsolver=linsolver, ordest=False)
        
        t1 = timeit.time()
        xpre, err_exp, res_exp, krylovcheck, m1_SAI, solvar = expm_ArnoldiSAI(
                A, D, W, v_rhs, dt, gamma, m_max, tol_exp, kvec, 
                linsolver=linsolver, ordest=False)
        print(timeit.time() - t1)
        
        # # Dt = D.dot(gamma/dt)
        # xpre1, m_rat, err_rat = expm_rational(A, D, W, v_rhs, dt, gamma, 12, 5, tol_exp, kvec=kvec, linsolver=linsolver, nshift=8)
        # tot_SAI += m1_SAI
        # # sys.exit(3)
        # texp1 = timeit.time() - texp1   
        # print(np.linalg.norm(xpre - xpre1))
        # # print(np.linalg.norm(xpre1 - xpre0))
        # print([m1_SAI, m_rat])
        
        
        if not krylovcheck:
            tmp = 1
            raise ValueError('expm_SAI does not converge')
            
        rhs = xold - xpre[:mna_size]

        residual1 = res_exp #+ deltaTx[idx1] 
         
        rhs1 = rhs[idx1]
        residual[idx1] = residual1 
        
        
        x1_size = len(idx1)
        
        Jv = sp.sparse.linalg.LinearOperator(
                (x1_size, x1_size), matvec=lambda v: jacobvec(
                        A, D, J, M21, v, dt, kvec, linsolver=linsolver))
        
        # Jv_aug = sp.sparse.linalg.LinearOperator(
        #         (2 * x1_size, 2 * x1_size), matvec=lambda v: jacobvec_aug(
        #                 A, D, J, M21, v, dt, kvec, linsolver=linsolver))
        res_Krylov = []
        dx0 = sp.zeros((x1_size, 1))
        tgmres = timeit.time()
        if np.max(abs(rhs1)) == 0:
            dx1 = sp.zeros((x1_size, 1))
            niter= 0
            print('    Zero rhs, no GMRES needed')
        else:
            GMREStol = np.max((options.lin_GMRES_rtol*1, 
                                options.lin_GMRES_atol*1 / np.max(abs(rhs1)))) 
            # GMREStol = options.lin_GMRES_rtol
            GMRESmaxit = np.min((options.lin_GMRES_maxiter, x1_size))
            tot_SAI_old = tot_SAI
            (dx1, info) = pyamg.krylov.gmres(
                    Jv, -rhs1, x0=dx0, tol=GMREStol, maxiter=GMRESmaxit, 
                    residuals=res_Krylov, orthog='mgs')
            dx1 = dx1.reshape((-1, 1))
            niter = len(res_Krylov) - 1
            if info >= 0 and info <= GMRESmaxit:
                tmp = 1
                # print('    GMRES converged to {0} in {1} iterations. SAI dim = {2}'.format(GMREStol, niter, tot_SAI - tot_SAI_old))
            else:
                # print('    GMRES does not converge to {0} in {1} iterations. Min residual={2}. Break'.format(GMREStol, niter, min(abs(np.array(res_Krylov)))))
                break
        tgmres = timeit.time() - tgmres
        
        tot_gmres += niter
        dx[idx1] = dx1
        
        conv1, conv_data1 = utilities.convergence_check(x, dx, residual, nv - 1, ni)
        # dampfactor = dc_analysis.get_td(dx, locked_nodes, n=iteration)
        dampfactor = 0.5
        # dx = dx * dampfactor
        # if iteration > 5:
        #     dampfactor = 0.5
        x[idx1] += dampfactor * dx[idx1]
        if np.all(conv1) and iteration > 1:
            for it in range(MAXIT):
                J, Tx = dc_analysis.generate_J_and_Tx(circ, x, dt, nojac=False)
                J21 = J[idx2, :][:, idx1]
                J22 = J[idx2, :][:, idx2]
                M22 = mna22 + J22
                rhs2 = mna2.dot(x) + Tx[idx2] + T2
                # dx2 = spsolve(M22, -rhs2, factorize=True, squeeze=False, solver=linsolver['pardiso1'])
                dx2 = dc_analysis.lin_solve(M22, -rhs2, linsolver, options.use_sparse)
                dx[idx2] = dx2
                residual[idx2] = rhs2
                conv2, conv_data2 = utilities.convergence_check(x, dx, residual, nv - 1, ni)
                x[idx2] += dx2
                
                # print(conv_data2)
                if np.all(conv2):
                    break
        else:
            if kvec is not None:
                residual2 = mna2.dot(x) + Tx[idx2] + T2
                # rhs2 = residual2 + J21.dot(dx1) * 1
                rhs2 = residual2 + M21.dot(dx1)
                # tpardiso1 = timeit.time() 
                # dx2 = spsolve(M22, -rhs2, factorize=True, squeeze=False, solver=linsolver['param1'])
                # rhs2 = M21.dot(x[idx1] + dx1 - xtold[idx1])
                # dx2 = spsolve(M22, -rhs2, factorize=True, squeeze=False, solver=linsolver['pardiso1'])
                dx2 = dc_analysis.lin_solve(M22, -rhs2, linsolver, options.use_sparse)
                # tpardiso1 = timeit.time() - tpardiso1
                # print('pardiso time dx2: {}'.format(tpardiso1))
                # dx2 = lu22.solve(-rhs2)
                dx[idx2] = dx2
                residual[idx2] = residual2 * 1
            else:
                dx2 = 0
            conv2, conv_data2 = utilities.convergence_check(x, dx, residual, nv - 1, ni)    
            it = 1
            if kvec is not None: 
                x[idx2] += dampfactor * dx[idx2]    
        
        # dampfactor = dc_analysis.get_td(dx, locked_nodes, n=iteration)
        
 
        # conv, conv_data = utilities.convergence_check(x, dx, residual, nv - 1, ni)
        conv, conv_data = conv2, conv_data2
        conv_history.append(conv_data1)
        tnr = timeit.time() - tnr
        
        if not options.use_pss:
            print('Newton: {0} Convergence {1} with {2}. Time used: {3}s'.format([iteration, it], conv, conv_data, tnr))
#        import sys
        # if iteration > 4:
        #     sys.exit(3)
        if not nonlinear_circuit:
            converged = True
            pss_data = []
            break
        elif np.all(conv):
            converged = True
            if options.useBsim3:
                if bsimOpt['mode'] == 'dc':
                    bsimOpt['iter'] = -1
                    Cnl, J, Tx, Tx_mexp, ECnl = dc_analysis.generate_J_and_Tx_bsim(circ, x, dt, bsimOpt)
                    # M = sp.sparse.csc_matrix(mna + J)
                    # # residual = mna.dot(x) + T + Tx
                    # residual = Tx - T
                    # x = utilities.lin_solve(M, residual, linsolver, options.use_sparse)    
            else: 
                if options.use_pss:
                    _, Tx = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=True)
                    dTx = (Tx - Txtold) / dt
                    
                    # W = np.hstack((-(dT + dTx), -(Txtold + T)))
                    W = np.hstack((-(dT + dTx), -(Txtold + T + (time - dt) * dTx)))
                    v_rhs = np.zeros((mna_size, 1))
                    v_rhs = np.vstack((v_rhs, [[0],[1]]))
                    int_F, err_exp_F, res_exp_F, krylovcheck_F, m1_SAI_F, solvar_F = expm_ArnoldiSAI(
                                        A, D, W, v_rhs, dt, gamma, m_max, tol_exp, kvec, 
                                        linsolver=linsolver, ordest=False)
                    
                    # W1 = W * 0
                    # v_rhs1 = np.vstack((xtold, [[0],[1]]))
                    # eAx, err_exp_F1, res_exp_F1, krylovcheck_F, m1_SAI_F1, solvar_F = expm_ArnoldiSAI(
                    #                     A, D, W1, v_rhs1, dt, gamma, m_max, tol_exp, kvec, 
                    #                     linsolver=linsolver, ordest=False)
                    # int_F1 = xpre - eAx
                    # int_F = 0
                    pss_data = {'int_F': int_F, 'F': Tx}                   
                else:
                    _, Tx = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=True)
                    pss_data = []
                pass
            break
        if vector_norm(dx) == np.nan: #Overflow
           raise OverflowError
    tick.hide(print_steps)
    error = np.abs(dx)
    if not converged:
        print('Non-convergence data: {0}'.format(conv_data + [dampfactor]))
    else:
        conv_data = conv_data
        
    return (x, Tx, error, residual, converged, iteration, conv_history, tot_gmres, tot_SAI, pss_data)        

def expm_ArnoldiSAI(A, C, W, v, dt, gamma, m, tol, kvec=None, linsolver=None, ordest=False):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    tKrylov = timeit.time()
    
    n = len(v)
    n0 = C.shape[0]     
    order = np.round(n - n0)
    mmax = min(1 * m, n0 + order)
    v0 = v.copy()
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    converged = False

    global horder
    
    # v0[:n0] = 0
    if kvec is not None:
        v0[:n0][~kvec] = 0
    normC = False    
    if normC == True:    
        beta = abs(np.sqrt(v0[:n0].T.dot(C.dot(v0[:n0])) + v0[n0:].T.dot(v0[n0:])))
    else:       
        beta = np.linalg.norm(v0)
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, res, converged, j = np.zeros((n0,1)), np.zeros((n0,1)), True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)
    V[:, 0] = v0.flatten() / beta
    IJinv = IJINV(order, gamma)
        
    eigDebug = False
    if eigDebug:
        sHall = []   
        sHtall = []
        C0 = C * gamma
        G = A - C
        se0 = np.sort(sp.linalg.eigvals(G.todense(), -C0.todense()))*(dt)
    
    
    for j in range(mmax):
        v = V[:, j]
        # w = -invAv_aug(A, C, W, -IJinv, v, linsolver=linsolver, kvec=kvec, order=order) 
        z1 = C.dot(V[:n0, j])[:, None] 
#        if kvec is not None:
#            z1[~kvec] = 0
        z2 = IJinv.dot(V[n0:, j][:, None])
        if order > 0:
#            w2 = IJinv.dot(z2)
            wz2 = W.dot(z2)
            v1 = (z1 + wz2).reshape(-1, 1)
            tsolve = timeit.time() 
            if linsolver['name'] == 'pardiso':  
                # tpardiso = timeit.time()                  
                w1 = spsolve(A, v1, factorize=True, squeeze=False, solver=linsolver['param'])
                
            elif linsolver['name'] == 'splu':
                # lu = sp.sparse.linalg.splu(A)
                w1 = linsolver['lu'].solve(v1) 
                w1 = np.array(w1)
            else:
                pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
            tsolve = timeit.time() - tsolve
            # print('linear solve time: {}'.format(tsolve))    
            if kvec is not None:
                    w1[~kvec] = 0
            w = np.vstack((w1, z2))
        else:
            if linsolver is not None:  
                w = spsolve(A, z1, factorize=True, squeeze=True, solver=linsolver).reshape((-1, 1))
                if kvec is not None:
                    w1[~kvec] = 0                    
            else:
                pass
            
#        for l in range(2):   # double orthogonalization 
        for i in range(j + 1):
            if normC == True: 
                Cvi = np.vstack((C.dot(V[:n0, i])[:, None], V[n0:, i][:, None])) 
                h = np.dot(w.conj().T, Cvi)
            else:
                h = np.dot(w.conj().T, V[:, i])
            H[i,j] += h
            w = w - h * V[:,i][:, None]
#        
        if normC == True: 
            hj = np.sqrt(abs(w[:n0].T.dot(C.dot(w[:n0])) + w[n0:].T.dot(w[n0:])))
        else:
            hj = np.linalg.norm(w)  
        H[j+1, j] = hj
        invH = np.linalg.inv(H[:j + 1, :j + 1])
        Hjj = (np.eye(j + 1) - invH) * (dt / gamma)
        invHj = invH[j, :]      
       
        
        maxeig = np.max(np.linalg.eigvals(Hjj).real)
        # if eigDebug:
        #     sH = np.linalg.eigvals(H[:j + 1, :j + 1]).reshape(-1, 1)
        #     sHt = np.linalg.eigvals(Hjj).reshape(-1, 1)
        #     sHall.append(np.sort(sH))
        #     sHtall.append(np.sort(sHt))
            # se1 = np.linalg.eigvals(H[:j + 1, :j + 1]).reshape(-1, 1)
        # Happy breakdown
        if (hj < 1e-12) and (maxeig < 600):           
#            vm1_norm = np.linalg.norm(vm1)
            expHm = sp.linalg.expm(Hjj)
            converged = True
            err = np.zeros((n0,1))[idx1]
            res = err
            vm1 = V[:n0, j+1]
            break 
        V[:,j+1] = w[:, 0] / hj
        if (maxeig < 300) and (np.isfinite(invH).all()):  
        # if True:
            expHm = sp.linalg.expm(Hjj) 
            vm1 = V[:n0, j+1] # error
            Avm1 = A[idx1, :].dot(vm1) # residual
#
            krylovcheck, err, res = exp_error_check(
                    tol[idx1], v0[:n0], beta, gamma, hj, invHj, expHm, vm1[idx1], Avm1)  
            # krylovcheck = False
            if (krylovcheck or (j == len(idx1))) and j > 0: # error check passed
#                nonlocal kappa
#                kappa = (np.linalg.norm(err) / np.linalg.norm(err0))**(1/(m0 - j))
                converged = True                 
                break
        else:
#            [print(se) for se in seall]
#            print('something wrong. better stop')
            pass        
            
    Vj = V[:, :j + 1]  
  
    if not converged:
        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err))) 
#        y = v
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj[:, :eHm.shape[0]].dot(beta * eHm)
#        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err)))
    else:
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
        if np.isnan(y).any():
            printing.print_warning('Solution contains nan')
            
    
    tKrylov = timeit.time() - tKrylov
    
    return y, err, res, converged, j + 1, (beta, Vj, Hjj, hj, invHj, vm1)


def exp_error_check(tol, x, beta, gamma, hj, invHj, expHm, vm1, vm2):
    
#    krylovTol = np.linalg.norm(x) * options.krylovTolr + options.krylovTola
#    krylovTol = options.krylovTolr
    krylovTol = tol
    
#    err = (beta * hj * np.abs(invH[-1].dot(expHm[:,0]))) * vm1 
    const = beta * hj
    err1 = [const * np.abs(invHj.dot(expHm[:,0]))]
    
    
#    for s in (1/3, 2/3): 
#        expHms = sp.linalg.fractional_matrix_power(expHm, s)
#        err1s = const * np.abs(invHj.dot(expHms[:,0]))
##        err = abs(err1s * vm1)
#        err1.append(err1s)
   
    err1max = np.max(np.abs(err1))
    err = abs(err1max * vm1[:, None])
    res = abs(err1max * vm2[:, None])
#    err3 = abs(err1max * vm3)
    if np.all(err <= krylovTol) and np.all(res <= krylovTol):
        krylovcheck = True
    else:
        krylovcheck = False
#    err_norm = np.max(np.abs(err))
#    if err_norm < krylovTol:
#        krylovcheck = True
#    else:
#        krylovcheck = False
    
    return krylovcheck, err.reshape(-1, 1), res.reshape(-1, 1)

def exp_error_check_new(tol, x, beta, gamma, y, y0):
    
#    krylovTol = np.linalg.norm(x) * options.krylovTolr + options.krylovTola
#    krylovTol = options.krylovTolr
    krylovTol = tol
    
    dy = y - y0
    dynrm = np.linalg.norm(dy)
    ynrm = np.linalg.norm(y)
    if ynrm == 0 or dynrm == ynrm:
        errnrm = 1
    else:            
        dm = dynrm / ynrm
        err1nrm = (dm / (1 - dm)) * ynrm
        err2nrm = 1 + ynrm
        errnrm = np.min((err1nrm, err2nrm))
    
#    tmpnew.append(np.max(abs(errnew)))
    if errnrm < options.vea + options.ver * beta:
        err = np.abs(dy)
        if np.all(err <= krylovTol):
            krylovcheck = True
        else:
            krylovcheck = False
    else:
        err = np.abs(dy)
        krylovcheck = False
#    y0 = y
    
#    err_norm = np.max(np.abs(err))
#    if err_norm < krylovTol:
#        krylovcheck = True
#    else:
#        krylovcheck = False
    
    return krylovcheck, err.reshape(-1, 1)

def phim_ArnoldiSAI(A, C, v, dt, gamma, m, kvec=None, linsolver=None):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    itvl = 1
    tKrylov = timeit.time()
    mmax = 1 * m
    n = len(v)
    n0 = C.shape[0] 
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    v = v.reshape(-1, 1)
    if kvec is not None:
        v[:n0][~kvec] = 0
    beta = np.linalg.norm(v) 
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, krylovcheck, j = 0, True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, krylovcheck, j + 1, (beta, Vj, Hjj, hj, invHj, 0)
    V[:, 0] = v.flatten() / beta
    
    for j in range(mmax):
        v = C.dot(V[:n0, j]).reshape(-1, 1) 

        if linsolver['name'] == 'pardiso':                    
            w = spsolve(A, v, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
        elif linsolver['name'] == 'splu':
            lu = sp.sparse.linalg.splu(A)
            w = lu.solve(v)  
        elif linsolver['name'] == 'GMRES':
            dx, info, niter = utilities.GMRES_wrapper(A, v, v, linsolver['tol'], linsolver['maxiter'])
            dx = dx.reshape((-1, 1))
        else:
            pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
        if kvec is not None:
                w[~kvec] = 0
#        w = np.vstack((w1, w2))

        for i in range(j + 1):
            H[i,j] = np.dot(w.conj().T, V[:, i])
            w = w - H[i,j] * V[:,i][:, None]
        hj = np.linalg.norm(w)    
        H[j+1, j] = hj
        invH = np.linalg.inv(H[:j + 1, :j + 1])
        Hjj = (np.eye(j + 1) - invH) * (dt / gamma) 
        invHj = invH[j, :]
        # Happy breakdown
        if hj < 1e-12:   
            e1 = np.eye(j, 1)
#            expHm = sp.linalg.expm(Hjj)
            phiHm = (sp.linalg.expm(Hjj) - np.eye((j, j)))*(np.linalg.solve(Hjj, e1))
            err = 0
            krylovcheck = True                       
    #         fprintf('      Happy breakdown at %d steps. Time consumed %.5g s\n',j,tKrylov)
            break 
        # invH = inv(H(1:j,1:j))
        V[:,j+1] = w[:, 0] / hj
        if j >= 2 and np.mod(j,itvl) == 0:
            e1 = np.eye(j + 1, 1)
            phiHm = (sp.linalg.expm(Hjj) - np.eye(j + 1, j + 1)).dot(np.linalg.solve(Hjj, e1))
#            expHm = sp.linalg.expm(Hjj)
            
#            vm1 = np.linalg.norm(A.dot(V[:n0, j+1]))
            vm1 = A.dot(V[:n0, j+1])
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1, invH, phiHm)
            err_old = err
            if krylovcheck: # error check passed
    #             fprintf('      Krylov_SAI converges in %d steps with error %.5g. Time consumed %.5gs\n',j,err,tKrylov) 
                kappa = (np.linalg.norm(err) / np.linalg.norm(err_old))**(-1)
                break
    Vj = V[:, :j + 1]
    if err is None or not krylovcheck:
#        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err)) 
        y = v
        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err))
    else:
        eHm = phiHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
    tKrylov = timeit.time() - tKrylov
    
    return y, err, krylovcheck, j + 1, (beta, Vj, Hjj, hj, invHj, vm1)

def phim(A, v, dt, order):    
    
    n = A.shape[0]
    if order == 0:
        x = sp.linalg.expm(A * dt).dot(v)
    elif order == 1:
        x = (sp.linalg.expm(A * dt) - np.eye(n)).dot(np.linalg.inv(A).dot(v))
    elif order == 2:
        Ainv = np.linalg.inv(A)
        x = (sp.linalg.expm(A * dt) - A * dt - sp.eye(n)).dot((Ainv.dot(Ainv.dot(v))))
    else:
        raise ValueError('undefined order of phi function')
        
    return x

def expm_rational(A, C, W, v, dt, gamma, m, k, tol, kvec=None, linsolver=None, nshift=12):
    
    tKrylov = timeit.time()
    
    n = len(v)
    n0 = C.shape[0]     
    order = np.round(n - n0)
    mmax = min(5 * m, n0 + order) 
    mstart = 0
    nrestart = 0
    ntrisolve = 0
    nrestart_max = np.ceil(mmax / m)
    v0 = v.copy()
    V = np.zeros((n, mmax+1))
    H = np.zeros((mmax+1, mmax))
    converged = False
    
    alpha = dt/gamma
    zk, ck = cf(nshift)
    zk = zk - alpha
    Omega = np.diag(ck, 0)
    beta = np.linalg.norm(v0)
    Zall = [0] * nshift
    Rall = [1] * nshift
    Rall_conv = [False] * nshift
    Z = np.zeros((n, 1))
    
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, res, converged, j = np.zeros((n0,1)), np.zeros((n0,1)), True, 0
        return y, err, converged
    V[:, 0] = v0.flatten() / beta
    J = IJINV(order, gamma, inv=False)
    JI = np.linalg.inv(J - np.eye(order))
    
    while not converged:
        Zall = [0] * nshift
        Rall = [1] * nshift
        Rall_conv = [False] * nshift
        for j in range(mstart, m):
            v = V[:, j]
            w = invAv_aug(A, C, W, JI, v, linsolver=linsolver, kvec=kvec, order=order)   
            ntrisolve += 1
    #        for l in range(2):   # uncomment to enable double orthogonalization 
            for i in range(j + 1):
                h = np.dot(w.conj().T, V[:, i])
                H[i,j] = h
                w = w - h * V[:,i][:, None]
    #        
            hjplus1 = np.linalg.norm(w)  
            H[j+1, j] = hjplus1
            Hjj = H[:j+1, :j+1]
            vjplus1 = w[:, 0] / hjplus1
            V[:,j+1] = vjplus1
            E1Ri = np.eye(j+1, 1) * beta
            for i in range(nshift-1, -1, -1):
                Zi = np.linalg.solve(alpha * np.eye(j+1) - zk[i] * Hjj, E1Ri)
                Ri = np.abs((zk[i] * hjplus1 * Zi[-1]) * vjplus1)
                if np.all(Ri < tol):
                    Zall[i] = Zi
                    Rall[i] = np.linalg.norm(Ri)
                    Rall_conv[i] = True
                elif (j + 1 == m):
                    Zall[i] = Zi
                    Rall[i] = np.linalg.norm(Ri)
                else:
                    break
                
            if np.all(Rall_conv):
                Zall = np.hstack(Zall)
                Yall = V[:, :j+1].dot(Zall)
                y = np.sum(Yall.dot(Omega), 1).real
                y1 = invAv_aug(A, C, W, JI, y, linsolver=linsolver, kvec=kvec, order=order)
                Z += y1 
                converged = True
                break
            elif (j + 1 == m):
                Zall = np.hstack(Zall)
                Yall = V[:, :j+1].dot(Zall)
                y = np.sum(Yall.dot(Omega), 1).real
                y1 = invAv_aug(A, C, W, JI, y, linsolver=linsolver, kvec=kvec, order=order)
                Z += y1
            else:
                pass
            
        if (not converged) and (k > 0):
            T, Y = np.linalg.eig(Hjj)
            idx = np.argsort(abs(T))
            T = T[idx]
            Y = Y[:, idx]
            Pk = []
            for l in range(k):
                yl = Y[:, l][:, None]
                if max(abs(yl.imag)) > 1e-12:
                    Pk.extend([yl.real, yl.imag])
                else:
                    Pk.extend([yl.real])               
            mstart = len(Pk)
            Pk = np.hstack(Pk)
            # Pk = np.zeros((Y.shape[0], 2*k))
            # Pk[:, :(2*k-1):2] = Y[:, :k].real
            # Pk[:, 1::2] = Y[:, :k].imag
            Pk, _ = np.linalg.qr(Pk) 
            Pkplus1 = sp.linalg.block_diag(Pk, np.eye(1))
            H[:mstart+1, :mstart] = Pkplus1.T.dot(H[:j+2, :j+1].dot(Pk))
            V[:, :mstart+1] = V[:, :j+2].dot(Pkplus1) 
            V[:, mstart+1:] = 0
        
            tmp = 1
            nrestart += 1
            if nrestart > nrestart_max:
                print('maximum number of restarting')
        else:
            V[:, 0] = vjplus1
            beta = 1
            
                
    Vj = V[:, :j + 1]  
    tKrylov = timeit.time() - tKrylov
    print(tKrylov)
    return Z, ntrisolve+1, Rall
   
  
def invAv_aug(A, C, W, J, v, linsolver=None, kvec=None, order=2):
    # n = len(v)
    n0 = C.shape[0] 

    if len(v.shape) == 1:
        v = v[:, None]        
    
    if order == 0:
        v1 = C.dot(v)[:, None]
        z2 = []
    else:
        z1 = C.dot(v[:n0, :])
        z2 = J.dot(v[n0:, :])
        wz2 = W.dot(z2)
        v1 = (-z1 + wz2)
    if linsolver['name'] == 'pardiso':  
        w1 = spsolve(A, v1, factorize=True, squeeze=False, solver=linsolver['param'])               
    elif linsolver['name'] == 'splu':
        w1 = linsolver['lu'].solve(v1) 
    else:
        pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
    
    if len(w1.shape) == 1:
        w1 = w1[:, None] 
    if kvec is not None:
        w1[~kvec.squeeze(), :] = 0
    w = w1 if order ==0 else np.vstack((w1, z2))
    return w

def cf(n):
    """
    Compute poles {zk} and residues {ck} by the Carath´eodory–Fej´er method for the type (n, n) best approximation r∗ to exp(z) on R−
    
    Parameters
    ----------
    n : int
        order of rational approximation 

    Returns
    -------
    zk, poles
    ck, residues

    """
    K = 75 # no of Cheb coeffs
    nf = 1024 # no of pts for FFT
    w = np.exp(2j  * np.pi * np.arange(nf) / nf) # roots of unity
    t = w.real # Cheb pts (twice over)
    scl = 9 # scale factor for stability
    F = np.exp(scl * (t-1) / (t+1+1e-16)) # exp(x) transpl. to [-1,1]
    c = (np.fft.fft(F) / nf).real # Cheb coeffs of F
    f = np.polyval(c[K::-1], w) # analytic part f of F
    U, S, V = np.linalg.svd(sp.linalg.hankel(c[1:K+1])) # SVD of Hankel matrix
    V = V.conj().T
    s = S[n].conj().T # singular value
    u = U[K-1::-1, n].conj().T
    v = V[:, n].conj().T # singular vector
    zz = np.zeros(nf - K) # zeros for padding
    b = np.fft.fft(np.hstack((u, zz))) / np.fft.fft(np.hstack((v, zz))) # finite Blaschke product
    rt = f - s * (w**K) * b # extended function r-tilde
    rtc = (np.fft.fft(rt) / nf).real # its Laurent coeffs
    zr = np.roots(v) 
    qk = zr[abs(zr) > 1] # poles
    qc = np.poly(qk) # coeffs of denominator
    pt = rt * np.polyval(qc, w) # numerator
    ptc = (np.fft.fft(pt) / nf).real # coeffs of numerator
    ptc = ptc[n::-1]
    ck = 0 * qk
    for k in range(n): # calculate residues
        q = qk[k] 
        q2 = np.poly(qk[qk != q])
        ck[k] = np.polyval(ptc, q) / np.polyval(q2, q)

    zk = scl * (qk-1)**2 / (qk+1)**2 # poles in z-plane
    ck = 4 * ck * zk / (qk**2 - 1) # residues in z-plane
    
    return zk, ck  

def IJINV(order, gamma, inv=True):
    
    if order == 0:
        mat = np.array([0]) if inv else np.array([0])
    elif order == 1:
        mat = np.array([[1]]) if inv else np.array([[0]])
    elif order == 2:
        mat = np.array([[1, gamma],[0, 1]]) if inv else np.array([[0, gamma],[0, 0]])
    elif order == 3:
    	mat = np.array([[1,gamma,gamma**2], [0, 1, gamma], [0, 0, 1]]) # not complete
    else:
        raise ValueError('order higher than 3')   
        
        
    return mat