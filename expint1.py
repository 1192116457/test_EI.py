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
from numpy.linalg import inv
import time as timeit

from .py3compat import range_type
from . import options
from . import dc_analysis
from . import results
from . import ticker
from . import utilities
from . import printing

from pypardiso import PyPardisoSolver
from pypardiso import spsolve

order = 9
m_max = 30
gamma = 1e-8
kvec = None

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

def ei_solve(mna, D, J0, Ndc, circ, Gmin=None, x0=None, time=None, tstep=None, 
             MAXIT=None, locked_nodes=None, skip_Tt=False, linsolver=None, verbose=3):
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
#    if m is None:
#        m = m_max
#    if gamma is None:
#        gamma = gamma0
    if linsolver is None:
        linsolver ='splu'
        
    mna_size = mna.shape[0]
    nv = circ.get_nodes_number() - 1
    tot_iterations = 0
    tot_gmres = 0

    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    Tt = np.zeros((mna_size, 1))
    if not skip_Tt:
       Tt = dc_analysis.generate_Tt(circ, time, mna_size)
    # update N to include the time variable sources
    Ndc = Ndc + Tt
#    v = x
    # initial guess, if specified, otherwise it's zero
    if x0 is not None:
        if isinstance(x0, results.op_solution):
            x = x0.asarray()
        else:
            x = x0
    else:
        x = np.zeros((mna_size, 1))
    
    if not circ.isnonlinear:
        A = sp.sparse.csc_matrix(D + mna) # gamma is absorbed into D
        W = Ndc
    elif circ.isnonlinear and (not options.ei_newton):
        Jold, Txold = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
        Dold = Txold - Jold.dot(x)
        #    A = sp.sparse.csc_matrix(D + gamma* (mna + Jold))
        A = sp.sparse.csc_matrix(D + (mna + Jold)) # gamma is absorbed into D
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
    
        while(not converged):
            if standard_solving["enabled"]:
                mna_to_pass = mna + Gmin
                N_to_pass = Ndc # no Ntran for EI
            elif gmin_stepping["enabled"]:
                # print "gmin index:", str(gmin_stepping["index"])+", gmin:", str(
                # 10**(gmin_stepping["factors"][gmin_stepping["index"]]))
                printing.print_info_line(
                    ("Setting Gmin to: " + str(10 ** gmin_stepping["factors"][gmin_stepping["index"]]), 6), verbose)
                mna_to_pass = dc_analysis.build_gmin_matrix(
                    circ, 10 ** (gmin_stepping["factors"][gmin_stepping["index"]]), mna_size, verbose) + mna
                N_to_pass = Ndc
            elif source_stepping["enabled"]:
                printing.print_info_line(
                    ("Setting sources to " + str(source_stepping["factors"][source_stepping["index"]] * 100) + "% of their actual value", 6), verbose)
                mna_to_pass = mna + Gmin
                N_to_pass = source_stepping["factors"][source_stepping["index"]]*Ndc
    
            try:
#                (x, error, converged, n_iter, convergence_by_node, n_gmres) = mdn_solver(x, mna_to_pass, circ, T=N_to_pass,
#                                                                                nv=nv, print_steps=(verbose > 0), 
#                                                                                locked_nodes=locked_nodes, 
#                                                                                time=time, MAXIT=MAXIT, 
#                                                                                linsolver=linsolver,
#                                                                                debug=(verbose == 6))
                (x1, error, converged, n_iter, conv_history, n_gmres) = ei_mdn_solver(x, mna_to_pass, D, J0, circ, N_to_pass, 
                                                                                    tstep, MAXIT, nv, locked_nodes, 
                                                                                    time=time, print_steps=False, 
                                                                                    vector_norm=lambda v: max(abs(v)),
                                                                                    linsolver=linsolver, debug=True, verbose=3)
                tot_iterations += n_iter
                tot_gmres += n_gmres
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
                if dc_analysis.more_solve_methods_available(standard_solving, gmin_stepping, source_stepping):
                    standard_solving, gmin_stepping, source_stepping = dc_analysis.set_next_solve_method(
                                                                        standard_solving, gmin_stepping, source_stepping, verbose)
                    
                else:
                    # print "Giving up."
                    x = None
                    error = None
                    break
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
#        _, Txold = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=True)
#        A = sp.sparse.csc_matrix(D + mna)
        


    if options.ei_newton:
        return x1, error, converged, n_iter, conv_history, tot_gmres
    else:
        return x1, error, converged, m1, solvar, tot_gmres  

def ei_mdn_solver(x, mna, D, J0, circ, T, dt, MAXIT, nv, locked_nodes, time=None,
               print_steps=False, vector_norm=lambda v: max(abs(v)),
               linsolver=None, debug=True, verbose =3):
    
    # OLD COMMENT: FIXME REWRITE: solve through newton
    # problem is F(x)= mna*x +H(x) = 0
    # H(x) = N + T(x)
    # lets say: J = dF/dx = mna + dT(x)/dx
    # J*dx = -1*(mna*x+N+T(x))
    # dT/dx e' lo jacobiano -> g_eq (o gm)
    # print_steps = False
    # locked_nodes = get_locked_nodes(element_list)
#    mna = mna + sp.sparse.csc_matrix(sp.eye(mna.shape[0]) * 1e-4)
    A = sp.sparse.csc_matrix(D + mna)
    mna_size = A.shape[0]
#    Deye = sp.sparse.spdiags(np.ones(mna_size) * (1 / 1), [0], mna_size, mna_size, format='csr')
#    D = D+Deye
#    kvec = None
    
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
    
#    kvec = None
    if kvec is not None:
        idx1 = np.nonzero(kvec)[0]
        idx2 = np.nonzero(~kvec)[0]
#        D11 = D[idx1, :][:, idx1]
        mna11 = mna[idx1, :][:, idx1]
        mna12 = mna[idx1, :][:, idx2]
        mna21 = mna[idx2, :][:, idx1]
        mna22 = mna[idx2, :][:, idx2]
        T1 = T[idx1]
        T2 = T[idx2]
        x1 = x[idx1]
        x2 = x[idx2]
        Ts = sp.zeros(T.shape)
        Ts[idx1] = T[idx1] - mna12.dot(sp.sparse.linalg.spsolve(mna22, T[idx2])[:, None])
        
#        tspan = 200*dt
#        nt = 40
#        dt = tspan / nt
#        Cs = D[idx1, :][:, idx1] * (gamma / dt)
#        Gs = mna11 - mna12.dot(sp.sparse.linalg.spsolve(mna22, mna21))
#        Csinv = sp.sparse.linalg.inv(Cs)
#        xnew1 = x1
#        xnew2 = x1
#        x1all = np.zeros((Cs.shape[0], nt))
#        x2all = x1all.copy()
#        As = sp.sparse.bmat([[-Csinv.dot(Gs), -Csinv.dot(Ts)],[None, 0]])
#        expAs = sp.linalg.expm(As)
#        for nstep in range(nt): 
#            x1all[:, nstep] = xnew1[:, 0]
#            v1 = sp.sparse.vstack([xnew1,[1]])
#            xnew1 = expAs.dot(v1)[:-1].toarray()
#            
##            print(nstep)
#        for nstep in range(nt):    
#            x2all[:, nstep] = xnew2[:, 0]
#            rhs2 = (Cs - Gs.multiply(0.5)).dot(xnew2) - Ts        
#            xnew2 = sp.sparse.linalg.spsolve(Cs + Gs.multiply(0.5), rhs2)[:, None]
#            
#        print(np.abs(xnew1 - xnew2))
#        import matplotlib.pyplot as plt
#        data = sp.io.loadmat('x0.mat')
#        t0 = data['t0'][0]
#        x0all = data['x0']       
#        plt.figure()
#        t = dt * np.arange(nt)
#        V1 = x1all[0]
#        V2 = x2all[0]
##        plt.plot(t, V1, '-o', t, V2, '-*', ms=4, lw=1.25)
##        plt.figlegend()
##        plt.gca().legend(('exp','trap'))
##        plt.legend(loc='best')
#        plt.plot(t0, x0all[0], t, V1, '-o', t, V2, '-*', ms=4, lw=1.25)
#        plt.gca().legend(('ref', 'exp', 'trap'))
#        plt.show()
    converged = False
    iteration = 0
    iteration1 = 0
    conv_history = []
    niter = 0
    tot_gmres = 0
    dx = np.zeros(mna_size)
#    x[~kvec] = 0
    xtold = x.copy()
    _, Txtold = dc_analysis.generate_J_and_Tx(circ, xtold, time - dt, nojac=True)
    
    MAXIT = 10
    while iteration < MAXIT:  # newton iteration counter
        iteration += 1
        xold = x.copy()
        tick.step()
        J, Tx = dc_analysis.generate_J_and_Tx(circ, xold, time, nojac=False)
#        J *= 0
#        Tx *= 0
        Tx = (Tx + Txtold) * 0.5
        Txs = sp.zeros(Tx.shape)
        Txs[idx1] = Tx[idx1] - mna12.dot(sp.sparse.linalg.spsolve(mna22, Tx[idx2])[:, None])

        w_rhs = -(Txs + Ts)
#        w_rhs = w_rhs[idx1]
#        w_rhs = -(0.5 * (Tx + Txtold) + T + mna.dot(xtmp))
#        w_rhs[~kvec] = 0
#        v_rhs = np.vstack((sp.zeros((mna_size, 1)), [1]))
        
        
        v_rhs = np.vstack((xtold, [1]))
        
        xold1, err1_krylov, krylovcheck, m1_krylov, solvar = expm_ArnoldiSAI(
                A, D, w_rhs, v_rhs, dt, gamma, m_max, kvec, linsolver=linsolver)[0:5]
        if not krylovcheck:
            raise ValueError('expm_SAI does not converge')
#        res1, err1_krylov, krylovcheck, m1_krylov, solvar = expm_ArnoldiSAI(A, D, w_rhs, v_rhs, dt, gamma, m_max, kvec, linsolver=linsolver)[0:5]
#        beta, Vj, Hjj, hj, invHj, vm1 = solvar
#        expHm = sp.linalg.expm(Hjj)
#        err1_vec = (beta) * hj * vm1 * (invHj.dot(expHm[:,0])) 
##        M = (-sp.sparse.linalg.inv(D * gamma) * mna * dt).todense()
#        Dt = D.todense() * gamma
#        s0, Q0 = sp.linalg.eig(mna.todense(), -(Dt / dt))
#        Hjj = solvar[2]
#        s, Q = sp.linalg.eig(Hjj)
##        res0 = phim(M, np.linalg.inv(D.todense() * gamma) * (w_rhs), 1, 1)
##        err_res = res1[:mna_size] - res0
#        residual = res1[:mna_size] + xold - xtold 
        residual = xold[idx1] - xold1[:-1] 
        residual1 = residual[idx1]
#        residual[~kvec] = 0
# 
        x1_size = len(idx1)
        Jv = sp.sparse.linalg.LinearOperator((x1_size, x1_size), matvec=lambda v: jacobvec(A, D, J.multiply(0.5), mna, v, dt, kvec, linsolver=linsolver)[0])
        dx0 = sp.zeros((x1_size, x1_size))
        dx1, info, niter = utilities.GMRES_wrapper(Jv, -residual1, dx0, options.lin_GMRES_tol, options.lin_GMRES_maxiter)
        if info == 0:
            print('GMRES converged to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
        else:
            print('GMRES doesn not converge to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
        tot_gmres += niter
        dx1 = dx.reshape((-1, 1))
        
#        dx[~kvec] = 0
#        dx1 = dx[idx1]
        
        dampfactor = dc_analysis.get_td(dx, locked_nodes, n=iteration)
        x[idx1] = x[idx1] + dampfactor * dx1
        
        if kvec is not None:
            x1 = x[idx1]     
            b2 = mna21.dot(x1) + T2 + 0.5 * Txtold[idx2]
            iteration1 = 0
            while iteration1 < 1: 
                iteration1 += 1
                x2 = x[idx2]
                Jnew, Txnew = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
                Jnew22 = Jnew[idx2, :][:, idx2]
                Txnew2 = Txnew[idx2]
#                J2 = mna22 + Jnew22
#                residual2 = mna22.dot(x2) + Txnew2 + b2
                J2 = mna22 + 0.5 * Jnew22
                residual2 = mna22.dot(x2) + 0.5 * Txnew2 + b2
                if linsolver['name'] == 'pardiso':
                    dx2 = spsolve(J2, -residual2, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
                elif linsolver['name'] == 'splu':
                    lu = sp.sparse.linalg.splu(J2)
                    dx2 = lu.solve(-residual2)
                else:
                    raise ValueError('undefined linear solver')
                x2 = x2 + dx2
                x[idx2] = x2 
                dx_norm = np.linalg.norm(dx2)
                rhs_norm = np.linalg.norm(residual2)
                conv2, conv_data2 = utilities.custom_convergence_check(x2, dx2, residual2, 
                                                 options.ver, options.vea, options.ver, debug=False)
                
                if conv2:
                    break
        if iteration1 >= MAXIT:
            print('Newton for x2 does not converge in {0} iters with dx={1}, rhs={2}'.format(iteration1, dx_norm, rhs_norm))
        residual[idx2] = residual2
        dx[idx2] = dx2
        print([np.linalg.norm(residual), np.linalg.norm(dx)])
        conv, conv_data = utilities.convergence_check(x, dx, residual, nv - 1, ni)
        conv_history.append(conv_data)
#        print('Nonlinear iter: {0} Convergence data: {1}'.format(iteration, conv_data))
        if not nonlinear_circuit:
            converged = True
            break
        elif conv:
            converged = True
            break
        # if vector_norm(dx) == np.nan: #Overflow
        #   raise OverflowError
    tick.hide(print_steps)
    if not converged:
        # re-run the convergence check, only this time get the results
        # by node, so we can show to the users which nodes are misbehaving.
#        converged, convergence_by_node = convergence_check(
#            x, dx, residual, nv - 1, ni, debug=True)
        print('Non-convergence data: {0}'.format(conv_data + [dampfactor]))
#        convergence_by_node = []
    else:
        conv_data = conv_data
    return (x, residual, converged, iteration, conv_history, tot_gmres)    


def jacobvec(A, D, J, mna, v, dt, kvec=None, linsolver=None): 
    mna_size = A.shape[0]
    v = v.reshape(-1, 1)
#    v[~kvec] = 0
    
    if kvec is not None:
        idx1 = np.nonzero(kvec)[0]
        idx2 = np.nonzero(~kvec)[0]
        J11 = J[idx1, :][:, idx1]
        J21 = J[idx2, :][:, idx1]
        mna12 = mna[idx1, :][:, idx2]
        mna22 = mna[idx2, :][:, idx2]
        v1 = v[idx1]
        w1 = sp.zeros(v.shape)
        w1[idx1] = J11.dot(v1) - mna12.dot(sp.sparse.linalg.spsolve(mna22, J21.dot(v1))[:, None])
    else:
        w1 = J.dot(v)
#    if kvec is not None:
#        w1[~kvec] = 0
    v1 = np.vstack((sp.zeros((mna_size, 1)), [1]))
#    x1, error, solved, m1, solvar = phim_ArnoldiSAI(
#                    A, sp.sparse.eye(mna_size), v1, dt, gamma, m_max, kvec, linsolver=linsolver)
    x1, error, solved, m1, solvar = expm_ArnoldiSAI(A, D, w1, v1, dt, gamma, m_max, kvec, linsolver=linsolver)
    if not solved:
        raise ValueError('expm_SAI in jacobvec does not converge')
#        print('expm_SAI in jacobvec does not converge in {0} with error {1}'.format(m1, error))
#    M = -sp.sparse.linalg.inv(D * gamma) * mna
#    x0 = phim(M, sp.sparse.linalg.inv(D * gamma) * w1, dt, 1)
#    err_x = x1[:mna_size] - x0
    y = x1[:mna_size] + v
    y = y[idx1]
#    if kvec is not None:
#        y[~kvec] = 0
        
    return (y, error, solved, m1, solvar)
    

def expm_ArnoldiSAI(A, C, W, v, dt, gamma, m, kvec=None, linsolver=None):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    itvl = 1
    tKrylov = timeit.time()
    mmax = 1 * m
    n = len(v)
    n0 = C.shape[0] 
    order = np.round(n - n0)
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    converged = False
    if kvec is not None:
        v[:n0][~kvec] = 0
    beta = np.linalg.norm(v)
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, converged, j = 0, True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)
    V[:, 0] = v.flatten() / beta
    if order == 0:
        IJinv = 0
    elif order == 1:
        IJinv = 1
    elif order == 2:
        IJinv = np.array([[1, gamma],[0, 1]])
    elif order == 3:
    	IJinv = np.array([[1,gamma,gamma**2], [0, 1, gamma], [0, 0, 1]])
    else:
        raise ValueError('order higher than 3')
    # Ce = blkdiag(C,eye(order))
#    yold = v[:n0]
    for j in range(mmax):
        z1 = C.dot(V[:n0, j]) 
        z2 = V[n0:, j]
        if order > 0:
            w2 = np.multiply(IJinv, z2)
            v1 = (z1 + W.dot(w2)).reshape(-1, 1)
#            if kvec is not None:
#                v1[~kvec] = 0
            if linsolver['name'] == 'pardiso':                    
                w1 = spsolve(A, v1, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
            elif linsolver['name'] == 'splu':
                lu = sp.sparse.linalg.splu(A)
                w1 = lu.solve(v1)  
            elif linsolver['name'] == 'GMRES':
                dx, info, niter = utilities.GMRES_wrapper(A, v1, v1, linsolver['tol'], linsolver['maxiter'])
                dx = dx.reshape((-1, 1))
            else:
                pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
            if kvec is not None:
                    w1[~kvec] = 0
            w = np.vstack((w1, w2))
        else:
            if linsolver is not None:  
                w = spsolve(A, z1, factorize=True, squeeze=True, solver=linsolver).reshape((-1, 1))
                if kvec is not None:
                    w1[~kvec] = 0
                    
            else:
                pass
#                w = Q*(U\(L\(P*z1)))
    #         w = Q*(U\(L\(P*z1)))
        for l in range(2):    
            for i in range(j + 1):
                h = np.dot(w.conj().T, V[:, i])
                H[i,j] += h
                w = w - h * V[:,i][:, None]
        hj = np.linalg.norm(w)    
        H[j+1, j] = hj
        invH = np.linalg.inv(H[:j + 1, :j + 1])
        Hjj = (np.eye(j + 1) - invH) * (dt / gamma)
        invHj = invH[j, :]
        # Happy breakdown
        if hj < 1e-12:           
            expHm = sp.linalg.expm(Hjj)
            vm1 = A.dot(V[:n0, j+1])
            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1_norm, invH, expHm)
            converged = True
            break 
        # invH = inv(H(1:j,1:j))
        V[:,j+1] = w[:, 0] / hj
        if j >= 0 and np.mod(j,itvl) == 0:
            expHm = sp.linalg.expm(Hjj)
    #         c = norm(w-Ce\(Ge*(gamma*(w)))) # h(j+1,j)*(I-gamma A)*v_j+1
    #         err1 = (1/gamma)*c*abs(invH(j,:)*expHm(:,1))
#            err = (beta / gamma) * H[j+1,j] * np.abs(invH[j,:]*expHm[:,0]) # a less accurate err estimate without involving Ce inverse
#            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, invH, expHm)
            vm1 = A.dot(V[:n0, j+1])
            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1_norm, invH, expHm)
            if krylovcheck or (j == n0): # error check passed
    #             fprintf('      Krylov_SAI converges in %d steps with error %.5g. Time consumed %.5gs\n',j,err,tKrylov) 
                converged = True            
                break
    Vj = V[:, :j + 1]
    if not converged:
#        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err)) 
        y = v
        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err))
    else:
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
    tKrylov = timeit.time() - tKrylov
    
    return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj, vm1)

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
            
            vm1 = np.linalg.norm(A.dot(V[:n0, j+1]))
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1, invH, phiHm)
            if krylovcheck: # error check passed
    #             fprintf('      Krylov_SAI converges in %d steps with error %.5g. Time consumed %.5gs\n',j,err,tKrylov) 
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

def exp_error_check(x, beta, gamma, hj, vm1, invH, expHm):
    
    krylovTol = np.linalg.norm(x) * options.krylovTolr + options.krylovTola
    
    err = (beta) * hj * vm1 * np.abs(invH[-1].dot(expHm[:,0])) 
#    err1 = (beta / gamma) * hj * np.abs(invH[-1].dot(expHm[:,0])) 
    if err < krylovTol:
        krylovcheck = True
    else:
        krylovcheck = False
    
    return krylovcheck, err

def phim(A, v, dt, order):    
    
    n = A.shape[0]
    if order == 0:
        x = sp.linalg.expm(A * dt) * v
    elif order == 1:
        x = (sp.linalg.expm(A * dt) - np.eye(n)) * (np.linalg.inv(A) * v)
    elif order == 2:
        x = (A * sp.linalg.expm(A * dt) - sp.linalg.expm(A * dt) - sp.eye(n)) * (np.linalg.inv(A) * (np.linalg.inv(A) * v))
    else:
        raise ValueError('undefined order of phi function')
        
    return x    

def expm_Arnoldi(A, C, W, v, dt, gamma, m, kvec=None, linsolver=None):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    itvl = 1
    tKrylov = timeit.time()
    mmax = 1 * m
    n = len(v)
    n0 = C.shape[0] 
    order = np.round(n - n0)
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    converged = False
    if kvec is not None:
        v[:n0][~kvec] = 0
    beta = np.linalg.norm(v)
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, converged, j = 0, True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)
    V[:, 0] = v.flatten() / beta
    if order == 0:
        J = []
    elif order == 1:
        J = 1
    elif order == 2:
        J = np.array([[1, gamma],[0, 1]])
    elif order == 3:
    	J = np.array([[1,gamma,gamma**2], [0, 1, gamma], [0, 0, 1]])
    else:
        raise ValueError('order higher than 3')
    # Ce = blkdiag(C,eye(order))
#    yold = v[:n0]
    for j in range(mmax):
        z1 = V[:n0, j] 
        z2 = V[n0:, j]
        if order > 0:
            w2 = np.multiply(J, z2)
            v1 = (z1 + W.dot(w2)).reshape(-1, 1)
            if linsolver['name'] == 'pardiso':                    
                w1 = spsolve(A, v1, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
            elif linsolver['name'] == 'splu':
                lu = sp.sparse.linalg.splu(A)
                w1 = lu.solve(v1)  
            elif linsolver['name'] == 'GMRES':
                dx, info, niter = utilities.GMRES_wrapper(A, v1, v1, linsolver['tol'], linsolver['maxiter'])
                dx = dx.reshape((-1, 1))
            else:
                pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
            if kvec is not None:
                    w1[~kvec] = 0
            w = np.vstack((w1, w2))
        else:
            if linsolver is not None:  
                w = spsolve(A, z1, factorize=True, squeeze=True, solver=linsolver).reshape((-1, 1))
                if kvec is not None:
                    w1[~kvec] = 0
            else:
                pass
#                w = Q*(U\(L\(P*z1)))
    #         w = Q*(U\(L\(P*z1)))
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
            expHm = sp.linalg.expm(Hjj)
            vm1 = A.dot(V[:n0, j+1])
            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1_norm, invH, expHm)
            converged = True
            break 
        # invH = inv(H(1:j,1:j))
        V[:,j+1] = w[:, 0] / hj
        if j >= 0 and np.mod(j,itvl) == 0:
            expHm = sp.linalg.expm(Hjj)
    #         c = norm(w-Ce\(Ge*(gamma*(w)))) # h(j+1,j)*(I-gamma A)*v_j+1
    #         err1 = (1/gamma)*c*abs(invH(j,:)*expHm(:,1))
#            err = (beta / gamma) * H[j+1,j] * np.abs(invH[j,:]*expHm[:,0]) # a less accurate err estimate without involving Ce inverse
#            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, invH, expHm)
            vm1 = A.dot(V[:n0, j+1])
            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(v[:n0], beta, gamma, hj, vm1_norm, invH, expHm)
            if krylovcheck or (j == n0-1): # error check passed
    #             fprintf('      Krylov_SAI converges in %d steps with error %.5g. Time consumed %.5gs\n',j,err,tKrylov) 
                converged = True            
                break
    Vj = V[:, :j + 1]
    if not converged:
#        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err)) 
        y = v
        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, err))
    else:
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
    tKrylov = timeit.time() - tKrylov
    
    return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj, vm1)


def ei_mdn_solver1(x, mna, D, J0, circ, T, dt, MAXIT, nv, locked_nodes, time=None,
               print_steps=False, vector_norm=lambda v: max(abs(v)),
               linsolver=None, debug=True, verbose =3):
    
    # OLD COMMENT: FIXME REWRITE: solve through newton
    # problem is F(x)= mna*x +H(x) = 0
    # H(x) = N + T(x)
    # lets say: J = dF/dx = mna + dT(x)/dx
    # J*dx = -1*(mna*x+N+T(x))
    # dT/dx e' lo jacobiano -> g_eq (o gm)
    # print_steps = False
    # locked_nodes = get_locked_nodes(element_list)
#    mna = mna + sp.sparse.csc_matrix(sp.eye(mna.shape[0]) * 1e-4)
    A = sp.sparse.csc_matrix(D + mna)
    mna_size = A.shape[0]
#    Deye = sp.sparse.spdiags(np.ones(mna_size) * (1 / 1), [0], mna_size, mna_size, format='csr')
#    D = D+Deye
#    kvec = None
    
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
    
#    kvec = None
    if kvec is not None:
        idx1 = np.nonzero(kvec)[0]
        idx2 = np.nonzero(~kvec)[0]
        D11 = D[idx1, :][:, idx1]
        mna11 = mna[idx1, :][:, idx1]
        mna12 = mna[idx1, :][:, idx2]
        mna21 = mna[idx2, :][:, idx1]
        mna22 = mna[idx2, :][:, idx2]
        
        T1 = T[idx1]
        T2 = T[idx2]
        x1 = x[idx1]
        x2 = x[idx2]
        A11 = mna11 + D11
        mna_size1 = A11.shape[0]
    converged = False
    iteration = 0
    iteration1 = 0
    conv_history = []
    niter = 0
    tot_gmres = 0
    dx = np.zeros((mna_size, 1))  
    dx1 = dx[idx1]    
#    x[~kvec] = 0
    xtold = x.copy()
    xtold1 = xtold[idx1]
    xtold2 = xtold[idx2]
    _, Txtold = dc_analysis.generate_J_and_Tx(circ, xtold, time, nojac=False)
    Txtold = Txtold * 0
    Txtold1 = Txtold[idx1]
    Txtold2 = Txtold[idx2]
#    T[~kvec] = 0
    MAXIT = 50
    while iteration < MAXIT:  # newton iteration counter
        iteration += 1
        residual = np.zeros((mna_size, 1))
        xold = x.copy()
        tick.step()
        J, Tx = dc_analysis.generate_J_and_Tx(circ, xold, time, nojac=False)
        J = J * 0
        Tx = Tx * 0
        J11 = J[idx1, :][:, idx1]
        J011 = J0[idx1, :][:, idx1]
        J0x = J0.dot(0.5 * (xold + xtold))
        J0x1 = J0x[idx1]
#        J22 = J[idx2, :][:, idx2]
        Tx1 = Tx[idx1]
#        Tx2 = Tx[idx2]
        xold1 = xold[idx1]
        xold2 = xold[idx2]
        
        w_rhs = -(0.5 * (Tx1 + Txtold1) - J0x1 + T1 + mna12.dot(xold2))
        v_rhs = np.vstack((xtold1, [1]))
        xold1_, err1_krylov, krylovcheck, m1_krylov, solvar = expm_ArnoldiSAI(
                A11, D11, w_rhs, v_rhs, dt, gamma, m_max, kvec=None, linsolver=linsolver)[0:5] 
        D11t = D11.todense() * gamma
        A = -np.linalg.inv(D11t) * mna11
        s0, X0 = np.linalg.eig(A * dt)
        Hjj = solvar[2]
        s, X = np.linalg.eig(Hjj)
        expv = sp.linalg.expm(A * dt).dot(xtold1)
        phiA_Cinv = (sp.linalg.expm(A * dt) - sp.eye(mna_size1)).dot(np.linalg.inv(A).dot(np.linalg.inv(D11t)))
        phiv = phiA_Cinv.dot(w_rhs)
        xold10 = expv + phiv
        rhs = xold1 - xold10
#        J1 = sp.eye(mna_size1) + 0.5 * phiA_Cinv.dot(J11.todense()) + 0.5 * J0.dot(xold)
#        J1inv = np.linalg.inv(J1)
#        dx0 = J1inv.dot(-rhs)
        if not krylovcheck:
            raise ValueError('expm_SAI does not converge')

        residual1 = xold1 - xold1_[:mna_size1] 
        residual[idx1] = residual1
        Jv1 = sp.sparse.linalg.LinearOperator((mna_size1, mna_size1), 
                                              matvec=lambda v: jacobvec(A11, D11, 0.5 * (J11 - J011), mna11, v, dt, kvec=None, linsolver=linsolver)[0])
        dx1, info, niter = utilities.GMRES_wrapper(Jv1, -residual1, dx1, options.lin_GMRES_tol, options.lin_GMRES_maxiter)
        if info == 0:
            print('GMRES converged to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
        else:
            print('GMRES doesn not converge to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
        tot_gmres += niter
        dx1 = dx1.reshape((-1, 1))
        dx[idx1] = dx1
        dampfactor = dc_analysis.get_td(dx, locked_nodes, n=iteration)
#        dampfactor = 0.5
        x1 = xold1 + dampfactor * dx1        
        x[idx1] = x1
        
        if kvec is not None:
            b2 = mna21.dot(x1) + T2 + 0.5 * Txtold2
            while iteration1 < MAXIT: 
                iteration1 += 1
                Jnew, Txnew = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
                Jnew22 = Jnew[idx2, :][:, idx2]
                Txnew2 = Txnew[idx2]
    #                J2 = mna22 + Jnew22
    #                residual2 = mna22.dot(x2) + Txnew2 + b2
                J2 = mna22 + 0.5 * Jnew22
                residual2 = mna22.dot(x2) + 0.5 * Txnew2 + b2
                if linsolver['name'] == 'pardiso':
                    dx2 = spsolve(J2, -residual2, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
                elif linsolver['name'] == 'splu':
                    lu = sp.sparse.linalg.splu(J2)
                    dx2 = lu.solve(-residual2)
                else:
                    raise ValueError('undefined linear solver')
                x2 = x[idx2] + dx2
                x[idx2] = x2 
                dx_norm = np.linalg.norm(dx2)
                rhs_norm = np.linalg.norm(residual2)
                conv2, conv_data2 = utilities.custom_convergence_check(x2, dx2, residual2, 
                                                 options.ver, options.vea, options.ver, debug=False)                
                if conv2:
                    break
        if iteration1 >= MAXIT:
            print('Newton for x2 does not converge in {0} iters with dx={1}, rhs={2}'.format(iteration1, dx_norm, rhs_norm))
        residual[idx2] = residual2
        
        dx[idx2] = dx2
        print([np.linalg.norm(residual), np.linalg.norm(dx)])
        conv, conv_data = utilities.convergence_check(x, dx, residual, nv - 1, ni)
        conv_history.append(conv_data)
#        print('Nonlinear iter: {0} Convergence data: {1}'.format(iteration, conv_data))
        if not nonlinear_circuit:
            converged = True
            break
        elif conv:
            converged = True
            break
        # if vector_norm(dx) == np.nan: #Overflow
        #   raise OverflowError
    tick.hide(print_steps)
    if not converged:
        # re-run the convergence check, only this time get the results
        # by node, so we can show to the users which nodes are misbehaving.
#        converged, convergence_by_node = convergence_check(
#            x, dx, residual, nv - 1, ni, debug=True)
        print('Non-convergence data: {0}'.format(conv_data + [dampfactor]))
#        convergence_by_node = []
    else:
        conv_data = conv_data
    return (x, residual, converged, iteration, conv_history, tot_gmres) 