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

def ei_solve(A, mna, D, Ndc, circ, Gmin=None, x0=None, lu=None, time=None, tstep=None, 
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
    if linsolver is None:
        linsolver ='splu'
        
    mna_size = mna.shape[0]
    nv = circ.get_nodes_number() - 1
    tot_iterations = 0
    tot_gmres = 0
    tot_sai = 0

    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    Tt = np.zeros((mna_size, 1))
    if not skip_Tt:
        Tt = dc_analysis.generate_Tt(circ, time, mna_size) 
        Ttold = dc_analysis.generate_Tt(circ, time - tstep, mna_size) 
#        Ttold = Tt
        dTt = (Tt - Ttold) / tstep
    # update N to include the time variable sources
        if expord == 1:
            Ndc = Ndc + Tt
        elif expord == 2:
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
#        A = D + mna # gamma is absorbed into D
        W = Ndc
    elif circ.isnonlinear and (not options.ei_newton):
        Jold, Txold = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
        Dold = Txold - Jold.dot(x)
        #    A = sp.sparse.csc_matrix(D + gamma* (mna + Jold))
#        A = sp.sparse.csr_matrix(D + (mna + Jold)) # gamma is absorbed into D
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
            try:
                (x1, error, converged, n_iter, conv_history, n_gmres, n_sai) = ei_mdn_solver(x, A, mna_to_pass, D, circ, N_to_pass, dTt_to_pass,
                                                                                    tstep, MAXIT, lu, locked_nodes, 
                                                                                    time=time, print_steps=False, 
                                                                                    vector_norm=lambda v: max(abs(v)),
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

    if options.ei_newton:
        return x1, error, converged, tot_iterations, conv_history, tot_gmres, tot_sai
    else:
        return x1, error, converged, m1, solvar, tot_gmres  

def ei_mdn_solver(x, A, mna, D, circ, T, dT, dt, MAXIT, lu, locked_nodes, time=None,
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
#        mna12 = mna[idx1, :][:, idx2]
        mna21 = mna[idx2, :][:, idx1]
        mna22 = mna[idx2, :][:, idx2].tocsc()
        mna2 = mna[idx2]
        T2 = T[idx2] + dT[idx2] * dt

    converged = False
    iteration = 0
    tot_SAI = 0
    alpha = 1
    
    def jacobvec(A, D, J11, J21, v, dt, kvec=None, linsolver=None):
        nonlocal tot_SAI, alpha
        nonlocal mna2, J2, mna22, J22
        mna_size = A.shape[0]       
        v = v.reshape(-1, 1)      
        if (kvec is not None) and (J21 is not None):
            w1 = sp.zeros((mna_size, 1)) 
            w1[idx1] = J11.dot(v)
            w1[idx2] = J21.dot(v)
        else:
            w1 = J11.dot(v)   
        v1 = np.vstack((sp.zeros((mna_size, 1)), [1]))
        x1, error, solved, m1, solvar = expm_ArnoldiSAI(A, D, w1, v1, dt, gamma, m_max, options.lin_GMRES_atol, kvec, linsolver=linsolver)
        tot_SAI += m1
#        print(m1)
        if not solved:
            raise ValueError('expm_SAI in jacobvec does not converge')
            
        if (kvec is not None) and (J21 is not None):      
            y = alpha * x1[idx1] + v
        else:
            y = sp.zeros((mna_size, 1))
#            y[idx1] = alpha * x1[idx1] + v[idx1]
            y[idx1] = alpha * x1[idx1] + D.dot(v)[idx1]
#            y[idx2] = v[idx2]
#            y[idx2] = (mna2 + J2).dot(v)
            y[idx2] = (mna22 + J22).dot(v[idx2])
        return y
    
    conv_history = []
    niter = 0
    tot_gmres = 0
    dx = np.zeros((mna_size, 1))
    residual = np.zeros(dx.shape)
    xtold = x.copy()
    Jtold, Txtold = dc_analysis.generate_J_and_Tx(circ, xtold, time - dt, nojac=False)
#    Jtold22 = Jtold[idx2, :][:, idx2]
#    Txtold = Txtold * 0
    v_rhs = xtold.copy()
    if expord == 1:
        v_rhs = np.vstack((v_rhs, [1]))
    elif expord == 2:
        v_rhs = np.vstack((v_rhs, [[0],[1]]))
#    rhs2 = mna2.dot(xtold) + Txtold[idx2] + T2
#    dx2 = spsolve(mna22 + Jtold22, -rhs2, factorize=True, squeeze=False, 
#                  solver=linsolver['param1'])
#    dx1 = np.zeros((len(idx1), 1))
#    dx2 = np.zeros((len(idx2), 1)) 
#    x[idx2] += dx2
    MAXIT = 20
#    x = spio.loadmat('xref.mat')['xref']
    while iteration < MAXIT:  # newton iteration counter
        tnr = timeit.time()
        iteration += 1
        xold = x.copy()
        tick.step()
        J, Tx = dc_analysis.generate_J_and_Tx(circ, xold, time, nojac=False)
#        J = sp.sparse.csr_matrix(sp.zeros((mna_size, mna_size)))
#        Tx = np.zeros((mna_size, 1))
        if kvec is not None:
            J11 = J[idx1, :][:, idx1]
            J12 = J[idx1, :][:, idx2]
            J21 = J[idx2, :][:, idx1]
            J22 = J[idx2, :][:, idx2]
            J2 = J[idx2]
        else:
            J11 = J
            J21 = None
        Txnew = Tx * alpha + Txtold * (1 - alpha)     
        dTx = (Tx - Txtold) / dt

        
        w1 = sp.zeros((mna_size, 1)) 
#        w1[idx1] = J12.dot(dx2)
#        w1[idx2] = J22.dot(dx2)
#        if kvec is not None:
#            Txs = sp.zeros(Tx.shape)
#            Txs[idx1] = Tx[idx1] - mnainv.dot(Tx[idx2])
#        else:
#            Txs = Tx

        if expord == 1:
            W = -(Txnew + T) 
        elif expord == 2:
            W = np.hstack((-(dT), -(Tx + T)))
#        J2 = J[idx2, :][:, idx2] + mna22
#        t1 = timeit.time()
#        x1 = spsolve(A, xtold, factorize=True, squeeze=False, solver=linsolver['param'])
#        t1 = timeit.time() - t1
#        t2 = timeit.time()
#        x2 = spsolve(A, 2*xtold, factorize=True, squeeze=False, solver=linsolver['param'])
#        t2 = timeit.time() - t2
#        t3 = timeit.time()
#        x3 = spsolve(J2, xtold[idx2], factorize=True, squeeze=False, solver=linsolver['param1'])
#        t3 = timeit.time() - t3
#        print([x1, x2, x3])
        
        texp1 = timeit.time()
        xpre, residual1, krylovcheck, m1_SAI, solvar = expm_ArnoldiSAI(
                A, D, W, v_rhs, dt, gamma, m_max, options.ver/100, kvec, linsolver=linsolver)[0:5]
        tot_SAI += m1_SAI
        texp1 = timeit.time() - texp1
        
        if iteration >= 100:
            Cs = D[idx1, :][:, idx1].toarray()
            Cs = Cs * (gamma / dt)
            Csinv = np.linalg.inv(Cs)
            G11 = mna[idx1, :][:, idx1].toarray()
            G12 = mna[idx1, :][:, idx2].toarray()
            G21 = mna21.toarray()
            G22 = mna22.toarray()
            G22inv = np.linalg.inv(G22)
            del G22
            Gs = G11 - G12.dot(G22inv.dot(G21))
            del  G11, G21
            As = -Csinv.dot(Gs)
            del Gs
            J2 = np.array([[0, 1], [0, 0]])
            W1 = W[idx1, :]
            W2 = W[idx2, :]
            Ws = W1 - G12.dot(G22inv.dot(W2))
            expAs = sp.linalg.expm(As)
#            Ast = np.block([[As, Csinv.dot(Ws)], [np.zeros((2, len(idx1))), J2 * dt]])
#            eAst = sp.linalg.expm(Ast)
#            vt = np.block([[xtold[idx1]], [np.array([[0], [1]])]])
#            y = eAst.dot(vt)
#        w = np.linalg.inv(np.eye(5,5) - gamma * Ast).dot(vt)
        
        
        if not krylovcheck:
            raise ValueError('expm_SAI does not converge')

        rhs = xold - xpre[:mna_size] 
        rhs1 = rhs[idx1]
        dx1p = -rhs1
        residual[idx1] = residual1[idx1]
    
#        rhs2 = mna2.dot(xold) + Tx[idx2] + T2
#        Jnew, Txnew = dc_analysis.generate_J_and_Tx(circ, xold1[:mna_size], time, nojac=False)
#        Jnew21 = Jnew[idx2, :][:, idx1]
        rhs2 = mna2.dot(x) + Tx[idx2] + (mna21 + J21).dot(dx1p) + T2
#        rhs2 = mna2.dot(x) + Txnew[idx2] + (mna21 + Jnew21).dot(dx1p) + T2
#        rhs2 = mna2.dot(x) + Txnew[idx2] + T2 
#        dx2 = spsolve(mna22 + J22, -rhs2, factorize=True, squeeze=False, 
#                      solver=linsolver['param1'])
        residual[idx2] = rhs2
#        dx[idx2] = dx2
        
        rhs = np.zeros((mna_size, 1))
        rhs[idx1] = rhs1
        rhs[idx2] = rhs2
#        rhs[idx2] = -dx2
        residual = rhs
        
        
        x1_size = len(idx1)
#        v1 = J1.dot(rhs1)
#        v01 = np.zeros((mna_size, 1))
#        v01[idx1] = J11.dot(rhs1)
#        v01[idx2] = J21.dot(rhs1)
#        Jv = sp.sparse.linalg.LinearOperator(
#                (x1_size, x1_size), matvec=lambda v: jacobvec(A, D, J11, J21, v, dt, kvec, linsolver=linsolver))
        Jv = sp.sparse.linalg.LinearOperator(
                (mna_size, mna_size), matvec=lambda v: jacobvec(A, D, J, None, v, dt, kvec, linsolver=linsolver))
#        Jv0 = sp.sparse.linalg.LinearOperator((x1_size, x1_size), matvec=lambda v: jacobvec0(A, D, J11, J21, mnainv, v, dt, kvec, linsolver=linsolver))
        
#        dx0 = dx[idx1]
#        dx1, info, niter = utilities.GMRES_wrapper(Jv, -residual1, dx0, options.lin_GMRES_tol, options.lin_GMRES_maxiter)
        res_Krylov = []
        tgmres = timeit.time()
        if np.max(abs(rhs1)) == 0:
            pass
        GMREStol = np.max((options.lin_GMRES_rtol, 
                           options.lin_GMRES_atol / np.max(abs(rhs1))))
        GMRESmaxit = np.min((options.lin_GMRES_maxiter, x1_size))
#        dx0 = sp.zeros((x1_size, 1))
#        (dx1, info) = pyamg.krylov.gmres(
#                Jv, -rhs1, x0=dx0, tol=GMREStol, maxiter=GMRESmaxit, 
#                residuals=res_Krylov, orthog='mgs')
#        dx1 = dx1.reshape((-1, 1))
        dx0 = sp.zeros((mna_size, 1))
        (dx, info) = pyamg.krylov.gmres(
                Jv, -rhs, x0=dx0, tol=GMREStol, maxiter=GMRESmaxit, 
                residuals=res_Krylov, orthog='mgs')
        dx = dx.reshape((-1, 1))
        niter = len(res_Krylov) - 1
        tgmres = timeit.time() - tgmres
        if info >= 0 and info < GMRESmaxit:
            print('GMRES converged to {0} in {1} iterations'.format(GMREStol, niter))
        else:
            print('GMRES doesn not converge to {0} in {1} iterations'.format(GMREStol, niter))
        tot_gmres += niter
        
        dx1 = dx[idx1]
#        dx2p = dx[idx2]
        rhs21 = mna2.dot(x) + Tx[idx2] + (mna21 + J21).dot(dx1) + T2
        dx2 = spsolve((J22 + mna22), -rhs21, factorize=True, squeeze=False, solver=linsolver['param1'])
        dx[idx2] = dx2
#        dampfactor = dc_analysis.get_td(dx, locked_nodes, n=iteration)
        dampfactor = 0.2
        x += dampfactor * dx
#        x[idx1] = x[idx1] + dampfactor * dx1
#        x[idx2] = x[idx2] + dampfactor * dx2
#        tmp = 1
#        if kvec is not None:
#            if options.ei_use_jac22:
##                Jnew, Txnew = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=False)
##                Jnew22 = Jnew[idx2, :][:, idx2]
###                Jnew22 = J[idx2, :][:, idx2]
##                Txnew2 = Txnew[idx2]
###                Txnew2 = Tx[idx2]
##                J2 = mna22 + Jnew22
##                residual2 = mna2.dot(x) + Txnew2 + T2
#                
#                J2 = mna22 + J22
#                residual2 = mna2.dot(x) + Tx[idx2] + J21.dot(dx1) + T2
#            else:
#                _, Txnew = dc_analysis.generate_J_and_Tx(circ, x, time, nojac=True)
#                J2 = mna22
#                Txnew2 = Txnew[idx2]
#                residual2 = mna21.dot(x[idx1]) + Txnew2 + T2
#            if linsolver['name'] == 'pardiso':
#                dx2 = spsolve(J2, -residual2, factorize=True, squeeze=False, solver=linsolver['param1'])
##                if options.ei_use_jac22:
##                    lu = sp.sparse.linalg.splu(J2)
##                dx2 = lu.solve(-residual2)
#            elif linsolver['name'] == 'splu':
#                lu = sp.sparse.linalg.splu(J2)
#                dx2 = lu.solve(-residual2)
#            else:
#                raise ValueError('undefined linear solver')
#            x[idx2] += dx2
#            residual[idx2] = residual2
#            dx[idx2] = dx2
             
#        print([np.linalg.norm(residual), np.linalg.norm(dx)])
        conv, conv_data = utilities.convergence_check(x, dx, residual, nv - 1, ni)
        conv_history.append(conv_data)
        tnr = timeit.time() - tnr
        print('Newton iter: {0} Convergence data: {1}'.format(iteration, conv_data))
#        import sys
#        if iteration > 4:
#            sys.exit(3)
        if not nonlinear_circuit:
            converged = True
            break
        elif conv:
            converged = True
            break
        if vector_norm(dx) == np.nan: #Overflow
           raise OverflowError
    tick.hide(print_steps)
    if not converged:
        print('Non-convergence data: {0}'.format(conv_data + [dampfactor]))
    else:
        conv_data = conv_data
    return (x, residual, converged, iteration, conv_history, tot_gmres, tot_SAI)        

def expm_ArnoldiSAI(A, C, W, v, dt, gamma, m, tol, kvec=None, linsolver=None):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    tKrylov = timeit.time()
    mmax = 1 * m
    n = len(v)
    n0 = C.shape[0] 
    order = np.round(n - n0)
    v0 = v.copy()
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    converged = False
    if kvec is not None:
        v0[:n0][~kvec] = 0
    beta = np.linalg.norm(v0)
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, converged, j = np.zeros((n0,1)), True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)
    V[:, 0] = v0.flatten() / beta
    if order == 0:
        IJinv = np.array([0])
    elif order == 1:
        IJinv = np.array([[1]])
    elif order == 2:
        IJinv = np.array([[1, gamma],[0, 1]])
    elif order == 3:
    	IJinv = np.array([[1,gamma,gamma**2], [0, 1, gamma], [0, 0, 1]])
    else:
        raise ValueError('order higher than 3')
    for j in range(mmax):
        z1 = C.dot(V[:n0, j])[:, None] 
#        if kvec is not None:
#            z1[~kvec] = 0
        z2 = IJinv.dot(V[n0:, j][:, None])
        if order > 0:
#            w2 = IJinv.dot(z2)
            wz2 = W.dot(z2)
            v1 = (z1 + wz2).reshape(-1, 1)
            if linsolver['name'] == 'pardiso':                    
                w1 = spsolve(A, v1, factorize=True, squeeze=False, solver=linsolver['param'])
            elif linsolver['name'] == 'splu':
                lu = sp.sparse.linalg.splu(A)
                w1 = lu.solve(v1)  
            else:
                pass #w1 = Q*(U\(L\(P*(z1+W*(gamma*w2)))))
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
        for l in range(2):   # double orthogonalization 
            for i in range(j + 1):
                h = np.dot(w.conj().T, V[:, i])
                H[i,j] += h
                w = w - h * V[:,i][:, None]
        hj = np.linalg.norm(w)    
        H[j+1, j] = hj
        invH = np.linalg.inv(H[:j + 1, :j + 1])
        Hjj = (np.eye(j + 1) - invH) * (dt / gamma)
        invHj = invH[j, :]      
        
        maxeig = np.max(np.linalg.eigvals(Hjj).real)
#        if np.max(v[:n0]) == 0:
#            se = np.linalg.eigvals(Hjj).reshape(-1, 1)
#            seall.append(np.sort(se))
#            se1 = np.linalg.eigvals(H[:j + 1, :j + 1]).reshape(-1, 1)
            
#            print(se)
        # Happy breakdown
        if (hj < 1e-12) and (maxeig < 600):           
#            vm1_norm = np.linalg.norm(vm1)
            expHm = sp.linalg.expm(Hjj)
            converged = True
            err = np.zeros((n0,1))
            break 
        V[:,j+1] = w[:, 0] / hj
        if (maxeig < 600) and (np.isfinite(invH).all()):  
            expHm = sp.linalg.expm(Hjj) 
            vm1 = A.dot(V[:n0, j+1])
#            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(tol, v0[:n0], beta, gamma, hj, vm1, invH, expHm)
            if krylovcheck or (j == n0): # error check passed
                converged = True            
                break
        else:
#            [print(se) for se in seall]
#            print('something wrong. better stop')
            pass        
            
    Vj = V[:, :j + 1]  
#    if options.ei_reg:
#        if (np.max(v[:n0]) == 0) and (j > 2): 
##    if (np.max(v[:n0]) == 0) and (j > 2):     
##            np.savetxt('eig.txt', seall)
#            with open('eig_reg.txt', 'w') as file:
#                [np.savetxt(file, se, fmt='%.4g') for se in seall]
##                file.writelines(str(se) + '\n' for se in seall)
#            tmp = 1 
#    else:
#        if (np.max(v[:n0]) == 0) and (maxeig > 600): 
#    #    if (np.max(v[:n0]) == 0) and (j > 2):     
#    #            np.savetxt('eig.txt', seall)
#            with open('eig.txt', 'w') as file:
#                [np.savetxt(file, se, fmt='%.4g') for se in seall]
##                file.writelines(str(se) + '\n' for se in seall)
#            tmp = 1    
    if not converged:
        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err))) 
        y = v
#        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err)))
    else:
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
        if np.isnan(y).any():
            printing.print_warning('Solution contains nan')

    tKrylov = timeit.time() - tKrylov
    
    return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)



def exp_error_check(tol, x, beta, gamma, hj, vm1, invH, expHm):
    
#    krylovTol = np.linalg.norm(x) * options.krylovTolr + options.krylovTola
#    krylovTol = options.krylovTolr
    krylovTol = tol
    
#    err = (beta * hj * np.abs(invH[-1].dot(expHm[:,0]))) * vm1 
    err1 = beta * hj * np.abs(invH[-1].dot(expHm[:,0]))
    err = err1 * vm1
    err_norm = np.max(np.abs(err))
    if err_norm < krylovTol:
        krylovcheck = True
    else:
        krylovcheck = False
    
    return krylovcheck, err.reshape(-1, 1)

   

def expm_ArnoldiSAI0(A, C, W, v, dt, gamma, m, tol, kvec=None, linsolver=None):
    # y = expm_ArnoldiSAI(A,v,t,toler,m)
    # computes $y = \exp(-t A) v$
    # input: A (n x n)-matrix, v n-vector, t>0 time interval,
    # toler>0 tolerance, m maximal Krylov dimension
    tKrylov = timeit.time()
    mmax = 1 * m
    n = len(v)
    n0 = C.shape[0] 
    order = np.round(n - n0)
    v0 = v.copy()
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))
    converged = False
    if kvec is not None:
        v0[:n0][~kvec] = 0
    beta = np.linalg.norm(v0)
    if beta < 1e-15:
        y = np.zeros((n, 1))
        err, converged, j = np.zeros((n0,1)), True, 0
        Vj, Hjj, hj, invHj = None, None, None, None
        return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)
    V[:, 0] = v0.flatten() / beta
    if order == 0:
        IJinv = 0
    elif order == 1:
        IJinv = np.array([[1]])
    elif order == 2:
        IJinv = np.array([[1, gamma],[0, 1]])
    elif order == 3:
    	IJinv = np.array([[1,gamma,gamma**2], [0, 1, gamma], [0, 0, 1]])
    else:
        raise ValueError('order higher than 3')
    seall = []
    for j in range(mmax):
        z1 = C.dot(V[:n0, j])[:, None] 
        z2 = V[n0:, j][:, None]
        if order > 0:
            w2 = IJinv.dot(z2)
            v1 = (z1 + W.dot(w2)).reshape(-1, 1)
            if linsolver['name'] == 'pardiso':                    
                w1 = spsolve(A, v1, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
            elif linsolver['name'] == 'splu':
                lu = sp.sparse.linalg.splu(A)
                w1 = lu.solve(v1)  
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
        for l in range(2):   # double orthogonalization 
            for i in range(j + 1):
                h = np.dot(w.conj().T, V[:, i])
                H[i,j] += h
                w = w - h * V[:,i][:, None]
        hj = np.linalg.norm(w)    
        H[j+1, j] = hj
        invH = np.linalg.inv(H[:j + 1, :j + 1])
        Hjj = (np.eye(j + 1) - invH) * (dt / gamma)
        invHj = invH[j, :]      
        
        maxeig = np.max(np.linalg.eigvals(Hjj).real)
#        if np.max(v[:n0]) == 0:
#            se = np.linalg.eigvals(Hjj).reshape(-1, 1)
#            seall.append(np.sort(se))
#            se1 = np.linalg.eigvals(H[:j + 1, :j + 1]).reshape(-1, 1)
            
#            print(se)
        # Happy breakdown
        if (hj < 1e-12) and (maxeig < 600):           
#            vm1_norm = np.linalg.norm(vm1)
            expHm = sp.linalg.expm(Hjj)
            converged = True
            err = np.zeros((n0,1))
            break 
        V[:,j+1] = w[:, 0] / hj
        if (maxeig < 600) and (np.isfinite(invH).all()):  
            expHm = sp.linalg.expm(Hjj) 
            vm1 = A.dot(V[:n0, j+1])
#            vm1_norm = np.linalg.norm(vm1)
            krylovcheck, err = exp_error_check(tol, v0[:n0], beta, gamma, hj, vm1, invH, expHm)
            if krylovcheck or (j == n0): # error check passed
                converged = True            
                break
        else:
#            [print(se) for se in seall]
#            print('something wrong. better stop')
            break        
            
    Vj = V[:, :j + 1]  
#    if options.ei_reg:
#        if (np.max(v[:n0]) == 0) and (j > 2): 
##    if (np.max(v[:n0]) == 0) and (j > 2):     
##            np.savetxt('eig.txt', seall)
#            with open('eig_reg.txt', 'w') as file:
#                [np.savetxt(file, se, fmt='%.4g') for se in seall]
##                file.writelines(str(se) + '\n' for se in seall)
#            tmp = 1 
#    else:
#        if (np.max(v[:n0]) == 0) and (maxeig > 600): 
#    #    if (np.max(v[:n0]) == 0) and (j > 2):     
#    #            np.savetxt('eig.txt', seall)
#            with open('eig.txt', 'w') as file:
#                [np.savetxt(file, se, fmt='%.4g') for se in seall]
##                file.writelines(str(se) + '\n' for se in seall)
#            tmp = 1    
    if not converged:
        raise ValueError('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err))) 
        y = v
#        print('Krylov_SAI not converged at maximum m = %d with err = %g' % (j + 1, np.linalg.norm(err)))
    else:
        eHm = expHm[:, 0].reshape(-1, 1)
        y = Vj.dot(beta * eHm)
        if np.isnan(y).any():
            printing.print_warning('Solution contains nan')

    tKrylov = timeit.time() - tKrylov
    
    return y, err, converged, j + 1, (beta, Vj, Hjj, hj, invHj)

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