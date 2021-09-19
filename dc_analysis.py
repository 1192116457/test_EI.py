# -*- coding: iso-8859-1 -*-
# dc_analysis.py
# DC simulation methods
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

"""
This module provides the functions needed to perform OP and DC simulations.

The principal are:

* :func:`dc_analysis` - which performs a dc sweep,
* :func:`op_analysis` - which does an operation point analysis or

Notice that internally, :func:`dc_analysis` calls :func:`op_analysis`,
since a DC sweep is nothing but a series of OP analyses..

The actual circuit solution is done by :func:`mdn_solver`, that uses a
modified version of the Newton Rhapson method.

Module reference
################

"""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import sys
import re
import copy
import time as timeit

import numpy as np
import numpy.linalg
import scipy as sp
# import scipy.sparse
# import scipy.sparse.linalg
import scipy.io as spio
# import scipy.sparse.linalg.norm as spnorm
from joblib import Parallel, delayed
import pyamg

import devices
import diode
import memristor
import constants
import ticker
import options
import circuit
import printing
import utilities
import dc_guess
import results
import bsim3

from utilities import convergence_check

from pypardiso import PyPardisoSolver
from pypardiso import spsolve

# pardiso = PyPardisoSolver()
# pardiso.set_iparm(1, 1)
# pardiso.set_iparm(2, 0)
##for i in range(5):
##pardiso.set_iparm(3, 4)
# pardiso.set_iparm(10, 13)
# pardiso.set_iparm(11, 1)
# pardiso.set_iparm(13, 1)
# pardiso.set_iparm(21, 1)
# pardiso.set_iparm(25, 1)
# pardiso.set_iparm(34, 1)

specs = {'op': {
    'tokens': ({
                   'label': 'guess',
                   'pos': None,
                   'type': bool,
                   'needed': False,
                   'dest': 'guess',
                   'default': options.dc_use_guess
               },
               {
                   'label': 'ic_label',
                   'pos': None,
                   'type': str,
                   'needed': False,
                   'dest': 'x0',
                   'default': None
               }
    )
},
    'dc': {'tokens': ({
                          'label': 'source',
                          'pos': 0,
                          'type': str,
                          'needed': True,
                          'dest': 'source',
                          'default': None
                      },
                      {
                          'label': 'start',
                          'pos': 1,
                          'type': float,
                          'needed': True,
                          'dest': 'start',
                          'default': None
                      },
                      {
                          'label': 'stop',
                          'pos': 2,
                          'type': float,
                          'needed': True,
                          'dest': 'stop',
                          'default': None
                      },
                      {
                          'label': 'step',
                          'pos': 3,
                          'type': float,
                          'needed': True,
                          'dest': 'step',
                          'default': None
                      },
                      {
                          'label': 'type',
                          'pos': None,
                          'type': str,
                          'needed': False,
                          'dest': 'sweep_type',
                          'default': options.dc_lin_step
                      }
    )
    }
}


# import ctypes
# dll_file = "BSIM3_Dll_new.dll"
# bsim = ctypes.cdll.LoadLibrary(dll_file)

# class bsimMatrix(ctypes.Structure):
#     _fields_ = [ 
#         ("M",ctypes.c_int),
#         ("N",ctypes.c_int),
#         ("NNZ",ctypes.c_int),

#         ("Ir",ctypes.POINTER(ctypes.c_int)),
#         ("Jc",ctypes.POINTER(ctypes.c_int)),
#         ("Pr",ctypes.POINTER(ctypes.c_double)),
#     ]    


def dc_solve(mna, Ndc, circ, Ntran=None, Gmin=None, x0=None, Tx0=None, time=None, tstep=None,
             MAXIT=None, locked_nodes=None, skip_Tt=False, bsimOpt=None, linsolver=None, verbose=3, D=None):
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
    mna_size = mna.shape[0]
    nv = circ.get_nodes_number()
    tot_iterations = 0
    tot_gmres = 0

    if Gmin is None:
        Gmin = 0

    if Ntran is None:
        Ntran = 0

    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    Tt = np.zeros((mna_size, 1))
    if not skip_Tt:
        Tt = generate_Tt(circ, time, mna_size)
        Ttold = generate_Tt(circ, time - tstep, mna_size)
        if time > tstep:
            Tt = (Tt + Ttold * 0)
    # update N to include the time variable sources
    Ndc = (Ndc + Tt)

    # initial guess, if specified, otherwise it's zero
    if x0 is not None:
        if isinstance(x0, results.op_solution):
            x = x0.asarray()
        else:
            x = x0
    else:
        x = np.zeros((mna_size, 1))
        # has n-1 rows because of discard of ^^^

    converged = False
    standard_solving, gmin_stepping, source_stepping = get_solve_methods()
    standard_solving, gmin_stepping, source_stepping = set_next_solve_method(standard_solving, gmin_stepping,
                                                                             source_stepping, verbose)

    convergence_by_node = None
    printing.print_info_line(("Solving... ", 3), verbose, print_nl=False)

    while (not converged):
        if standard_solving["enabled"]:
            mna_to_pass = mna + Gmin
            N_to_pass = Ndc + Ntran * (Ntran is not None)
        elif gmin_stepping["enabled"]:
            # print "gmin index:", str(gmin_stepping["index"])+", gmin:", str(
            # 10**(gmin_stepping["factors"][gmin_stepping["index"]]))
            printing.print_info_line(
                ("Setting Gmin to: " + str(10 ** gmin_stepping["factors"][gmin_stepping["index"]]), 6), verbose)
            mna_to_pass = build_gmin_matrix(
                circ, 10 ** (gmin_stepping["factors"][gmin_stepping["index"]]), mna_size, verbose) + mna
            N_to_pass = Ndc + Ntran * (Ntran is not None)
        elif source_stepping["enabled"]:
            printing.print_info_line(
                ("Setting sources to " + str(
                    source_stepping["factors"][source_stepping["index"]] * 100) + "% of their actual value", 6),
                verbose)
            mna_to_pass = mna + Gmin
            N_to_pass = source_stepping["factors"][source_stepping["index"]] * Ndc + Ntran * (Ntran is not None)
        else:
            mna_to_pass = mna
            N_to_pass = Ndc + Ntran * (Ntran is not None)
        try:
            (x, Tx, error, converged, n_iter, convergence_by_node, n_gmres, pss_data) = mdn_solver(x, mna_to_pass, circ,
                                                                                                   T=N_to_pass, Tx=Tx0,
                                                                                                   nv=nv, print_steps=(
                            verbose > 0),
                                                                                                   locked_nodes=locked_nodes,
                                                                                                   time=time,
                                                                                                   tstep=tstep,
                                                                                                   MAXIT=MAXIT,
                                                                                                   bsimOpt=bsimOpt,
                                                                                                   linsolver=linsolver,
                                                                                                   debug=(verbose == 6),
                                                                                                   D=D)
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
            #                for ivalue in range(len(convergence_by_node)):
            #                    if not convergence_by_node[ivalue] and ivalue < nv - 1:
            #                        print("Convergence problem node %s" % (circ.int_node_to_ext(ivalue + 1),)) # gnd node not inclued (Alex)
            #                    elif not convergence_by_node[ivalue] and ivalue >= nv and ivalue < nv + circ.ni - 1:
            ##                        e = circ.find_vde(ivalue - nv + 1)
            ##                        print("Convergence problem current in %" % e.part_id)
            #                        print("Convergence problem in current")
            #                    elif not convergence_by_node[ivalue]:
            #                        print("Convergence problem in x-state")
            if n_iter == MAXIT - 1:
                printing.print_general_error(
                    "Error: MAXIT exceeded (" + str(MAXIT) + ")")
            if more_solve_methods_available(standard_solving, gmin_stepping, source_stepping):
                standard_solving, gmin_stepping, source_stepping = set_next_solve_method(
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
                # spio.savemat('x0_ram_gmin.mat', {'x': x})
                converged = False
            else:
                printing.print_info_line((" done.", 3), verbose)
                # spio.savemat('x0_ram.mat', {'x': x})
                # sys.exit(3)
    return (x, Tx, error, converged, tot_iterations, tot_gmres, pss_data)


def mdn_solver(x, mna, circ, T, Tx, MAXIT, nv, locked_nodes, time=None, tstep=None,
               print_steps=False, vector_norm=lambda v: max(abs(v)),
               bsimOpt=None, linsolver=None, debug=True, verbose=3, D=None):
    """
    Solves a problem like F(x) = 0 using the Newton Algorithm with a variable
    damping.

    Where:

    .. math::

        F(x) = mna*x + T + T(x)

    * :math:`mna` is the Modified Network Analysis matrix of the circuit
    * :math:`T(x)` is the contribute of nonlinear elements to KCL
    * :math:`T` contains the contributions of the independent sources, time
    * invariant and linear

    If :math:`x(0)` is the initial guess, every :math:`x(n+1)` is given by:

    .. math::
        x(n+1) = x(n) + td \\cdot dx

    Where :math:`td` is a damping coefficient to avoid overflow in non-linear
    components and excessive oscillation in the very first iteration. Afterwards
    :math:`td=1` To calculate :math:`td`, an array of locked nodes is needed.

    The convergence check is done this way:

    **Parameters:**

    x : ndarray
        The initial guess. If set to ``None``, it will be initialized to all
        zeros. Specifying a initial guess may improve the convergence time of
        the algorithm and determine which solution (if any) is found if there
        are more than one.
    mna : ndarray
        The Modified Network Analysis matrix of the circuit, reduced, see above.
    circ : circuit instance
        The circuit instance.
    T : ndarray,
        The :math:`T` vector described above.
    MAXIT : int
        The maximum iterations that the method may perform.
    nv : int
        Number of nodes in the circuit (counting the ref, 0)
    locked_nodes : list of tuples
        A list of ports driving non-linear elements, generated by
        ``circ.get_locked_nodes()``
    time : float or None, optional
        The value of time to be passed to non_linear _and_ time variant
        elements.
    print_steps : boolean, optional
        Show a progress indicator, very verbose. Defaults to ``False``.
    vector_norm : function, optional
        An R^N -> R^1 function returning the norm of a vector, for convergence
        checking. Defaults to the maximum norm, ie :math:`f(x) = max(|x|)`,
    debug : int, optional
        Debug flag that will result in an array being returned containing
        node-by-node convergence information.

    **Returns:**

    sol : ndarray
        The solution.
    err : ndarray
        The remaining error.
    converged : boolean
        A boolean that is set to ``True`` whenever the method exits because of a
        successful convergence check. ``False`` whenever convergence problems
        where found.
    N : int
        The number of NR iterations performed.
    convergence_by_node : list
        If ``debug`` was set to ``True``, this list has the same size of the MNA
        matrix and contains the information regarding which nodes fail to
        converge in the circuit. Ie. ``if convergence_by_node[j] == False``,
        node ``j`` has a convergence problem. This may significantly help
        debugging non-convergent circuits.

    """
    # OLD COMMENT: FIXME REWRITE: solve through newton
    # problem is F(x)= mna*x +H(x) = 0
    # H(x) = N + T(x)
    # lets say: J = dF/dx = mna + dT(x)/dx
    # J*dx = -1*(mna*x+N+T(x))
    # dT/dx e' lo jacobiano -> g_eq (o gm)
    # print_steps = False
    # locked_nodes = get_locked_nodes(element_list)
    mna_size = mna.shape[0]

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

    converged = False
    iteration = 0
    conv_history = [[1e5, 1e5]]
    niter = 0
    # Txtold = Tx.copy()
    # bsimOpt['first_time'] = 1
    dx = np.zeros(mna_size)
    dx_smallest = 10
    x_to_save = x
    #    xall = np.zeros((mna_size, 10))
    #    x = spio.loadmat('xref.mat')['xref']
    #    idx1 = spio.loadmat('idx.mat')['idx1']
    #    idx2 = spio.loadmat('idx.mat')['idx2']
    while iteration < MAXIT:  # newton iteration counter
        tnr = timeit.time()
        # xold = x.copy()
        iteration += 1
        tick.step()
        if nonlinear_circuit:
            # build dT(x)/dx (stored in J) and Tx(x)
            if options.useBsim3:
                bsimOpt['iter'] = iteration
                # bsimOpt['first_time'] = iteration
                Cnl, J, Tx, Tx_mexp, ECnl = generate_J_and_Tx_bsim(circ, x, tstep, bsimOpt)
                bsimOpt['first_time'] = 0
                if 'x_coeff' in bsimOpt:
                    M = sp.sparse.csc_matrix(mna + J + Cnl.multiply(bsimOpt['x_coeff']))
                    # residual = mna.dot(x) + T + Tx #+ Cnl.dot(bsimOpt['const'])
                    residual = Tx - T
                else:
                    M = sp.sparse.csc_matrix(mna + J)
                    # residual = mna.dot(x) + T + Tx
                    residual = Tx - T
            else:
                J, Tx = generate_J_and_Tx(circ, x, time, nojac=False)
                M = sp.sparse.csc_matrix(mna + J)
                residual = mna.dot(x) + T + Tx
        #            J = sp.zeros((mna_size, mna_size))
        #            Tx = np.zeros((mna_size, 1))

        else:
            M = sp.sparse.csc_matrix(mna)
            residual = mna.dot(x) + T
        #        sys.exit(1)

        if options.useBsim3:
            xnew = lin_solve(M, residual, linsolver, options.use_sparse)
            dx = xnew - x
            residual[:] = 0
        else:
            dx = lin_solve(M, -residual, linsolver, options.use_sparse)

        # ### test for pss (not working)         
        # lu = sp.sparse.linalg.splu(M)
        # D = D.tocsc()
        # tluc = timeit.time()
        # luc = sp.sparse.linalg.splu(D)
        # tluc = timeit.time() - tluc

        # t1 = timeit.time()
        # # dx1 = lin_solve(M, D.dot(x), linsolver, options.use_sparse)
        # dx1 = lu.solve(D.dot(x))
        # t1 = timeit.time() - t1

        # t2 = timeit.time()
        # Jv = sp.sparse.linalg.LinearOperator(
        #         (mna_size, mna_size), matvec=lambda v: luc.solve(M.dot(v)))
        # # dx2 = lin_solve(D, M.dot(residual), linsolver, options.use_sparse)
        # dx0 = np.zeros(mna_size)
        # GMREStol = 1e-6
        # GMRESmaxit = 50
        # res_Krylov = []
        # (dx2, info) = pyamg.krylov.gmres(
        #             Jv, x, x0=x, tol=GMREStol, maxiter=GMRESmaxit, 
        #             residuals=res_Krylov, orthog='mgs')
        # t2 = timeit.time() - t2
        # print([t1, t2])

        # dampfactor = get_td(dx, locked_nodes, n=iteration)
        dampfactor = 1.0
        x = x + dampfactor * dx
        conv, conv_data = convergence_check(x, dx, residual, nv - 1, ni)

        # idx = spio.loadmat('idx_io.mat')
        # idx1 = idx['idx1'][0]
        # idx2 = idx['idx2'][0]
        # print([max(abs(dx[idx1])), max(abs(dx[idx2])), max(abs(residual[idx1])), max(abs(residual[idx2]))])
        #     dx_smallest = conv_data[0]
        #     x_to_save = x - dampfactor * dx
        # if conv_data[0] > conv_history[iteration - 1][0]:
        #     # spio.savemat('x0_io.mat', {'x': x - dampfactor * dx})
        #     print('convergence problem')

        conv_history.append(conv_data)
        #        conv_data = [np.linalg.norm(dx[idx1]), np.linalg.norm(dx[idx2]),
        #                     np.linalg.norm(residual[idx1]), np.linalg.norm(residual[idx2])]

        tnr = timeit.time() - tnr
        # sys.exit(3)
        # print('Nonlinear iter: {0} Convergence data: {1}. Time used: {2}s'.format(iteration, conv_data, tnr))

        if not nonlinear_circuit:
            converged = True
            pss_data = []
            break
        elif np.all(conv):
            converged = True
            if options.useBsim3:
                if bsimOpt['mode'] == 'dc':
                    bsimOpt['iter'] = -1
                    Cnl, J, Tx, Tx_mexp, ECnl = generate_J_and_Tx_bsim(circ, x, tstep, bsimOpt)
                    M = sp.sparse.csc_matrix(mna + J)
                    # residual = mna.dot(x) + T + Tx
                    residual = Tx - T
                    x = lin_solve(M, residual, linsolver, options.use_sparse)
            else:
                if options.use_pss:
                    J, Tx = generate_J_and_Tx(circ, x, time, nojac=False)
                    # pss_data = {'lu': []}  
                    pss_data = {'lu': sp.sparse.linalg.splu(sp.sparse.csc_matrix(mna + J)),
                                'CG': mna + J, 'C': D}
                    tmp = 1
                else:
                    _, Tx = generate_J_and_Tx(circ, x, time, nojac=True)
                    pss_data = []
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

    return (x, Tx, residual, converged, iteration, conv_history, niter, pss_data)


def generate_J_and_Tx_bsim(circ, x, tstep, bsimOpt):
    mna_size = len(x)
    if mna_size == 0:
        raise ValueError('Empty x')
    Tx = np.zeros((mna_size, 1))
    # mna_size1 = circ.get_nodes_number() + circ.ni - 1

    voltage = x[:circ.nv - 1, 0]
    voltage = np.insert(voltage, 0, 0)
    voltage_ptr = (ctypes.c_double * len(voltage))(*voltage)
    # step_size = 1e-12
    tstep = 0.0 if tstep is None else tstep
    first_time = bsimOpt['first_time']
    iter_num = bsimOpt['iter']
    mode = ctypes.c_char_p(bsimOpt['mode'].encode('utf-8'))
    # read the voltage, M includes ground node
    M = len(voltage)

    # define the output of Matrix
    Tx = np.zeros((mna_size, 1))
    Tx = (ctypes.c_double * len(Tx))(*Tx)
    Tx_mexp = np.zeros((mna_size, 1))
    Tx_mexp = (ctypes.c_double * len(Tx_mexp))(*Tx_mexp)
    outFlag = bsimOpt['outFlag']
    matrix = bsimMatrix()
    Jsp = bsimMatrix()
    Cnlsp = bsimMatrix()
    ECsp = bsimMatrix()

    bsim.BSIM3Eval.argtypes = [ctypes.c_double, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.c_double, ctypes.c_char_p,
                               ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                               ctypes.POINTER(bsimMatrix), ctypes.POINTER(bsimMatrix),
                               ctypes.POINTER(bsimMatrix), ctypes.POINTER(bsimMatrix)]
    bsim.BSIM3Eval.restype = ctypes.c_double

    # t1 = timeit.time()
    bsim.BSIM3Eval(circ.bsim3list, M, voltage_ptr,
                   tstep, mode, first_time, iter_num,
                   Tx, Tx_mexp, outFlag,
                   ctypes.byref(matrix),
                   ctypes.byref(Jsp),
                   ctypes.byref(Cnlsp),
                   ctypes.byref(ECsp))
    J = utilities.bsim2cscMatrix(Jsp, (mna_size, mna_size))
    Tx1 = np.array(Tx[:]).reshape(-1, 1)
    Tx_mexp = np.array(Tx_mexp[:]).reshape(-1, 1)
    # t1 = timeit.time() - t1
    # print(t1)
    # Tx1 = Tx_mexp
    # F1 = J.dot(x)
    F1 = 0
    if first_time == 1:
        Tx = (Tx1 - F1)
    else:
        Tx = (Tx1 - F1)
    # Tx = utilities.bsim2cscMatrix(Txsp, (mna_size, 1))
    Cnl = ECnl = []
    if bsimOpt['mode'] == 'tran':
        Cnl = utilities.bsim2cscMatrix(Cnlsp, (mna_size, mna_size)) * 1
        if outFlag == 1:
            ECnl = utilities.bsim2cscMatrix(ECsp, (mna_size, mna_size))

    return Cnl, J, Tx, Tx_mexp, ECnl


def generate_J_and_Tx(circ, x, time, nojac=False):
    mna_size = len(x)
    if mna_size == 0:
        raise ValueError('Empty x')
    Tx = np.zeros((mna_size, 1))
    Tx0 = Tx.copy()
    mna_size1 = circ.get_nodes_number() + circ.ni - 1
    if options.use_sparse:
        row, col, val = [], [], []

        # result = Parallel(n_jobs=2)(delayed(update_J_and_Tx_sparse_parallel)(x, elem, time, mna_size1, nojac) for elem in circ)
        for elem in circ:
            if elem.is_nonlinear:
                # telem = timeit.time()
                _update_J_and_Tx_sparse(row, col, val, Tx, x, elem, time, mna_size1, nojac)
                # telem = timeit.time() - telem
                # tmp = 1
        if nojac:
            J = 0
        else:
            J = sp.sparse.csc_matrix((val, (row, col)), shape=(mna_size, mna_size))
    else:
        J = np.zeros((mna_size, mna_size))
        for elem in circ:
            if elem.is_nonlinear:
                _update_J_and_Tx(J, Tx, x, elem, time)

    return J, Tx


def update_J_and_Tx_sparse_parallel(x, elem, time, mna_size1=0, nojac=False):
    row, col, val = [], [], []
    iis1, i = [], []
    if isinstance(elem, memristor.memristor):
        v = 0.
        if elem.n1:
            v = v + x[elem.n1 - 1, 0]
        if elem.n2:
            v = v - x[elem.n2 - 1, 0]
        xm = x[mna_size1 + elem.mem_index, 0]
        vx = [v, xm]
        if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
            if not nojac:
                iis, gs = elem.gstamp(vx, time, mna_size1)
                row += list(iis[0])
                col += list(iis[1])
                val += list(gs.reshape(-1))
            #            J[iis] += gs.reshape(-1)
            iis1, i = elem.istamp(vx, time, mna_size1)
            # Tx[iis1] += i.reshape(-1)
        else:
            raise Exception("gstamp or istamp not defined for memristor")
    #            continue
    elif isinstance(elem, devices.BISource):
        v = 0.
        if elem.n1:
            v = v + x[elem.n1 - 1, 0]
        if elem.n2:
            v = v - x[elem.n2 - 1, 0]
        if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
            if not nojac:
                iis, gs = elem.gstamp(v, time)
                row += list(iis[0])
                col += list(iis[1])
                val += list(gs.reshape(-1))
            #            J[iis] += gs.reshape(-1)
            iis1, i = elem.istamp(v, time)
            # Tx[iis1] += i.reshape(-1)
        else:
            raise Exception("gstamp or istamp not defined for BISource")
    elif elem.is_nonlinear:
        out_ports = elem.get_output_ports()
        for index in range(len(out_ports)):
            n1, n2 = out_ports[index]
            n1m1, n2m1 = n1 - 1, n2 - 1
            dports = elem.get_drive_ports(index)
            v_dports = []
            for port in dports:
                v = 0.  # build v: remember we removed the 0 row and 0 col of mna -> -1
                if port[0]:
                    v = v + x[port[0] - 1, 0]
                if port[1]:
                    v = v - x[port[1] - 1, 0]
                v_dports.append(v)
            if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
                if not nojac:
                    iis, gs = elem.gstamp(v_dports, time)
                    row += list(iis[0])
                    col += list(iis[1])
                    val += list(gs.reshape(-1))
                #            J[iis] += gs.reshape(-1)
                iis, i = elem.istamp(v_dports, time)
                # Tx[iis] += i.reshape(-1)
                continue
            if n1 or n2:
                iel = elem.i(index, v_dports, time)
            if n1:
                # Tx[n1m1, 0] = Tx[n1m1, 0] + iel
                iis1.append(n1m1)
                i.append(+iel)
            if n2:
                # Tx[n2m1, 0] = Tx[n2m1, 0] - iel
                iis1.append(n2m1)
                i.append(-iel)
            for iindex in range(len(dports)):
                if n1 or n2:
                    g = elem.g(index, v_dports, iindex, time)
                if n1:
                    if dports[iindex][0]:
                        row += [n1m1]
                        col += [dports[iindex][0] - 1]
                        val += [g]
                    #                    J[n1m1, dports[iindex][0] - 1] += g
                    if dports[iindex][1]:
                        row += [n1m1]
                        col += [dports[iindex][1] - 1]
                        val += [-g]
                #                    J[n1m1, dports[iindex][1] - 1] -= g
                if n2:
                    if dports[iindex][0]:
                        row += [n2m1]
                        col += [dports[iindex][0] - 1]
                        val += [-g]
                    #                    J[n2m1, dports[iindex][0] - 1] -= g
                    if dports[iindex][1]:
                        row += [n2m1]
                        col += [dports[iindex][1] - 1]
                        val += [g]
    #                    J[n2m1, dports[iindex][1] - 1] += g
    return (row, col, val), (iis1, i)


def _update_J_and_Tx_sparse(row, col, val, Tx, x, elem, time, mna_size1=0, nojac=False):
    if isinstance(elem, memristor.memristor):
        v = 0.
        if elem.n1:
            v = v + x[elem.n1 - 1, 0]
        if elem.n2:
            v = v - x[elem.n2 - 1, 0]
        xm = x[mna_size1 + elem.mem_index, 0]
        vx = [v, xm]
        if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
            if not nojac:
                iis, gs = elem.gstamp(vx, time, mna_size1)
                row += list(iis[0])
                col += list(iis[1])
                val += list(gs.reshape(-1))
            #            J[iis] += gs.reshape(-1)
            iis1, i = elem.istamp(vx, time, mna_size1)
            Tx[iis1] += i.reshape(-1)
        else:
            raise Exception("gstamp or istamp not defined for memristor")
    #            continue
    elif isinstance(elem, devices.BISource):
        v = 0.
        if elem.n1:
            v = v + x[elem.n1 - 1, 0]
        if elem.n2:
            v = v - x[elem.n2 - 1, 0]
        if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
            if not nojac:
                iis, gs = elem.gstamp(v, time)
                row += list(iis[0])
                col += list(iis[1])
                val += list(gs.reshape(-1))
            #            J[iis] += gs.reshape(-1)
            iis1, i = elem.istamp(v, time)
            Tx[iis1] += i.reshape(-1)
        else:
            raise Exception("gstamp or istamp not defined for BISource")
    elif elem.is_nonlinear:
        out_ports = elem.get_output_ports()
        for index in range(len(out_ports)):
            n1, n2 = out_ports[index]
            n1m1, n2m1 = n1 - 1, n2 - 1
            dports = elem.get_drive_ports(index)
            v_dports = []
            for port in dports:
                v = 0.  # build v: remember we removed the 0 row and 0 col of mna -> -1
                if port[0]:
                    v = v + x[port[0] - 1, 0]
                if port[1]:
                    v = v - x[port[1] - 1, 0]
                v_dports.append(v)
            if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
                if not nojac:
                    iis, gs = elem.gstamp(v_dports, time)
                    row += list(iis[0])
                    col += list(iis[1])
                    val += list(gs.reshape(-1))
                #            J[iis] += gs.reshape(-1)
                iis, i = elem.istamp(v_dports, time)
                Tx[iis] += i.reshape(-1)
                continue
            if n1 or n2:
                iel = elem.i(index, v_dports, time)
                # if elem.part_id == 'clk1_ctr:mp_1':
                #     print('clk1_ctr:mp_1 current = {}'.format(iel))
                # elif elem.part_id == 'clk1_ctr:mn_1':
                #     print('clk1_ctr:mn_1 current = {}'.format(iel))
            if n1:
                Tx[n1m1, 0] = Tx[n1m1, 0] + iel
            if n2:
                Tx[n2m1, 0] = Tx[n2m1, 0] - iel
            for iindex in range(len(dports)):
                if n1 or n2:
                    g = elem.g(index, v_dports, iindex, time)
                if n1:
                    if dports[iindex][0]:
                        row += [n1m1]
                        col += [dports[iindex][0] - 1]
                        val += [g]
                    #                    J[n1m1, dports[iindex][0] - 1] += g
                    if dports[iindex][1]:
                        row += [n1m1]
                        col += [dports[iindex][1] - 1]
                        val += [-g]
                #                    J[n1m1, dports[iindex][1] - 1] -= g
                if n2:
                    if dports[iindex][0]:
                        row += [n2m1]
                        col += [dports[iindex][0] - 1]
                        val += [-g]
                    #                    J[n2m1, dports[iindex][0] - 1] -= g
                    if dports[iindex][1]:
                        row += [n2m1]
                        col += [dports[iindex][1] - 1]
                        val += [g]
    #                    J[n2m1, dports[iindex][1] - 1] += g
    return


def generate_Tt_compact(circ, time):
    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    # Tt = np.zeros((mna_size, 1))
    # nv = circ.get_nodes_number()
    # v_eq = 0
    It = []
    Vt = []
    dVt = []
    for elem in circ:
        if isinstance(elem, devices.VSource):
            Vt.append(elem.V(time))
            # dVt.append(elem.dV(time)) # compute dVsrc/dt for EI (TODO: implement dV/dT for time_functions)
        elif isinstance(elem, devices.ISource):
            It.append(elem.I(time))

    # Tt = np.array(np.hstack(([1], It, dVt)))[:, None] 
    Tt = np.array(np.hstack((It, Vt)))[:, None]

    return Tt


def generate_Tt(circ, time, mna_size):
    # time variable component: Tt this is always the same in each iter. So we
    # build it once for all.
    # the vsource or isource is assumed constant within a time step
    Tt = np.zeros((mna_size, 1))
    nv = circ.get_nodes_number()
    v_eq = 0
    for elem in circ:
        if (isinstance(elem, devices.VSource) or isinstance(elem, devices.ISource)) and elem.is_timedependent:
            if isinstance(elem, devices.VSource):
                Tt[nv - 1 + v_eq, 0] = -1 * elem.V(time)
            elif isinstance(elem, devices.ISource):
                if elem.n1:
                    Tt[elem.n1 - 1, 0] = Tt[elem.n1 - 1, 0] + elem.I(time)
                if elem.n2:
                    Tt[elem.n2 - 1, 0] = Tt[elem.n2 - 1, 0] - elem.I(time)
        if elem.is_voltage_defined:
            v_eq = v_eq + 1

    return Tt


def build_gmin_matrix(circ, gmin, mna_size, verbose, issparse=True):
    """Build a Gmin matrix

    **Parameters:**

    circ : circuit instance
        The circuit for which the matrix is built.
    gmin : scalar float
        The value of the minimum conductance to ground to be used.
    mna_size : int
        The size of the MNA matrix associated with the GMIN matrix being built.
    verbose : int
        The verbosity level, from 0 (silent) to 6 (debug).

    **Returns:**

    Gmin : ndarray of size (mna_size, mna_size)
        The Gmin matrix itself.

    """
    printing.print_info_line(("Building Gmin matrix...", 5), verbose)
    if issparse:
        row = range(circ.get_nodes_number() - 1)
        col = range(circ.get_nodes_number() - 1)
        val = np.full(circ.get_nodes_number() - 1, gmin)
        Gmin_matrix = sp.sparse.csr_matrix((val, (row, col)), shape=(mna_size, mna_size))
    else:
        #        Gmin_matrix = np.zeros((mna_size, mna_size))
        Gmin_matrix = sp.sparse.csr_matrix((mna_size, mna_size))
        for index in range(circ.get_nodes_number() - 1):
            Gmin_matrix[index, index] = gmin
            # the three missing terms of the stample matrix go on [index,0] [0,0] [0, index] but since
            # we discarded the 0 row and 0 column, we simply don't need to add them
            # the last lines are the KVL lines, introduced by voltage sources.
            # Don't add gmin there.
    return Gmin_matrix


def set_next_solve_method(standard_solving, gmin_stepping, source_stepping, verbose=3):
    """Select the next solving method.

    We have the standard solving method and two homotopies available. The
    homotopies are :math:`G_{min}` stepping and source stepping.

    They will be selected and enabled when failures occur according to the
    options values:

    * ``options.use_standard_solve_method``,
    * ``options.use_gmin_stepping``,
    * ``options.use_source_stepping``.

    The methods will be used in the order above.

    The inputs to this method are three dictionaries that keep track of which
    method is currently enabled and which ones has failed in the past.

    **Parameters:**

    standard_solving, gmin_stepping, source_stepping : dict
        The dictionaries contain the options and the status of the methods, they
        should be the values provided by :func:`get_solve_methods`.
    verbose : int, optional
        The verbosity level, from 0 (silent) to 6 (debug).

    **Returns:**

    standard_solving, gmin_stepping, source_stepping : dict
        The updated dictionaries.
    """
    if standard_solving["enabled"]:
        printing.print_info_line(("standard nonlinear solving failed.", 1), verbose)
        standard_solving["enabled"] = False
        standard_solving["failed"] = True
    elif gmin_stepping["enabled"]:
        printing.print_info_line(("gmin stepping failed.", 1), verbose)
        gmin_stepping["enabled"] = False
        gmin_stepping["failed"] = True
    elif source_stepping["enabled"]:
        printing.print_info_line(("source stepping failed.", 1), verbose)
        source_stepping["enabled"] = False
        source_stepping["failed"] = True
    if not standard_solving["failed"] and options.use_standard_solve_method:
        standard_solving["enabled"] = True
    elif not gmin_stepping["failed"] and options.use_gmin_stepping:
        gmin_stepping["enabled"] = True
        printing.print_info_line(
            ("Enabling gmin stepping convergence aid.", 3), verbose)
    elif not source_stepping["failed"] and options.use_source_stepping:
        source_stepping["enabled"] = True
        printing.print_info_line(
            ("Enabling source stepping convergence aid.", 3), verbose)

    return standard_solving, gmin_stepping, source_stepping


def more_solve_methods_available(standard_solving, gmin_stepping, source_stepping):
    """Are there more solving methods available?

    **Parameters:**

    standard_solving, gmin_stepping, source_stepping : dict
        The dictionaries contain the options and the status of the methods.

    **Returns:**

    rsp : boolean
        The answer.
    """

    if (standard_solving["failed"] or not options.use_standard_solve_method) and \
            (gmin_stepping["failed"] or not options.use_gmin_stepping) and \
            (source_stepping["failed"] or not options.use_source_stepping):
        return False
    else:
        return True


def get_solve_methods():
    """Get all the available solving methods

    We have the standard solving method and two homotopies available. The
    homotopies are :math:`G_{min}` stepping and source stepping.

    Solving methods may be enabled and disabled through the
    options values:

    * ``options.use_standard_solve_method``,
    * ``options.use_gmin_stepping``,
    * ``options.use_source_stepping``.

    **Returns:**

    standard_solving, gmin_stepping, source_stepping : dict
        The dictionaries contain the options and the status of the methods.
    """
    standard_solving = {"enabled": False, "failed": False}
    g_indices = list(range(int(numpy.log(options.gmin)), 0))
    g_indices.reverse()
    gmin_stepping = {"enabled": False, "failed":
        False, "factors": g_indices, "index": 0}
    source_stepping = {"enabled": False, "failed": False, "factors": (
        0.001, .005, .01, .03, .1, .3, .5, .7, .8, .9), "index": 0}
    return standard_solving, gmin_stepping, source_stepping


def dc_analysis(circ, start, stop, step, source, sweep_type='LINEAR', guess=True, x0=None, outfile="stdout", verbose=3):
    """Performs a sweep of the value of V or I of a independent source from start
    value to stop value using the provided step.

    For every circuit generated, computes the OP.  This function relays on
    :func:`dc_analysis.op_analysis` to actually solve each circuit.

    **Parameters:**

    circ : Circuit instance
        The circuit instance to be simulated.
    start : float
        Start value of the sweep source
    stop : float
        Stop value of the sweep source
    step : float
        The step size in the sweep
    source : string
        The part ID of the source to be swept, eg. ``'V1'``.
    sweep_type : string, optional
        Either options.dc_lin_step (default) or options.dc_log_step
    guess : boolean, optional
        op_analysis will guess to start the first NR iteration for the first point,
        the previsious OP is used from then on. Defaults to ``True``.
    outfile : string, optional
        Filename of the output file. If set to ``'stdout'`` (default), prints to
        screen.
    verbose : int
        The verbosity level, from 0 (silent) to 6 (debug).

    **Returns:**

    rstdc : results.dc_solution instance or None
        A ``results.dc_solution`` instance is returned, if a solution was found
        for at least one sweep value.  or ``None``, if an error occurred (eg
        invalid start/stop/step values) or there was no solution for any
        sweep value.
    """
    if outfile == 'stdout':
        verbose = 0
    printing.print_info_line(("Starting DC analysis:", 2), verbose)
    elem_type, elem_descr = source[0].lower(), source.lower()  # eg. 'v', 'v34'
    sweep_label = elem_type[0].upper() + elem_descr[1:]

    if sweep_type == options.dc_log_step and stop - start < 0:
        printing.print_general_error(
            "DC analysis has log sweeping and negative stepping.")
        sys.exit(1)
    if (stop - start) * step < 0:
        raise ValueError("Unbonded stepping in DC analysis.")

    points = (stop - start) / step + 1
    sweep_type = sweep_type.upper()[:3]

    if sweep_type == options.dc_log_step:
        dc_iter = utilities.log_axis_iterator(start, stop, points=points)
    elif sweep_type == options.dc_lin_step:
        dc_iter = utilities.lin_axis_iterator(start, stop, points=points)
    else:
        printing.print_general_error("Unknown sweep type: %s" % (sweep_type,))
        sys.exit(1)

    if elem_type != 'v' and elem_type != 'i':
        printing.print_general_error(
            "Sweeping is possible only with voltage and current sources. (" + str(elem_type) + ")")
        sys.exit(1)

    source_elem = None
    for index in range(len(circ)):
        if circ[index].part_id.lower() == elem_descr:
            if elem_type == 'v':
                if isinstance(circ[index], devices.VSource):
                    source_elem = circ[index]
                    break
            if elem_type == 'i':
                if isinstance(circ[index], devices.ISource):
                    source_elem = circ[index]
                    break
    if not source_elem:
        raise ValueError(".DC: source %s was not found." % source)

    if isinstance(source_elem, devices.VSource):
        initial_value = source_elem.dc_value
    else:
        initial_value = source_elem.dc_value

    # If the initial value is set to None, op_analysis will attempt a smart guess (if guess),
    # Then for each iteration, the last result is used as x0, since op_analysis will not
    # attempt to guess the op if x0 is not None.
    x = x0

    sol = results.dc_solution(
        circ, start, stop, sweepvar=sweep_label, stype=sweep_type, outfile=outfile)

    printing.print_info_line(("Solving... ", 3), verbose, print_nl=False)
    tick = ticker.ticker(1)
    tick.display(verbose > 2)

    # sweep setup

    # tarocca il generatore di tensione, avvia DC silenziosa, ritarocca etc
    index = 0
    for sweep_value in dc_iter:
        index = index + 1
        if isinstance(source_elem, devices.VSource):
            source_elem.dc_value = sweep_value
        else:
            source_elem.dc_value = sweep_value
        # silently calculate the op
        x = op_analysis(circ, x0=x, guess=guess, verbose=0)
        if x is None:
            tick.hide(verbose > 2)
            if not options.dc_sweep_skip_allowed:
                print("Could't solve the circuit for sweep value:", start + index * step)
                solved = False
                break
            else:
                print("Skipping sweep value:", start + index * step)
                continue
        solved = True
        sol.add_op(sweep_value, x)

        tick.step()

    tick.hide(verbose > 2)
    if solved:
        printing.print_info_line(("done", 3), verbose)

    # clean up
    if isinstance(source_elem, devices.VSource):
        source_elem.dc_value = initial_value
    else:
        source_elem.dc_value = initial_value

    return sol if solved else None


def op_analysis(circ, x0=None, guess=True, outfile=None, printvar=None, verbose=3):
    """Runs an Operating Point (OP) analysis

    **Parameters:**

    circ : Circuit instance
        The circuit instance on which the simulation is run
    x0 : op_solution instance or ndarray, optional
        The initial guess to be used to start the NR :func:`mdn_solver`.
    guess : boolean, optional
        If set to ``True`` (default) and ``x0`` is ``None``, it will generate a
        'smart' guess to use as ``x0``.
    verbose : int
        The verbosity level from 0 (silent) to 6 (debug).

    **Returns:**

    A ``result.op_solution`` instance, if successful, ``None`` otherwise.
    """
    if outfile == 'stdout':
        verbose = 0  # silent mode, print out results only.
    if not options.dc_use_guess:
        guess = False

    #    data = spio.loadmat('mna_N.mat') # load pre-computed mna and N matrices
    #    mna = data['mna'].tocsr()
    #    N = data['N'].tocsr()
    #    import operator
    #    mna, N = operator.itemgetter('mna', 'N')(data)
    t = timeit.time()
    (mna, N) = generate_mna_and_N(circ, verbose=verbose)  # sparse version of mna and N matrices
    t = timeit.time() - t
    #    (mna0, N0) = generate_mna_and_N(circ, verbose=verbose)
    #    print(timeit.time() - t)

    printing.print_info_line(
        ("MNA matrix and constant term generated in %f s (complete):" % t, 2), verbose)
    #    printing.print_info_line((mna, 4), verbose)
    #    printing.print_info_line((N, 4), verbose)

    # lets trash the unneeded col & row
    printing.print_info_line(
        ("Removing unneeded row and column...", 4), verbose)
    mna = utilities.remove_row_and_col(mna)
    N = utilities.remove_row(N, rrow=0)

    printing.print_info_line(("Starting op analysis:", 2), verbose)

    if x0 is None and guess:
        x0 = dc_guess.get_dc_guess(circ, verbose=verbose)
    # if options.testcase == 'clk': # for ram case only
    #     x0 = spio.loadmat('x0_clk.mat')['x']
    #     print('use x0_clk.mat as initial guess')      
    # if x0 is not None, use that
    Tx0 = 0
    printing.print_info_line(("Solving with Gmin:", 4), verbose)
    Gmin_matrix = build_gmin_matrix(circ, options.gmin, mna.shape[0], verbose - 2, issparse=True)

    if options.lin_sol_method == 'pardiso':
        pardiso = PyPardisoSolver()
        pardiso.set_iparm(1, 1)
        pardiso.set_iparm(2, 0)
        # for i in range(5):
        # pardiso.set_iparm(3, 4)
        pardiso.set_iparm(10, 13)
        pardiso.set_iparm(11, 1)
        pardiso.set_iparm(13, 1)
        pardiso.set_iparm(21, 1)
        pardiso.set_iparm(25, 1)
        pardiso.set_iparm(34, 1)
        linsolver = {'name': 'pardiso', 'param': pardiso}
    elif options.lin_sol_method == 'GMRES':
        linsolver = {'name': 'GMRES', 'tol': options.lin_GMRES_tol, 'maxiter': options.lin_GMRES_maxiter}
    else:
        linsolver = {'name': 'splu'}

    bsimOpt = {}
    if options.useBsim3:
        bsimOpt = {'mode': 'dc', 'first_time': 1, 'iter': 1, 'outFlag': 0}

    (x1, Tx1, error1, solved1, n_iter1, _, _) = dc_solve(mna, N,
                                                         circ, Gmin=Gmin_matrix, x0=x0, Tx0=Tx0, skip_Tt=True,
                                                         bsimOpt=bsimOpt, linsolver=linsolver, verbose=verbose)

    # We'll check the results now. Recalculate them without Gmin (using previsious solution as initial guess)
    # and check that differences on nodes and current do not exceed the
    # tolerances.
    if options.useBsim3:
        bsimOpt = {'mode': 'dc', 'first_time': 1, 'iter': 1, 'outFlag': 0}
    if solved1:
        op1 = results.op_solution(
            x1, error1, circ, outfile=outfile, iterations=n_iter1)
        printing.print_info_line(("Solving without Gmin:", 4), verbose)
        (x2, Tx2, error2, solved2, n_iter2, _, _) = dc_solve(
            mna, N, circ, Gmin=None, x0=x1, Tx0=Tx1, skip_Tt=True, bsimOpt=bsimOpt, linsolver=linsolver,
            verbose=verbose)
    else:
        solved2 = False

    if solved1 and not solved2:
        printing.print_general_error("Can't solve without Gmin.")
        if verbose:
            print("Displaying latest valid results.")
            op1.write_to_file(filename='stdout')
        opsolution = op1
    elif solved1 and solved2:
        op2 = results.op_solution(
            x2, error2, circ, outfile=outfile, iterations=n_iter1 + n_iter2)
        op2.gmin = 0
        badvars = results.op_solution.gmin_check(op2, op1)
        printing.print_result_check(badvars, verbose=verbose)
        check_ok = not (len(badvars) > 0)
        if not check_ok and verbose:
            print("Solution with Gmin:")
            #            op1.write_to_file(filename='stdout')
            print("Solution without Gmin:")
        #            op2.write_to_file(filename='stdout')
        opsolution = op2
    else:  # not solved1
        printing.print_general_error("Couldn't solve the circuit. Giving up.")
        opsolution = None

    if opsolution and outfile != 'stdout' and outfile is not None:
        pass
    #        opsolution.write_to_file()
    if opsolution and (verbose > 2 or outfile == 'stdout') and options.cli:
        pass
    #        opsolution.write_to_file(filename='stdout')
    #    exit(32)
    return opsolution


def _update_J_and_Tx(J, Tx, x, elem, time):
    out_ports = elem.get_output_ports()
    for index in range(len(out_ports)):
        n1, n2 = out_ports[index]
        n1m1, n2m1 = n1 - 1, n2 - 1
        dports = elem.get_drive_ports(index)
        v_dports = []
        for port in dports:
            v = 0.  # build v: remember we removed the 0 row and 0 col of mna -> -1
            if port[0]:
                v = v + x[port[0] - 1, 0]
            if port[1]:
                v = v - x[port[1] - 1, 0]
            v_dports.append(v)
        if hasattr(elem, 'gstamp') and hasattr(elem, 'istamp'):
            iis, gs = elem.gstamp(v_dports, time)
            J[iis] += gs.reshape(-1)
            iis, i = elem.istamp(v_dports, time)
            Tx[iis] += i.reshape(-1)
            continue
        if n1 or n2:
            iel = elem.i(index, v_dports, time)
        if n1:
            Tx[n1m1, 0] = Tx[n1m1, 0] + iel
        if n2:
            Tx[n2m1, 0] = Tx[n2m1, 0] - iel
        for iindex in range(len(dports)):
            if n1 or n2:
                g = elem.g(index, v_dports, iindex, time)
            if n1:
                if dports[iindex][0]:
                    J[n1m1, dports[iindex][0] - 1] += g
                if dports[iindex][1]:
                    J[n1m1, dports[iindex][1] - 1] -= g
            if n2:
                if dports[iindex][0]:
                    J[n2m1, dports[iindex][0] - 1] -= g
                if dports[iindex][1]:
                    J[n2m1, dports[iindex][1] - 1] += g


def get_td(dx, locked_nodes, n=-1):
    """Calculates the damping coefficient for the Newthon method.

    The damping coefficient is choosen as the lowest between:

    - the damping required for the first NR iterations, a parameter which is set
      through the integer ``options.nr_damp_first_iters``.
    - If ``options.nl_voltages_lock`` evaluates to ``True``, the biggest damping
      factor that keeps the change in voltage across the locked nodes pairs less
      than the maximum variation allowed, set by:
      ``(options.nl_voltages_lock_factor * Vth)``
    - Unity.

    **Parameters:**

    dx : ndarray
        The undamped increment returned by the NR solver.
    locked_nodes : list
        A vector of tuples of (internal) nodes that are a port of a non-linear
        component.
    n : int, optional
        The NR iteration counter

    .. note::

        If ``n`` is set to ``-1`` (or any negative value), ``td`` is independent
        from the iteration number and ``options.nr_damp_first_iters`` is ignored.

    **Returns:**

    td : float
        The damping coefficient.

    """

    if not options.nr_damp_first_iters or n < 0:
        td = 1
    else:
        if n < 10:
            td = 1e-2
        elif n < 20:
            td = 0.1
        else:
            td = 1
    td_new = 1
    if options.nl_voltages_lock:
        for (n1, n2) in locked_nodes:
            if n1 != 0:
                if n2 != 0:
                    if abs(dx[n1 - 1, 0] - dx[n2 - 1, 0]) > options.nl_voltages_lock_factor * constants.Vth():
                        td_new = (options.nl_voltages_lock_factor * constants.Vth()) / abs(
                            dx[n1 - 1, 0] - dx[n2 - 1, 0])
                else:
                    if abs(dx[n1 - 1, 0]) > options.nl_voltages_lock_factor * constants.Vth():
                        td_new = (options.nl_voltages_lock_factor * constants.Vth()) / abs(
                            dx[n1 - 1, 0])
            else:
                if abs(dx[n2 - 1, 0]) > options.nl_voltages_lock_factor * constants.Vth():
                    td_new = (options.nl_voltages_lock_factor * constants.Vth()) / abs(
                        dx[n2 - 1, 0])
            if td_new < td:
                td = td_new
    return td


def generate_mna_and_N(circ, verbose=3):
    """Generate the full *unreduced* MNA and N matrices required for an MNA analysis

    We wish to solve the linear stationary MNA problem:

    .. math::

        MNA \\cdot x + N = 0

    If ``nv`` is the number of nodes in the circuit, ``MNA`` is a square matrix
    composed by:

    * ``MNA[:nv, :]``, KCLs ordered by node, from node 0 up to node nv.

    In the above submatrix, we have a voltage part: ``MNA[:nv, :nv]``, where
    each term ``MNA[i, j]`` is due to the (trans-)conductances in between the
    nodes and a current part, ``MNA[:nv, nv:]``, where each term is due to a
    current variable introduced by elements whose current flow is not univocally
    defined by the voltage applied to their port(s).

    * ``MNA[nv:, :]`` are the KVL equations introduced by the above terms.

    ``N`` is similarly partitioned, but it is a vector of size ``(nv,)``.

    **Parameters:**

    circ : circuit instance
        The circuit for which the matrices are to be computed.
    verbose : int, optional
        The verbosity, from 0 (silent) to 6 (debug).

    **Returns:**

    MNA, N : ndarrays
        The MNA matrix and constant term vector computed as per above.

    """
    n_of_nodes = circ.get_nodes_number()

    if options.use_sparse:
        row, col, val = [], [], []
        rowN, colN, valN = [], [], []
        t = timeit.time()
        for elem in circ:
            if elem.is_nonlinear:
                continue
            elif isinstance(elem, devices.Resistor):
                row += [elem.n1, elem.n1, elem.n2, elem.n2]
                col += [elem.n1, elem.n2, elem.n1, elem.n2]
                val += [elem.g, -elem.g, -elem.g, elem.g]
            elif isinstance(elem, devices.Capacitor):
                pass  # In a capacitor I(V) = 0
            elif isinstance(elem, devices.GISource):
                row += [elem.n1, elem.n1, elem.n2, elem.n2]
                col += [elem.sn1, elem.sn2, elem.sn1, elem.sn2]
                val += [elem.alpha, -elem.alpha, -elem.alpha, elem.alpha]
            elif isinstance(elem, devices.ISource):
                if not elem.is_timedependent:  # convenzione normale!
                    rowN += [elem.n1, elem.n2]
                    colN += [0, 0]
                    valN += [elem.I(), -elem.I()]
                else:
                    pass  # vengono aggiunti volta per volta
            elif isinstance(elem, devices.InductorCoupling):
                pass
                # this is taken care of within the inductors
            elif elem.is_voltage_defined:
                pass
                # we'll add its lines afterwards
            elif isinstance(elem, devices.FISource):
                # we add these last, they depend on voltage sources
                # to sense the current
                pass
            else:
                print("dc_analysis.py: BUG - Unknown linear element. Ref. #28934")
        # process vsources
        # i generatori di tensione non sono pilotabili in tensione: g e' infinita
        # for each vsource, introduce a new variable: the current flowing through it.
        # then we introduce a KVL equation to be able to solve the circuit
        index = n_of_nodes - 1
        for elem in circ:
            if elem.is_voltage_defined:
                #            index = mna.shape[0]  # get_matrix_size(mna)[0]
                #            mna = utilities.expand_matrix(mna, add_a_row=True, add_a_col=True)
                #            N = utilities.expand_matrix(N, add_a_row=True, add_a_col=False)
                index = index + 1
                # KCL
                row += [elem.n1, elem.n2, index, index]
                col += [index, index, elem.n1, elem.n2]
                val += [+1.0, -1.0, +1.0, -1.0]
                if isinstance(elem, devices.VSource) and not elem.is_timedependent:
                    # corretto, se e' def una parte tempo-variabile ci pensa
                    # mdn_solver a scegliere quella giusta da usare.
                    rowN += [index]
                    colN += [0]
                    valN += [-1.0 * elem.V()]
                    elem.index = index - 1  # due to the elimination of gnd node later
                elif isinstance(elem, devices.VSource) and elem.is_timedependent:
                    elem.index = index - 1  # due to the elimination of gnd node later
                    pass  # taken care step by step
                elif isinstance(elem, devices.EVSource):
                    row += [index, index]
                    col += [elem.sn1, elem.sn2]
                    val += [-1.0 * elem.alpha, +1.0 * elem.alpha]
                elif isinstance(elem, devices.Inductor):
                    # N[index,0] = 0 pass, it's already zero
                    pass
                elif isinstance(elem, devices.HVSource):
                    index_source = circ.find_vde_index(elem.source_id)
                    row += [index]
                    col += [n_of_nodes + index_source]
                    val += [1.0 * elem.alpha]
                elif isinstance(elem, devices.Memristor):
                    pass
                else:
                    print("dc_analysis.py: BUG - found an unknown voltage_def elem.")
                    print(elem)
                    sys.exit(33)

        # iterate again for devices that depend on voltage-defined ones.
        for elem in circ:
            if isinstance(elem, devices.FISource):
                local_i_index = circ.find_vde_index(elem.source_id, verbose=0)
                row += [elem.n1, elem.n2]
                col += [n_of_nodes + local_i_index, n_of_nodes + local_i_index]
                val += [elem.alpha, -elem.alpha]

        for elem in circ:
            if isinstance(elem, memristor.memristor):
                index = index + 1  # each memristor introduces one extra variable (x), appended after the currents variables

        printing.print_info_line(("  coordinate assembly time: {} s".format(timeit.time() - t), 4), verbose)

        t = timeit.time()
        mna = sp.sparse.csr_matrix((val, (row, col)), shape=(index + 1, index + 1))
        N = sp.sparse.csr_matrix((valN, (rowN, colN)), shape=(index + 1, 1))
        printing.print_info_line(("  sparse matrix generation time: {} s".format(timeit.time() - t), 4), verbose)
    else:
        mna = np.zeros((n_of_nodes, n_of_nodes))
        N = np.zeros((n_of_nodes, 1))
        for elem in circ:
            if elem.is_nonlinear:
                continue
            elif isinstance(elem, devices.Resistor):
                mna[elem.n1, elem.n1] = mna[elem.n1, elem.n1] + elem.g
                mna[elem.n1, elem.n2] = mna[elem.n1, elem.n2] - elem.g
                mna[elem.n2, elem.n1] = mna[elem.n2, elem.n1] - elem.g
                mna[elem.n2, elem.n2] = mna[elem.n2, elem.n2] + elem.g
            elif isinstance(elem, devices.Capacitor):
                pass  # In a capacitor I(V) = 0
            elif isinstance(elem, devices.GISource):
                mna[elem.n1, elem.sn1] = mna[elem.n1, elem.sn1] + elem.alpha
                mna[elem.n1, elem.sn2] = mna[elem.n1, elem.sn2] - elem.alpha
                mna[elem.n2, elem.sn1] = mna[elem.n2, elem.sn1] - elem.alpha
                mna[elem.n2, elem.sn2] = mna[elem.n2, elem.sn2] + elem.alpha
            elif isinstance(elem, devices.ISource):
                if not elem.is_timedependent:  # convenzione normale!
                    N[elem.n1, 0] = N[elem.n1, 0] + elem.I()
                    N[elem.n2, 0] = N[elem.n2, 0] - elem.I()
                else:
                    pass  # vengono aggiunti volta per volta
            elif isinstance(elem, devices.InductorCoupling):
                pass
                # this is taken care of within the inductors
            elif elem.is_voltage_defined:
                pass
                # we'll add its lines afterwards
            elif isinstance(elem, devices.FISource):
                # we add these last, they depend on voltage sources
                # to sense the current
                pass
            else:
                print("dc_analysis.py: BUG - Unknown linear element. Ref. #28934")
        # process vsources
        # i generatori di tensione non sono pilotabili in tensione: g e' infinita
        # for each vsource, introduce a new variable: the current flowing through it.
        # then we introduce a KVL equation to be able to solve the circuit
        for elem in circ:
            if elem.is_voltage_defined:
                index = mna.shape[0]  # get_matrix_size(mna)[0]
                mna = utilities.expand_matrix(mna, add_a_row=True, add_a_col=True)
                N = utilities.expand_matrix(N, add_a_row=True, add_a_col=False)
                # KCL
                mna[elem.n1, index] = 1.0
                mna[elem.n2, index] = -1.0
                # KVL
                mna[index, elem.n1] = +1.0
                mna[index, elem.n2] = -1.0
                if isinstance(elem, devices.VSource) and not elem.is_timedependent:
                    # corretto, se e' def una parte tempo-variabile ci pensa
                    # mdn_solver a scegliere quella giusta da usare.
                    N[index, 0] = -1.0 * elem.V()
                elif isinstance(elem, devices.VSource) and elem.is_timedependent:
                    pass  # taken care step by step
                elif isinstance(elem, devices.EVSource):
                    mna[index, elem.sn1] = -1.0 * elem.alpha
                    mna[index, elem.sn2] = +1.0 * elem.alpha
                elif isinstance(elem, devices.Inductor):
                    # N[index,0] = 0 pass, it's already zero
                    pass
                elif isinstance(elem, devices.HVSource):
                    index_source = circ.find_vde_index(elem.source_id)
                    mna[index, n_of_nodes + index_source] = 1.0 * elem.alpha
                else:
                    print("dc_analysis.py: BUG - found an unknown voltage_def elem.")
                    print(elem)
                    sys.exit(33)

        # iterate again for devices that depend on voltage-defined ones.
        for elem in circ:
            if isinstance(elem, devices.FISource):
                local_i_index = circ.find_vde_index(elem.source_id, verbose=0)
                mna[elem.n1, n_of_nodes + local_i_index] = mna[elem.n1, n_of_nodes + local_i_index] + elem.alpha
                mna[elem.n2, n_of_nodes + local_i_index] = mna[elem.n2, n_of_nodes + local_i_index] - elem.alpha

    # Seems a good place to run some sanity check
    # for the time being we do not halt the execution
    t = timeit.time()
    #    utilities.check_ground_paths(mna, circ, reduced_mna=False, verbose=verbose)
    printing.print_info_line(("  ground path check: {} s".format(timeit.time() - t), 4), verbose)
    #    sys.exit(32)
    #    utilities.check_ground_paths(mna, circ, reduced_mna=False, verbose=verbose)

    # all done
    return (mna, N)


def generate_mna_and_B(circ, verbose=3):
    """Generate the full *unreduced* MNA and N matrices required for an MNA analysis

    We wish to solve the linear stationary MNA problem:

    .. math::

        MNA \\cdot x + Bu = 0

    If ``nv`` is the number of nodes in the circuit, ``MNA`` is a square matrix
    composed by:

    * ``MNA[:nv, :]``, KCLs ordered by node, from node 0 up to node nv.

    In the above submatrix, we have a voltage part: ``MNA[:nv, :nv]``, where
    each term ``MNA[i, j]`` is due to the (trans-)conductances in between the
    nodes and a current part, ``MNA[:nv, nv:]``, where each term is due to a
    current variable introduced by elements whose current flow is not univocally
    defined by the voltage applied to their port(s).

    * ``MNA[nv:, :]`` are the KVL equations introduced by the above terms.

    ``B`` is similarly partitioned, but it is a vector of size ``(nv, np)``., where p is the total number of sources
    
    u is assumed to has the form [Isource, Vsource]^T, i.e., current sources first and voltage sources later
    
    **Parameters:**

    circ : circuit instance
        The circuit for which the matrices are to be computed.
    verbose : int, optional
        The verbosity, from 0 (silent) to 6 (debug).

    **Returns:**

    MNA, B : ndarrays
        The MNA matrix and input matrix as per above.

    """
    n_of_nodes = circ.get_nodes_number()

    if options.use_sparse:
        row, col, val = [], [], []
        rowB, colB, valB = [], [], []
        id_IS = 0
        t = timeit.time()
        for elem in circ:
            if elem.is_nonlinear:
                continue
            elif isinstance(elem, devices.Resistor):
                row += [elem.n1, elem.n1, elem.n2, elem.n2]
                col += [elem.n1, elem.n2, elem.n1, elem.n2]
                val += [elem.g, -elem.g, -elem.g, elem.g]
            elif isinstance(elem, devices.Capacitor):
                pass  # In a capacitor I(V) = 0
            elif isinstance(elem, devices.GISource):
                row += [elem.n1, elem.n1, elem.n2, elem.n2]
                col += [elem.sn1, elem.sn2, elem.sn1, elem.sn2]
                val += [elem.alpha, -elem.alpha, -elem.alpha, elem.alpha]
            elif isinstance(elem, devices.ISource):
                # if not elem.is_timedependent:  # convenzione normale!
                rowB += [elem.n1, elem.n2]
                colB += [id_IS, id_IS]
                valB += [1, -1]
                id_IS += 1
                # else:
                #     pass  # vengono aggiunti volta per volta
            elif isinstance(elem, devices.InductorCoupling):
                pass
                # this is taken care of within the inductors
            elif elem.is_voltage_defined:
                pass
                # we'll add its lines afterwards
            elif isinstance(elem, devices.FISource):
                # we add these last, they depend on voltage sources
                # to sense the current
                pass
            else:
                print("dc_analysis.py: BUG - Unknown linear element. Ref. #28934")
        # process vsources
        # i generatori di tensione non sono pilotabili in tensione: g e' infinita
        # for each vsource, introduce a new variable: the current flowing through it.
        # then we introduce a KVL equation to be able to solve the circuit
        index = n_of_nodes - 1
        id_VS = 0
        for elem in circ:
            if elem.is_voltage_defined:
                #            index = mna.shape[0]  # get_matrix_size(mna)[0]
                #            mna = utilities.expand_matrix(mna, add_a_row=True, add_a_col=True)
                #            N = utilities.expand_matrix(N, add_a_row=True, add_a_col=False)
                index = index + 1
                # KCL
                row += [elem.n1, elem.n2, index, index]
                col += [index, index, elem.n1, elem.n2]
                val += [+1.0, -1.0, +1.0, -1.0]
                if isinstance(elem, devices.VSource):  # and not elem.is_timedependent:
                    rowB += [index]
                    colB += [id_IS + id_VS]
                    valB += [-1.0]
                    id_VS += 1
                    elem.index = index - 1  # because later the gnd node will be removed, so all indices reduced by 1
                # elif isinstance(elem, devices.VSource) and elem.is_timedependent:
                #     pass  # taken care step by step
                elif isinstance(elem, devices.EVSource):
                    row += [index, index]
                    col += [elem.sn1, elem.sn2]
                    val += [-1.0 * elem.alpha, +1.0 * elem.alpha]
                elif isinstance(elem, devices.Inductor):
                    # N[index,0] = 0 pass, it's already zero
                    pass
                elif isinstance(elem, devices.HVSource):
                    index_source = circ.find_vde_index(elem.source_id)
                    row += [index]
                    col += [n_of_nodes + index_source]
                    val += [1.0 * elem.alpha]
                elif isinstance(elem, devices.Memristor):
                    pass
                else:
                    print("dc_analysis.py: BUG - found an unknown voltage_def elem.")
                    print(elem)
                    sys.exit(33)

        # iterate again for devices that depend on voltage-defined ones.
        for elem in circ:
            if isinstance(elem, devices.FISource):
                local_i_index = circ.find_vde_index(elem.source_id, verbose=0)
                row += [elem.n1, elem.n2]
                col += [n_of_nodes + local_i_index, n_of_nodes + local_i_index]
                val += [elem.alpha, -elem.alpha]

        for elem in circ:
            if isinstance(elem, memristor.memristor):
                index = index + 1  # each memristor introduces one extra variable (x), appended after the currents variables

        printing.print_info_line(("  coordinate assembly time: {} s".format(timeit.time() - t), 4), verbose)

        t = timeit.time()
        mna = sp.sparse.csc_matrix((val, (row, col)), shape=(index + 1, index + 1))
        B = sp.sparse.csc_matrix((valB, (rowB, colB)), shape=(index + 1, id_IS + id_VS))
        printing.print_info_line(("  sparse matrix generation time: {} s".format(timeit.time() - t), 4), verbose)
    else:
        print('Error: Non-sparse version not supported')

    # Seems a good place to run some sanity check
    # for the time being we do not halt the execution
    t = timeit.time()
    #    utilities.check_ground_paths(mna, circ, reduced_mna=False, verbose=verbose)
    printing.print_info_line(("  ground path check: {} s".format(timeit.time() - t), 4), verbose)
    #    sys.exit(32)
    #    utilities.check_ground_paths(mna, circ, reduced_mna=False, verbose=verbose)

    # all done
    return (mna, B)


def build_x0_from_user_supplied_ic(circ, icdict):
    """Builds a vector of appropriate (reduced!) size from the values supplied
    in ``icdict``.

    Supplying a custom x0 can be useful:
    - To aid convergence in tough circuits,
    - To start a transient simulation from a particular x0.

    **Parameters:**

    circ: circuit instance
        The circuit the :math:`x_0` is being assembled for
    icdict: dict
        ``icdict`` is a a dictionary assembled as follows:
         - to specify a nodal voltage: ``{'V(node)':<voltage value>}``
           Eg. ``{'V(n1)':2.3, 'V(n2)':0.45, ...}``.
           All unspecified voltages default to 0.
         - to specify a branch current: ``'I(<element>)':<current value>}``
           ie. the elements names are sorrounded by ``I(...)``.
           Eg. ``{'I(L1)':1.03e-3, I(V4):2.3e-6, ...}``
           All unspecified currents default to 0.

    Notes: this simulator uses the standard convention.

    **Returns:**

    x0 : ndarray
        The x0 matrix assembled according to ``icdict``.

    :raises ValueError: whenever a malformed ``icdict`` is supplied.
    """
    #    Vregex = re.compile("V\s*\(\s*([a-z0-9]+)\s*\)", re.IGNORECASE | re.DOTALL)
    #    Iregex = re.compile("I\s*\(\s*([a-z0-9]+)\s*\)", re.IGNORECASE | re.DOTALL)
    Vregex = re.compile("V\s*\(\s*(.*?)\s*\)", re.IGNORECASE | re.DOTALL)
    Iregex = re.compile("I\s*\(\s*(.*?)\s*\)", re.IGNORECASE | re.DOTALL)
    nv = circ.get_nodes_number()  # number of voltage variables
    voltage_defined_elem_names = \
        [elem.part_id.lower() for elem in circ if elem.is_voltage_defined]
    ni = len(voltage_defined_elem_names)  # number of current variables
    memristor_names = \
        [elem.part_id.lower() for elem in circ if hasattr(elem, 'is_memristor') and elem.is_memristor]
    nx = len(memristor_names)
    x0 = np.zeros((nv + ni + nx, 1))
    for label in icdict.keys():
        value = icdict[label]
        if Vregex.search(label):
            ext_node = Vregex.findall(label)[0]
            int_node = circ.ext_node_to_int(ext_node)
            if int_node != -1:  # it is a node voltage ic
                x0[int_node, 0] = value
            else:  # it is a memristor x-state ic
                #                index = memristor_names.index(ext_node.lower())
                mem_name = ext_node[2:-2]  # omit the starting 'y:' and ending "_x"
                index = memristor_names.index(mem_name.lower())
                x0[nv + ni + index, 0] = value
        elif Iregex.search(label):
            element_name = Iregex.findall(label)[0]
            try:
                index = voltage_defined_elem_names.index(element_name.lower())
                x0[nv + index, 0] = value
            except ValueError:
                pass
        else:
            pass
            # raise ValueError("Unrecognized label " + label)
    return x0[1:, :]


def modify_x0_for_ic(circ, x0):
    """Modifies a supplied x0.

    Several circut elements allow the user to set their own Initial
    Conditions (IC) for either voltage or current, depending on what
    is appropriate for the element considered.

    This method, receives a preliminary ``x0`` value, typically computed
    by an OP analysis and goes through the circuit, looking for ICs and
    setting them in ``x0``.

    Notice it is possible to require ICs that are incompatible with each
    other -- for example supplying different ICs to two parallel caps.
    In that case we try to accommodate the user's requirements in a
    non-strict best-effort kind of way: for this reason, whenever
    multiple ICs are specified, it is best to visually inspect ``x0``
    to check that what you would have expected is indeed what you got.

    **Parameters**

    circ : circuit instance
        The circuit in which the ICs are specified.
    x0 : ndarray or results.op_solution
        The initial value to be modified

    **Returns:**

    x0p : ndarray or results.op_solution
        The modified ``x0``. Notice that we return the same
        kind of object as it was supplied. Additionally,
        the ``results.op_solution`` is a *new* *instance*,
        while the ``ndarray`` is simply the original array
        modified.
    """

    if isinstance(x0, results.op_solution):
        x0 = copy.copy(x0.asarray())
        return_obj = True
    else:
        return_obj = False

    nv = circ.get_nodes_number()  # number of voltage variables
    voltage_defined_elements = [
        x for x in circ if x.is_voltage_defined]

    # setup voltages this may _not_ work properly
    for elem in circ:
        # print(elem.part_id)
        if isinstance(elem, devices.Capacitor) and elem.ic or \
                isinstance(elem, diode.diode) and elem.ic:
            x0[elem.n1 - 1, 0] = x0[elem.n2 - 1, 0] + elem.ic

    # setup the currents
    for elem in voltage_defined_elements:
        if isinstance(elem, devices.Inductor) and elem.ic:
            x0[nv - 1 + voltage_defined_elements.index(elem), 0] = elem.ic

    if return_obj:
        xnew = results.op_solution(x=x0, \
                                   error=np.zeros(x0.shape), circ=circ, outfile=None)
        xnew.netlist_file = None
        xnew.netlist_title = "Self-generated OP to be used as tran IC"
    else:
        xnew = x0

    return xnew


def lin_solve(M, rhs, linsolver, sparse):
    if sparse:
        #            t1 = timeit.time()
        if linsolver['name'] == 'pardiso':
            #                print(linsolver._is_already_factorized(M))
            dx = spsolve(M, rhs, factorize=True, squeeze=True, solver=linsolver['param']).reshape((-1, 1))
        elif linsolver['name'] == 'GMRES':
            #                dx, info = sp.sparse.linalg.gmres(M, rhs, x0=dx, tol=linsolver['tol'], maxiter=linsolver['maxiter'])
            dx, info, niter = utilities.GMRES_wrapper(M, rhs, dx, linsolver['tol'], linsolver['maxiter'])
            dx = dx.reshape((-1, 1))
            if info == 0:
                print('GMRES converged to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
            else:
                print('GMRES does not converge to {0} in {1} iterations'.format(options.lin_GMRES_tol, niter))
        elif linsolver['name'] == 'splu':
            #                lu = sp.sparse.linalg.splu(M)
            #                dx = lu.solve(rhs)
            dx = sp.sparse.linalg.spsolve(M, rhs)[:, None]
        else:
            raise ValueError('undefined linear solver')
    #            printing.print_info_line(("pardiso time: {} s".format(timeit.time() - t1), 3), verbose)
    else:
        dx = np.linalg.solve(M, rhs)

    return dx


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
        nv = circ.nv  # - 1
        nc = circ.ni
        i_eq = 0  # each time we find a vsource or vcvs or ccvs, we'll add one to this.

        for elem in circ:
            if circuit.is_elem_voltage_defined(elem) and not isinstance(elem, devices.Inductor):
                i_eq = i_eq + 1
            elif isinstance(elem, devices.Capacitor):
                if elem.value != 0:
                    row += [elem.n1, elem.n1, elem.n2, elem.n2]
                    col += [elem.n1, elem.n2, elem.n1, elem.n2]
                    val += [elem.value, -elem.value, -elem.value, elem.value]
                else:
                    pass
            elif isinstance(elem, devices.Inductor):
                if elem.value != 0:
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
                else:
                    pass
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

        D = sp.sparse.csr_matrix((val, (row, col)), shape=(shape[0] + 1, shape[1] + 1))
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
        D = np.zeros((shape[0] + 1, shape[1] + 1))
        nv = circ.get_nodes_number()  # - 1
        i_eq = 0  # each time we find a vsource or vcvs or ccvs, we'll add one to this.
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
                D[nv + i_eq, nv + i_eq] = -1 * elem.value
                # Mutual inductors (coupled inductors)
                # need to add a -M dI/dt where I is the current in the OTHER inductor.
                if len(elem.coupling_devices):
                    for cd in elem.coupling_devices:
                        # get id+descr of the other inductor (eg. "L32")
                        other_id_wdescr = cd.get_other_inductor(elem.part_id)
                        # find its index to know which column corresponds to its current
                        other_index = circ.find_vde_index(other_id_wdescr, verbose=0)
                        # add the term.
                        D[nv + i_eq, nv + other_index] += -1 * cd.M
                # carry on as usual
                i_eq = i_eq + 1

        if options.cmin > 0:
            cmin_mat = np.eye(shape[0] + 1 - i_eq)
            cmin_mat[0, 1:] = 1
            cmin_mat[1:, 0] = 1
            cmin_mat[0, 0] = cmin_mat.shape[0] - 1
            if i_eq:
                D[:-i_eq, :-i_eq] += options.cmin * cmin_mat
            else:
                D += options.cmin * cmin_mat
    return D