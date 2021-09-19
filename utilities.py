# -*- coding: iso-8859-1 -*-
# utilities.py
# Utilities file for Ahkab simulator
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
This module holds miscellaneous utility functions.

Module reference
################

"""

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

import collections
import os
import os.path
import operator
import sys
#import scipy.io as io

import numpy as np
import scipy as sp

from functools import wraps

import printing
import py3compat
import options
import devices

#: The machine epsilon, the upper bound on the relative error due to rounding in
#: floating point arithmetic.
EPS = np.finfo(float).eps


def expand_matrix(matrix, add_a_row=False, add_a_col=False):
    """Append a row and/or a column to the given matrix

    **Parameters:**

    matrix : ndarray
        The matrix to be manipulated.
    add_a_row : boolean, optional
        If set to ``True`` a row is appended to the supplied matrix.
    add_a_col : boolean
        If set to ``True`` a column is appended.

    **Returns:**

    matrix : ndarray
        A reference to the same matrix supplied.

    """
    n_row, n_col = matrix.shape
    if sp.sparse.issparse(matrix):
        if add_a_col:
            col = sp.sparse.csr_matrix((n_row, 1))
            matrix = sp.sparse.hstack([matrix, col])
        if add_a_row:
            if add_a_col:
                n_col = n_col + 1
            row = sp.sparse.csr_matrix((1, n_col))
            matrix = sp.sparse.vstack([matrix, row])
    else:
        if add_a_col:
            col = np.zeros((n_row, 1))
            matrix = np.concatenate((matrix, col), axis=1)
        if add_a_row:
            if add_a_col:
                n_col = n_col + 1
            row = np.zeros((1, n_col))
            matrix = np.concatenate((matrix, row), axis=0)
    return matrix

def set_submatrix(row, col, dest_matrix, source_matrix):
    """Copies a source matrix into another matrix

    row, col : ints
        The coordinates of the upper left corner in the destination matrix where
        the source matrix will be copied.
    dest_matrix : ndarray
        The matrix to be copied to.
    source_matrix : ndarray
        The matrix to be copied from.

    **Returns:**

    dest_matrix : ndarray
        A reference to the modified destination matrix.
    """
    ls = source_matrix.shape[0]
    cs = source_matrix.shape[1]
    dest_matrix[row:row+ls, col:col+cs] = source_matrix[:, :]
    return dest_matrix

def remove_row_and_col(matrix, rrow=0, rcol=0):
    """Removes a row and/or a column from a matrix

    **Parameters:**

    matrix : ndarray
        The matrix to be modified.
    rrow : int or None, optional
        The index of the row to be removed. If set to ``None``, no row
        will be removed. By default the first row is removed.
    rcol : int or None, optional
        The index of the row to be removed. If set to ``None``, no row
        will be removed. By default the first column is removed.

    .. note::

        No size checking is done.

    **Returns:**

    matrix : ndarray
        A reference to the modified matrix.
    """
    if sp.sparse.issparse(matrix): # remove rows/cols by fancy indexing of sparse matrices
        if rrow is not None and rcol is not None:
            row_mask = np.ones(matrix.shape[0], dtype=bool)
            row_mask[rrow] = False
            col_mask = np.ones(matrix.shape[1], dtype=bool)
            col_mask[rcol] = False
            return matrix[row_mask][:,col_mask]
#            return np.vstack((np.hstack((matrix[0:rrow, 0:rcol],
#                                         matrix[0:rrow, rcol+1:])),
#                              np.hstack((matrix[rrow+1:, 0:rcol],
#                                         matrix[rrow+1:, rcol+1:]))
#                              ))
        elif rrow is not None:
            row_mask = np.ones(matrix.shape[0], dtype=bool)
            row_mask[rrow] = False
            return matrix[row_mask]
#            return np.vstack((matrix[:rrow, :], matrix[rrow+1:, :]))
        elif rcol is not None:
            col_mask = np.ones(matrix.shape[1], dtype=bool)
            col_mask[rcol] = False
            return matrix[:,col_mask]
#            return np.hstack((matrix[:, :rcol], matrix[:, rcol+1:]))
    else: # it is a numpy array
        if rrow is not None and rcol is not None:
            return np.vstack((np.hstack((matrix[0:rrow, 0:rcol],
                                         matrix[0:rrow, rcol+1:])),
                              np.hstack((matrix[rrow+1:, 0:rcol],
                                         matrix[rrow+1:, rcol+1:]))
                              ))
        elif rrow is not None:
            return np.vstack((matrix[:rrow, :], matrix[rrow+1:, :]))
        elif rcol is not None:
            return np.hstack((matrix[:, :rcol], matrix[:, rcol+1:]))


def remove_row(matrix, rrow=0):
    """Removes a row from a matrix

    **Parameters:**

    matrix : ndarray
        The matrix to be modified.
    rrow : int or None, optional
        The index of the row to be removed. If set to ``None``, no row
        will be removed. By default the first row is removed.

    .. note::

        No size checking is done.

    **Returns:**

    matrix : ndarray
        A reference to the modified matrix.
    """
    if sp.sparse.issparse(matrix):
        row_mask = np.ones(matrix.shape[0], dtype=bool)
        row_mask[rrow] = False
        return matrix[row_mask]
    else:
        return np.vstack((matrix[:rrow, :], matrix[rrow+1:, :]))


def check_file(filename):
    """Checks whether the supplied path refers to a valid file.

    **Parameters:**

    filename : string
        The file name.

    **Returns:**

    chk : boolean
        A value of ``True`` if ``filename`` is found and it is a file.

    :raises IOError: if no such file exists or if the supplied file is a directory.
    """
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        raise IOError(filename + " not found.")
    elif not os.path.isfile(filename):
        raise IOError(filename + " is not a file.")
    return True

def tuplinator(alist):
    """Convert a list of lists (of lists...) to tuples"""
    if type(alist) is list:
        return tuple([tuplinator(i) for i in alist])
    else:
        return alist

class combinations:

    """This class is an iterator that returns all the k-combinations
    _without_repetition_ of the elements of the supplied list.

    Each combination is made of a subset of the list, consisting of k
    elements.

    **Parameters:**

    L : list
        The set from which the elements are taken.
    k : int
        The size of the subset, the number of elements to be taken
    """

    def __init__(self, L, k):
        self.L = L
        self.k = k
        self._sub_iter = None
        self._i = 0
        if len(self.L) < k:
            raise ValueError("The set has to be bigger than the subset.")
        if k <= 0:
            raise ValueError("The size of the subset has to be strictly " +
                             "positive.")

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        # It's recursive
        if self.k > 1:
            if self._sub_iter == None:
                self._sub_iter = combinations(self.L[self._i + 1:], self.k - 1)
            try:
                nxt = self._sub_iter.__next__()
                cur = self.L[self._i]
            except StopIteration:
                if self._i < len(self.L) - self.k:
                    self._i = self._i + 1
                    self._sub_iter = combinations(
                        self.L[self._i + 1:], self.k - 1)
                    return self.__next__()
                else:
                    raise StopIteration
        else:
            nxt = []
            if self._i < len(self.L):
                cur = self.L[self._i]
                self._i = self._i + 1
            else:
                raise StopIteration

        return [cur] + nxt


class log_axis_iterator:
    """This iterator provides the values for a base-10 logarithmic sweep.

    **Parameters:**

    min : float
        The minimum value, also the start point of the axis.
    max : float
        The maximum value, also the end point of the axis.
    points : int
        The number of points which will be used to discretize the ``max`` -
        ``min`` interval.

    Notice that, differently from numpy's ``logspace()``, the
    values are only computed at access time, and hence the
    memory footprint of the iterator is low.

    Start and end values are always included.
    """

    def __init__(self, min, max, points):
        self.inc = 10.**((np.log10(max) - np.log10(min))/(points - 1))
        self.max = max
        self.min = min
        self.index = 0
        self.current = min
        self.points = points

    def next(self):
        return self.__next__()

    def __next__(self):
        """Iterator method: get the next value
        """
        if self.index == 0:
            ret = self.current
        elif self.index < self.points:
            self.current = self.current * self.inc
            ret = self.current
        else:
            raise StopIteration
        self.index = self.index + 1
        return ret

    def __getitem__(self, i):
        """Iterator method: get a particular value (n. i)
        """
        if i == 0:
            ret = self.min
        elif i < self.points:
            ret = self.min * self.inc**i
        else:
            ret = None
        return ret

    def __iter__(self):
        """Required iterator method.
        """
        return self


class lin_axis_iterator:
    """This iterator provides the values for a linear sweep.

    **Parameters:**

    min : float
        The minimum value, also the start point of the axis.
    max : float
        The maximum value, also the end point of the axis.
    num : int
        The number of samples to generate. In general, this should be greater than 1.
        A value of 1 is accepted only if ``min == max``, in which case, only one
        value is returned by the iterator: ``min``. 

    Start and end points are always included.

    Notice that, differently from numpy's ``linspace()``, the
    values are only computed at access time, and hence the
    memory footprint of the iterator is low.

    :raises ValueError: if the number ``points`` is either negative or does not
    respect the conditions above.

    """

    def __init__(self, min, max, points):
        if points < 2 and min != max:
            raise ValueError('Linear iterator from %d to %d with %d points.'%
                             (min, max, points))
        if points < 1:
            raise ValueError('Linear iterator from %d to %d with %d points.'%
                             (min, max, points))
        if points > 1:
            self.inc = (max - min) / (points - 1)
        elif points == 1 and max == min:
            # sometimes, they ask for this. They expect to get back [max]
            self.inc = 0
        self.max = max
        self.min = min
        self.index = 0
        self.current = min
        self.points = points

    def next(self):
        return self.__next__()

    def __next__(self):
        """Iterator method: get the next value
        """
        if self.index == 0:
            pass  # return min
        elif self.index < self.points:
            self.current = self.current + self.inc
        else:
            raise StopIteration
        ret = self.current
        self.index = self.index + 1
        return ret

    def __getitem__(self, i):
        """Iterator method: get a particular value (n. i)
        """
        if i < self.points:
            ret = self.min + self.inc * i
        else:
            ret = None
        return ret

    def __iter__(self):
        """Required iterator method.
        """
        return self


def Celsius2Kelvin(cel):
    """Convert Celsius degrees to Kelvin
    """
    return cel + 273.15


def Kelvin2Celsius(kel):
    """Convert Kelvin degrees to Celsius
    """
    return kel - 273.15

def convergence_check(x, dx, residual, nv_minus_one, ni, debug=False):
    """Perform a convergence check

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    nv_minus_one : int
        Number of voltage variables in x. If ``nv_minus_one`` is equal to
        ``n``, it means ``x[:n]`` are all voltage variables.
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """
    if not hasattr(x, 'shape'):
        x = np.array(x)
        dx = np.array(dx)
        residual = np.array(residual)
        
    v_idx = range(nv_minus_one)
    i_idx = range(nv_minus_one, nv_minus_one + ni)
    x_idx = range(nv_minus_one + ni, len(x))
        
    vcheck, vresults = voltage_convergence_check(x[v_idx, 0],
                                                 dx[v_idx, 0],
                                                 residual[v_idx, 0], debug)
    vcheck = [vcheck] if np.isscalar(vcheck) else vcheck
    if len(i_idx) > 0:
        icheck, iresults = current_convergence_check(x[i_idx, 0],
                                                     dx[i_idx, 0],
                                                     residual[i_idx, 0], debug)
        icheck = [icheck] if np.isscalar(icheck) else icheck
    else:
        icheck = [True]
        iresults = []
    
    if options.Xyce_netlist and len(x_idx) > 0:
        xcheck, xresults = xstate_convergence_check(x[x_idx, 0],
                                                     dx[x_idx, 0],
                                                     residual[x_idx, 0], debug)
        xcheck = [xcheck] if np.isscalar(xcheck) else xcheck
    else:
        xcheck = [True]
        xresults = []
    
    # return vcheck and icheck and xcheck, vresults + iresults + xresults
    return vcheck + icheck + xcheck, vresults + iresults + xresults

def convergence_check_Xyce(x, dx, residual, nv_minus_one, ni, debug=False):
    """Perform a convergence check

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    nv_minus_one : int
        Number of voltage variables in x. If ``nv_minus_one`` is equal to
        ``n``, it means ``x[:n]`` are all voltage variables.
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """
    if not hasattr(x, 'shape'):
        x = np.array(x)
        dx = np.array(dx)
        residual = np.array(residual)
        
    v_idx = range(nv_minus_one)
    i_idx = range(nv_minus_one, nv_minus_one + ni)
    x_idx = range(nv_minus_one + ni, len(x))    
    vcheck, vresults = voltage_convergence_check(x[v_idx, 0],
                                                 dx[v_idx, 0],
                                                 residual[v_idx, 0], debug)
    icheck, iresults = current_convergence_check(x[i_idx, 0],
                                                 dx[i_idx, 0],
                                                 residual[i_idx, 0], debug)
    xcheck, xresults = xstate_convergence_check(x[x_idx, 0],
                                                 dx[x_idx, 0],
                                                 residual[x_idx, 0], debug)
    
#    return vcheck and icheck and xcheck, vresults + iresults + xresults
    return vcheck and icheck and xcheck, vresults + iresults + xresults

def voltage_convergence_check(x, dx, residual, debug=False):
    """Perform a convergence check for voltage variables

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """

    if options.Xyce_netlist:
        return custom_convergence_check_Xyce(x, dx, residual, er=options.ver,
                                ea=options.vea, edelta=options.deltaTol, eresidual=options.ier,
                                debug=debug)
    else:
        return custom_convergence_check(x, dx, residual, er=options.ver,
                                ea=options.vea, eresidual=options.ier,
                                debug=debug) 


def current_convergence_check(x, dx, residual, debug=False):
    """Perform a convergence check for current variables

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """

    if options.Xyce_netlist:
        return custom_convergence_check_Xyce(x, dx, residual, er=options.ier,
                                ea=options.iea, edelta=options.deltaTol, eresidual=options.ver,
                                debug=debug)
    else:
        return custom_convergence_check(x, dx, residual, er=options.ier,
                                ea=options.iea, eresidual=options.ver,
                                debug=debug)
                                    
def xstate_convergence_check(x, dx, residual, debug=False):
    """Perform a convergence check for current variables

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """
    if options.Xyce_netlist:
        return custom_convergence_check_Xyce(x, dx, residual, er=options.ver,
                                    ea=options.vea, edelta=options.deltaTol, eresidual=options.ver,
                                    debug=debug) 
    else:
        return custom_convergence_check(x, dx, residual, er=options.ver,
                                    ea=options.vea, eresidual=options.ver,
                                    debug=debug)                                     

def custom_convergence_check_Xyce(x, dx, residual, er, ea, edelta, eresidual, debug=False):
    """Perform a custom convergence check

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    ea : float
        The value to be employed for the absolute error.
    er : float
        The value for the relative error to be employed.
    eresidual : float
        The maximum allowed error for the residual (left over error).
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """
    all_check_results = []
    if not hasattr(x, 'shape'):
        x = np.array(x)
        dx = np.array(dx)
        residual = np.array(residual)
    if x.shape[0]:
#        xnew = x + dx
        wt = er * np.max(np.abs(x)) + ea
        wtNormDx = np.max(np.abs(dx)/wt)
        updatesize = wtNormDx / 1.0
        maxNormRHS = np.max(np.abs(residual))
        normResidual = np.linalg.norm(residual, 2)
        if updatesize <= edelta and maxNormRHS <= eresidual:
            ret = True
        elif updatesize <= ea:
            ret = True # small update convergence
        elif normResidual <= np.finfo(float).eps:
            ret = True # RHS is too small and we consider it converged
        else:
            ret = False
            
    else:
        # We get here when there's no variable to be checked. This is because
        # there aren't variables of this type.  Eg. the circuit has no voltage
        # sources nor voltage defined elements. In this case, the actual check
        # is done only by current_convergence_check, voltage_convergence_check
        # always returns True.
        ret = True
        all_check_results = []
    all_check_results = [updatesize, maxNormRHS]
    return ret, all_check_results

def custom_convergence_check(x, dx, residual, er, ea, eresidual, debug=False):
    """Perform a custom convergence check

    **Parameters:**

    x : array-like
        The results to be checked.
    dx : array-like
        The last increment from a Newton-Rhapson iteration, solving
        ``F(x) = 0``.
    residual : array-like
        The remaining error, ie ``F(x) = residual``
    ea : float
        The value to be employed for the absolute error.
    er : float
        The value for the relative error to be employed.
    eresidual : float
        The maximum allowed error for the residual (left over error).
    debug : boolean, optional
        Whether extra information is needed for debug purposes. Defaults to
        ``False``.

    **Returns:**

    chk : boolean
        Whether the check was passed or not. ``True`` means 'convergence!'.
    rbn : ndarray
        The convergence check results by node, if ``debug`` was set to ``True``,
        else ``None``.
    """
    all_check_results = []
    maxNormRHS = np.max(np.abs(residual))
    maxNormUpdate = np.max(np.abs(dx))
    all_check_results = [maxNormUpdate, maxNormRHS]
    if not hasattr(x, 'shape'):
        x = np.array(x)
        dx = np.array(dx)
        residual = np.array(residual)    
    if x.shape[0]:
        if not debug:
            ret1 = np.allclose(x, x + dx, rtol=er, atol=ea) 
            ret2 = np.allclose(residual, np.zeros(residual.shape), rtol=0, atol=eresidual)
            # ret = ret1 and ret2  
            ret = [ret1, ret2]
        else:
            for i in range(x.shape[0]):
                if np.abs(dx[i]) < (er*np.abs(x[i]) + ea)*1.1 and \
                   np.abs(residual[i]) < eresidual:
                    all_check_results.append(True)
                else:
                    all_check_results.append(False)
                # uncomment the two lines below to break at the first non-convergent variable    
                if not all_check_results[-1]:
                    break

            ret = not (False in all_check_results)
            
    else:
        # We get here when there's no variable to be checked. This is because
        # there aren't variables of this type.  Eg. the circuit has no voltage
        # sources nor voltage defined elements. In this case, the actual check
        # is done only by current_convergence_check, voltage_convergence_check
        # always returns True.
        ret = True
        

    return ret, all_check_results

def check_step_and_points(step=None, points=None, period=None,
                          default_points=100):
    """Sets consistently the step size and the number of points

    The calculation is done according to the given period.

    **Parameters:**

    step : scalar, optional
        The discretization step.
    points : int, optional
        The number of points to be used in the discretization.
    period : scalar, optional
        The length of the interval to be discretized. Not setting
        this parameter will result in a ``ValueError``.
    default_points : int, optional
        The default number of points.

    **Returns:**

    (points, step) : tuple
        The adjusted number of points and step value.
    """

    if step is None and points is None:
        printing.print_warning("Neither step nor n. of points set. Using %d points." % default_points)
        points = default_points
    elif step is not None and points is not None:
        printing.print_warning("Both step and # of points set. Using step = %f." % step)
        points = None

    if points:
        step = float(period)/(points - 1)
    else:
        points = float(period)/step
        if points % 1 != 0:
            step = step + (step * (points % 1)) / int(points)
            points = int(float(period)/step)
            printing.print_warning("adapted step is %g" % (step,))
        else:
            points = int(points)
        # 0 - N where xN is in reality the first point of the second period!!
        points = points + 1

    return int(points), step

def check_circuit(circ):
    """Performs some easy sanity checks.

    Checks performed:

    * Has the circuit more than one node?
    * Has the circuit a connection to ground?
    * Has the circuit more than two elements?
    * Are there no two elements with the same ``part_id``?

    **Parameters:**

    circ : circuit instance
        The circuit to be checked.

    **Returns:**

    chk : boolean
        The logical ``and()`` of the answer to the above questions.
    msg : string
        A message describing the error, if any.
    """

    if circ.get_nodes_number() < 2:
        test_passed = False
        reason = "the circuit has less than two nodes."
    elif not 0 in circ.nodes_dict:
        test_passed = False
        reason = "the circuit has no ref. Quitting."
    elif len(circ) < 2:
        test_passed = False
        reason = "the circuit has less than two elements."
    elif circ.has_duplicate_elem():
        test_passed = False
        reason = "duplicate elements found (check the names!)"
    else:
        test_passed = True
        reason = ""

    return test_passed, reason


def check_ground_paths(mna, circ, reduced_mna=True, verbose=3):
    """Checks that every node has a DC path to ground

    The path to ground might be through non-linear elements.

    .. note::

        * This does not ensure that the circuit will have a DC solution.
        * A node without DC path to ground would be rescued (likely) by GMIN so
          (for the time being at least) we do *not* halt the execution.
        * Also, two series capacitors always fail this check (GMIN saves us)

    Bottom line: if there is no DC path to ground, there is probably a
    mistake in the netlist. Print a warning.

    **Returns:**

    chk : boolean
        A boolean set to true if there is a DC path to ground from all nodes
        in the circuit.
    """
    test_passed = True
    if reduced_mna:
        # reduced_correction
        r_c = 1
    else:
        r_c = 0
    to_be_checked_for_nonlinear_paths = []
    for node in iter(circ.nodes_dict.keys()):
        if node == 0:
            continue
            # ground
        if type(node) != int:
            # an ext handle
            continue
        if mna[node - r_c, node - r_c] == 0 and \
           not sp.sparse.csr_matrix(mna[node - r_c, circ.get_nodes_number() - r_c:]).toarray().any():
            to_be_checked_for_nonlinear_paths.append(node)
    for node in to_be_checked_for_nonlinear_paths:
        node_is_nl_op = False
        for elem in circ:
            if not elem.is_nonlinear:
                continue
            ops = elem.get_output_ports()
            for op in ops:
                if op.count(node):
                    node_is_nl_op = True
        if not node_is_nl_op:
            if verbose:
                printing.print_warning(
                    "No path to ground from node " + circ.nodes_dict[node])
            test_passed = False
    return test_passed

def memoize(f):
    """Memoization decorator

    **Parameters:**

    f : function
        The function to apply memoization to.

    **Returns:**

    fm : function
        The function with added memoization.

    **Implementation:**

    Originally from `this post
    <https://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c4>`_,
    it has been modified to provide a cache of size ``options.cache_len``.

    .. note::

        The size of the cache is per model instance and per function. If you
        have one model, shared by several elements, you probably prefer to have
        a big cache.

    """
    class memodict(collections.OrderedDict):
        __slots__ = ()
        @wraps(f)
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)
        def __missing__(self, key):
            ret = self[key] = f(*key)
            # set options.cache_len to None to disable any size limit.
            if options.cache_len is not None and len(self) > options.cache_len:
                self.popitem() #FIFO pop
            return ret
    return memodict().__getitem__

def GMRES_wrapper(A, b, x0, tol, maxiter):
    niter = 0
    
    def callback(rk):
        nonlocal niter
        niter += 1
        
    dx, info = sp.sparse.linalg.gmres(A,b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
    
    return dx, info, niter

#convert bsimMatrix to sp.sparse.csc_matrix
def bsim2cscMatrix(BM, sz):
    
    data = BM.Pr[:BM.NNZ]
    indices = BM.Ir[:BM.NNZ]
    indptr = BM.Jc[:BM.M+1]
    indptr = np.hstack((indptr, np.broadcast_to(indptr[-1], (sz[0] - BM.M))))
    ret = sp.sparse.csc_matrix((data, indices, indptr), shape=sz)
    
    return ret

#load LTspice raw data as initial guess
def load_LTSpice_raw_file(filepath):
    import ltspice
    # filepath = 'clock_ahkab.op.raw'
    lt = ltspice.Ltspice(filepath)
    lt.parse() # Data loading sequence. It may take few minutes.
    varname = lt.getVariableNames()
    icdict = {}
    for i in range(len(varname)):
        label = varname[i]
        value = lt.getData(label)[0]
        icdict[label] = value
        
    return icdict

# detect floating caps by connected components searching
# removing it by adding options.cmin to one of the nodes
def remove_floating_cap(C, circ):
    from collections import defaultdict
    nv = circ.nv - 1 # subtract gnd node
    Cv = C[:nv, :][:, :nv]
    rowSum = np.sum(Cv, 1) # sum over each row
    row_gnd = sp.sparse.csc_matrix(np.abs(rowSum) > np.finfo(float).eps) # if the sum is not zero, that node is good becoz it has a grouned cap 
    Cv1 = sp.sparse.bmat([[sp.sparse.csr_matrix([1]), row_gnd.T],[row_gnd, Cv]], format='csc') # include the contribution from gnd node
    adjMat = Cv1.astype(bool).tocsr()
    # adjMat = sp.sparse.tril(adjMat, -1).tocsr()
    ncomp, label = sp.sparse.csgraph.connected_components(adjMat, directed=False)
    dlabel = defaultdict(list)
    
    ncc = np.zeros(ncomp)
    ncc_tofix = 0
    for i, x in enumerate(label): 
        dlabel[x].append(i - 1) 
        ncc[x] += 1
    
    for x in range(1,ncomp): # starting from 1 not to count the connected components involving gnd
        if ncc[x] > 1:
            ncc_tofix += 1
            idx = dlabel[x][0] # pick the 1st node from the group
            C[idx, idx] += options.cmin
            tmp = 1
           
    return C

def remove_vsource(C, G, circ):
    
    for elem in circ:
        if isinstance(elem, devices.VSource):
            idx = elem.index
            C[:, idx] = G[:, idx]
            C[idx, :] = G[idx, :]
            G[:, idx] = 0
            G[idx, :] = 0
           
    return C, G


