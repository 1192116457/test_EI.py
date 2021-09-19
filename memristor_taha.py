# -*- coding: iso-8859-1 -*-
# memristor_Yakopcic.py
# Yakopcic memristor model
# Copyright 2019 Alex Chen

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
This module contains a memristor element and its model class.

.. image:: images/elem/memristor.svg

"""

#
#         |\|
#  n1 o---| ]---o n2
#         |/|
#

from __future__ import (unicode_literals, absolute_import,
                        division, print_function)

#import numpy as np
import autograd.numpy as np
from autograd import grad

from scipy.optimize import newton

from . import constants
from . import utilities
from . import options

damping_factor = 4.

class memristor(object):
    """A Yakopcic memristor element.

    **Parameters:**

    n1, n2 : string
        The memristor anode and cathode.
    model : model instance
        The memristor model providing the mathemathical modeling.
    ic : float
        The memristor initial voltage condition for transient analysis
        (ie :math:`V_D = V(n_1) - V(n_2)` at :math:`t = 0`).
    off : bool
         Whether the memristor should be initially assumed to be off when
         computing an OP.

    The other are the physical parameters reported in the following table:

    +---------------+-------------------+-----------------------------------+
    | *Parameter*   | *Default value*   | *Description*                     |
    +===============+===================+===================================+
    | AREA          | 1.0               | Area multiplier                   |
    +---------------+-------------------+-----------------------------------+
    | T             | circuit temp      | Operating temperature             |
    +---------------+-------------------+-----------------------------------+

    """

    def __init__(self, part_id, n1, n2, model, AREA=None, T=None, ic=None, off=False):
        self.part_id = part_id
        self.is_nonlinear = True
        self.is_symbolic = True
        self.dc_guess = [0.425]
        class _dev_class(object):
            pass
        self.device = _dev_class()
        self.device.AREA = AREA if AREA is not None else 1.0
        self.device.T = T
        self.device.last_vd = .425
        self.n1 = n1
        self.n2 = n2
        self.ports = ((self.n1, self.n2),)
        self.model = model
        if self.device.T is None:
            self.device.T = constants.T

        if ic is not None:  # fixme
            print("(W): ic support in diodes is very experimental.")
            self.dc_guess = ic
        self.ic = ic
        self.off = off
        if self.off:
            if self.ic is None:
                self.ic = 0
            else:
                print("(W): IC statement in diodes takes precedence over OFF.")
                print("(W): If you are performing a transient simulation with uic=2,")
                print("(W): you may want to check the initial value.")

    def _get_T(self):
        return self.device.T

    def set_temperature(self, T):
        """Set the operating temperature IN KELVIN degrees
        """
        # this automatically makes the memoization cache obsolete. self.device
        # is in fact passed as one of the arguments, hashed and stored:
        # if it changes, the old cache will return misses.
        self.device.T = T

    def __str__(self):
        rep = "%s area=%g T=%g" % (
            self.model.name, self.device.AREA, self._get_T())
        if self.ic is not None:
            rep = rep + " ic=" + str(self.ic)
        elif self.off:
            rep += " off"
        return rep

    def get_output_ports(self):
        return self.ports

    def get_drive_ports(self, op):
        if not op == 0:
            raise ValueError('Diode %s has no output port %d' %
                             (self.part_id, op))
        return self.ports

    def istamp(self, ports_v, time=0, reduced=True):
        """Get the current matrix

        A matrix corresponding to the current flowing in the element
        with the voltages applied as specified in the ``ports_v`` vector.

        **Parameters:**

        ports_v : list
            A list in the form: [voltage_across_port0, voltage_across_port1, ...]
        time: float
            the simulation time at which the evaluation is performed.
            It has no effect here. Set it to None during DC analysis.

        """
#        v = ports_v[0]
        i = self.model.get_i(self.model, ports_v, self.device)
        istamp = np.array((i, -i), dtype=np.float64)
        indices = ((self.n1 - 1*reduced, self.n2 - 1*reduced), (0, 0))
        if reduced:
            delete_i = [pos for pos, ix in enumerate(indices[0]) if ix == -1]
            istamp = np.delete(istamp, delete_i, axis=0)
            indices = tuple(zip(*[(ix, j) for ix, j in zip(*indices) if ix != -1]))
        return indices, istamp

    def i(self, op_index, ports_v, time=0):  # with gmin added
        v = ports_v[0]
        i = self.model.get_i(self.model, v, self.device)
        return i

    def gstamp(self, ports_v, time=0, reduced=True):
        """Returns the differential (trans)conductance wrt the port specified by port_index
        when the element has the voltages specified in ports_v across its ports,
        at (simulation) time.

        ports_v: a list in the form: [voltage_across_port0, voltage_across_port1, ...]
        port_index: an integer, 0 <= port_index < len(self.get_ports())
        time: the simulation time at which the evaluation is performed. Set it to
        None during DC analysis.
        """
        indices = ([self.n1 - 1]*2 + [self.n2 - 1]*2,
                   [self.n1 - 1, self.n2 - 1]*2)
        gm = self.model.get_gm(self.model, 0, utilities.tuplinator(ports_v), 0, self.device)
        if gm == 0:
            gm = options.gmin*2
        stamp = np.array(((gm, -gm),
                          (-gm, gm)), dtype=np.float64)
        if reduced:
            zap_rc = [pos for pos, i in enumerate(indices[1][:2]) if i == -1]
            stamp = np.delete(stamp, zap_rc, axis=0)
            stamp = np.delete(stamp, zap_rc, axis=1)
            indices = tuple(zip(*[(i, y) for i, y in zip(*indices) if (i != -1 and y != -1)]))
            stamp_flat = stamp.reshape(-1)
            stamp_folded = []
            indices_folded = []
            for ix, it in enumerate([(i, y) for i, y in zip(*indices)]):
                if it not in indices_folded:
                    indices_folded.append(it)
                    stamp_folded.append(stamp_flat[ix])
                else:
                    w = indices_folded.index(it)
                    stamp_folded[w] += stamp_flat[ix]
            indices = tuple(zip(*indices_folded))
            stamp = np.array(stamp_folded)
        return indices, stamp

    def g(self, op_index, ports_v, X, port_index, time=0):
        if not port_index == 0:
            raise Exception("Attepted to evaluate a memristor's gm on an unknown port.")
        gm = self.model.get_gm(self.model, op_index, utilities.tuplinator(ports_v), port_index, self.device)
        return gm

    def get_op_info(self, ports_v_v):
        """Information regarding the Operating Point (OP)

        **Parameters:**

        ports_v : list of lists
            The parameter is to be set to ``[[v]]``, where ``v`` is the voltage
            applied to the memristor terminals.

        **Returns:**

        op_keys : list of strings
            The labels corresponding to the numeric values in ``op_info``.
        op_info : list of floats
            The values corresponding to ``op_keys``.
        """
        vn1n2 = float(ports_v_v[0][0])
        idiode = self.i(0, (vn1n2,))
        gmdiode = self.g(0, (vn1n2,), 0)
        op_keys = ["Part ID", "V(n1-n2) [V]", "I(n1-n2) [A]", "P [W]",
                "gm [A/V]", u"T [\u00b0K]"]
        op_info = [self.part_id.upper(), vn1n2, idiode, vn1n2*idiode, gmdiode,
                   self._get_T()]
        return op_keys, op_info

    def get_netlist_elem_line(self, nodes_dict):
        ext_n1, ext_n2 = nodes_dict[self.n1], nodes_dict[self.n2]
        ret = "%s %s %s %s" % (self.part_id, ext_n1, ext_n2, self.model.name)
        # append the optional part:
        # [<AREA=float> <T=float> <IC=float> <OFF=boolean>]
        ret += " AREA=%g" % self.device.AREA
        if self.device.T is not None:
            ret += " T=%g" % self.device.T
        if self.ic is not None:
            ret += " IC=%g" % self.ic
        if self.off:
            ret += " OFF=1"
        return ret

A1_DEFAULT = 0.00036
A2_DEFAULT = 0.0004
B_DEFAULT = 2.2
VP_DEFAULT = 0.3
VN_DEFAULT = 0.3
AP_DEFAULT = 3000000.0
AN_DEFAULT = 3000000.0
XP_DEFAULT = 0.2
XN_DEFAULT = 0.2
ALPHAP_DEFAULT = 1
ALPHAN_DEFAULT = 5
ETA_DEFAULT = 1
T_DEFAULT = utilities.Celsius2Kelvin(26.85)
AREA_DEFAULT = 1.0


class Yakopcic_model(object):
    """A memristor model implementing the `Yakopcic memristor equation

    Currently the capacitance modeling part is missing.

    The principal parameters are:

    +---------------+-------------------+-----------------------------------+
    | *Parameter*   | *Default value*   | *Description*                     |
    +===============+===================+===================================+
    | IS            | 1e-14 A           | Specific current                  |
    +---------------+-------------------+-----------------------------------+
    | N             | 1.0               | Emission coefficient              |
    +---------------+-------------------+-----------------------------------+
    | ISR           | 0.0 A             | Recombination current             |
    +---------------+-------------------+-----------------------------------+
    | NR            | 2.0               | Recombination coefficient         |
    +---------------+-------------------+-----------------------------------+
    | RS            | 0.0 ohm           | Series resistance per unit area   |
    +---------------+-------------------+-----------------------------------+

    please refer to a textbook description of the Shockley memristor equation
    or to the source file ``memristor.py`` file for the other parameters.

    """
    def __init__(self, name, A1=None, A2=None, B=None, VP=None, VN=None,
                 AP=None, AN=None, XP=None, XN=None, ALPHAP=None, ALPHAN=None,
                 ETA=None):
        self.name = name
        self.A1 = float(A1) if A1 is not None else A1_DEFAULT
        self.A2 = float(A2) if A2 is not None else A2_DEFAULT
        self.B = float(B) if B is not None else B_DEFAULT
        self.VP = float(VP) if VP is not None else VP_DEFAULT
        self.VN = float(VN) if VN is not None else VN_DEFAULT
        self.AP = float(AP) if AP is not None else AP_DEFAULT
        self.AN = float(AN) if AN is not None else AN_DEFAULT
        self.XP = float(XP) if XP is not None else XP_DEFAULT
        self.XN = float(XN) if XN is not None else XN_DEFAULT
        self.ALPHAP = float(ALPHAP) if ALPHAP is not None else ALPHAP_DEFAULT
        self.ALPHAN = float(ALPHAN) if ALPHAN is not None else ALPHAN_DEFAULT
        self.ETA = float(ETA) if ETA is not None else ETA_DEFAULT
        self.T = T_DEFAULT
        self.last_vd = None
        self.i_grad = grad(self._get_i, (0, 1))
        self.x_grad = grad(self._get_x, (0, 1))
#        self.VT = constants.Vth(self.T)

    def print_model(self):
        strm = ".model memristor_Yakopcic %s A1=%g A2=%g B=%g VP=%g VN=%g AP=%g AN=%g " + \
               "XP=%g XN=%g ALPHAP=%g ALPHAN=%g ETA=%g  "
        print(strm % (self.name, self.A1, self.A2, self.B, self.VP, self.VN,
                      self.AP, self.AN, self.XP, self.XN, self.ALPHAP, self.ALPHAN,
                      self.ETA))

    @utilities.memoize
    def get_i(self, vext, dev):
        if dev.T != self.T:
            self.set_temperature(dev.T)
        if not self.RS:
            i = self._get_i(vext) * dev.AREA
            dev.last_vd = vext
        else:
            vd = dev.last_vd if dev.last_vd is not None else 10*self.VT
            vd = newton(self._obj_irs, vd, fprime=self._obj_irs_prime,
                        args=(vext, dev), tol=options.vea, maxiter=500)
            i = self._get_i(vext-vd)
            dev.last_vd = vd
        return i

    def _obj_irs(self, x, vext, dev):
        # obj fn for newton
        
        return x/self.RS-self._get_i(vext-x)*dev.AREA

    def _obj_irs_prime(self, x, vext, dev):
        # obj fn derivative for newton
        # first term
        ret = 1./self.RS
        # disable RS
        RSSAVE = self.RS
        self.RS = 0
        # second term
        ret += self.get_gm(self, 0, (vext-x,), 0, dev)
        # renable RS
        self.RS = RSSAVE
        return ret

    def _safe_exp(self, x):
        return np.exp(x) if x < 70 else np.exp(70) + 10 * x

    def G(self, V):   
#        V = V1 - V2
        if V <= self.VP:
            if V >= -self.VN:
                fval = 0.0
            else:
                fval = -self.AN * (np.exp(-V) - np.exp(self.VN) )
        else:
            fval = self.AP * (np.exp(V) - np.exp(self.VP) )
            
        return fval 
    
    def WP(self, X):
        return 1 + (self.XP - X) / (1.0 - self.XP)

    def WN(self, X):
        return X / (1.0 - self.XN)

    def F_x_equation(self, V, X):
        if self.ETA * V > 0:
          if X > self.XP:
            WPval = self.WP(X)
            fval = WPval * np.exp(-self.ALPHAP * (X - self.XP ))  # equation 3
          else:
            fval = 0.0
        else:
           if X <= (1-self.XN):
             WNval = self.WN(X)
             fval = WNval * np.exp(self.ALPHAN * (X + self.XN -1))  # equation 4
           else:
             fval = 1.0
             
        return fval
    
    def _get_x(self, V, X):
        Gval = self.G(V)
        FXval = self.F_x_equation(V, X)
        fval = self.ETA * Gval * FXval
        
        return fval

    def _get_i(self, V, X):
        if( V >= 0.0 ):  
            fval = self.A1 * X * np.sinh(self.B * (V))
        else:
            fval = self.A2 * X * np.sinh(self.B * (V))
      
        return fval
#        i = self.IS * (self._safe_exp(v/(self.N * self.VT)) - 1) \
#            + self.ISR * (self._safe_exp(v/(self.NR * self.VT)) - 1)
#            
#        return i

    @utilities.memoize
    def get_gm(self, op_index, ports_v, port_index, dev):
        if dev.T != self.T:
            self.set_temperature(dev.T)
        V = ports_v[0]
        X = ports_v[1]
        gm = self.i_grad(V, X)
        xgm = self.x_grad(V, X)
        gm = gm.append(xgm)
#        gm = self.IS / (self.N * self.VT) *\
#            self._safe_exp(ports_v[0] / (self.N * self.VT)) +\
#            self.ISR / (self.NR * self.VT) *\
#            self._safe_exp(ports_v[0] / (self.NR * self.VT))
#        if self.RS != 0.0:
#            gm = 1. / (self.RS + 1. / (gm + 1e-3*options.gmin))
        return dev.AREA * gm

    def __str__(self):
        pass

#    def set_temperature(self, T):
#        T = float(T)
#        self.EG = constants.si.Eg(T)
#        self.IS = self.IS*(T/self.T)**(self.XTI/self.N)* \
#                  np.exp(-constants.e*constants.si.Eg(300)/\
#                         (self.N*constants.k*T)*
#                         (1 - T/self.T))
#        self.BV = self.BV - self.TBV*(T - self.T)
#        self.RS = self.RS*(1 + self.TRS*(T - self.T))
#        self.T = T

