# -*- coding: iso-8859-1 -*-
# netlist_parser.py
# Netlist parser module
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

"""Parse spice-like netlist files and generate circuits instances.

The syntax is explained in :doc:`help/Netlist-Syntax` and it's based on [#1]_
whenever possible.

.. [#1] http://newton.ex.ac.uk/teaching/CDHW/Electronics2/userguide/

Introduction
------------

This module has one main circuit that is expected to be useful to the end user:
:func:`parse_circuit`, which encapsulates parsing a netlist file and returns the
circuit, the simulation objects and the post-processing directives (such as
plotting instructions).

Additionally, the module provides utility functions related to parsing, among
which the end user may be interested in the :func:`convert` function, which allows
converting from SPICE-like representations of floats, booleans and strings to
their Python representations.

The last type of functions in the module are utility functions to go through the
netlist files and remove comments.

Except for the aforementioned functions, the rest seem to be more suitable for
developers than end users.

Overview
--------

Function for parsing
====================

.. autosummary::
    parse_circuit
    main_netlist_parser
    parse_elem_resistor
    parse_elem_capacitor
    parse_elem_inductor
    parse_elem_inductor_coupling
    parse_elem_vsource
    parse_elem_isource
    parse_elem_diode
    parse_elem_mos
    parse_elem_vcvs
    parse_elem_vccs
    parse_elem_cccs
    parse_elem_ccvs
    parse_elem_switch
    parse_elem_user_defined
    parse_models
    parse_time_function
    parse_postproc
    parse_ics
    parse_analysis
    parse_single_analysis
    parse_temp_directive
    parse_param_value_from_string
    parse_ic_directive
    parse_sub_declaration
    parse_sub_instance
    parse_include_directive

Utility functions for conversions
=================================

.. autosummary::
    convert
    convert_units
    convert_boolean

Utility functions for file/txt handling
=======================================

.. autosummary::
    join_lines
    is_valid_value_param_string
    get_next_file_and_close_current


Module reference
----------------
"""

from __future__ import (unicode_literals, absolute_import, division, print_function)

import sys
import imp
import math
import copy
import os
import time
import json
import re
# from numba import jit
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
# import ctypes
# dll_file = "BSIM3_Dll_new.dll"
# bsim = ctypes.cdll.LoadLibrary(dll_file)

import circuit
import dc_analysis
import devices
import diode
import mosq
import ekv
import memristor
import switch
import printing
import utilities
import plotting
import options
import bsim3

# analyses syntax
from dc_analysis import specs as dc_spec
from ac import specs as ac_spec
from transient import specs as tran_spec
from pss import specs as pss_spec
from py3compat import StringIO
from pz import specs as pz_specs
from symbolic import specs as symbolic_spec
from time_functions import time_fun_specs
from time_functions import sin, pulse, exp, sffm, am, pwl

from fourier import specs as fft_specs

specs = {}
for i in dc_spec, ac_spec, tran_spec, pss_spec, symbolic_spec, pz_specs:
    specs.update(i)

time_functions = {}
for i in sin, pulse, exp, sffm, am, pwl:
    time_functions.update({i.__name__:i})
   
unit_conversion_table = {
        'T':1e12, 'G':1e9, 'MEG':1e6, 'K':1e3, 
        'M':1e-3, 'MS':1e-3, 'MA':1e-3,'U':1e-6, 'US':1e-6, 'UH':1e-6, 'UF':1e-6, 'N':1e-9, 'NS':1e-9, 'NH':1e-9, 'NF':1e-9, 
        'P':1e-12, 'PS':1e-12, 'PH':1e-12, 'PF':1e-12, 'F':1e-15, 'FS':1e-15, 'MIL':25.4e-6, 'FF':1e-15}    


    
def parse_circuit(filename, read_netlist_from_stdin=False, verbose=3):
    """Parse a SPICE-like netlist

    Directives are collected in lists and returned too, except for
    subcircuits, those are added to circuit.subckts_dict.

    **Returns:**

    (circuit_instance, analyses, plotting directives)
    """
    # Lots of differences with spice's syntax:
    # Support for alphanumeric node names, but the ref has to be 0. always
    # .end is not required, but if is used anything following it is ignored
    # many others, see doc.
    printing.print_info_line(("Start parsing netlist: {}".format(filename), 3), verbose)
    circ = circuit.Circuit(title="", filename=filename)

    if not read_netlist_from_stdin:
        ffile = open(filename, "r")
    else:
        buf = ""
        for aline in sys.stdin:
            buf += aline + "\n"
        ffile = StringIO(buf)

    file_list = [(ffile, "unknown", not read_netlist_from_stdin)]
    netlist_wd = os.path.split(filename)[0]
    file_index = 0
    directives = []
    parameters = []
    option_directives = []
    model_directives = []
    postproc = []
    subckts_list_temp = []
    netlist_lines = []
    current_subckt_temp = []
    within_subckt = False
    line_n = 0
    t = time.time()
    try:
        while ffile is not None:
            while 1:
                line = ffile.readline()
                if len(line) == 0:
                    break  # check for EOF
                line_n = line_n + 1
#                if line_n > 300000:
#                    print("parse time: {} s".format(time.time() - t))
#                    sys.exit()
                line = line.strip().lower()
                if line_n == 1:
                    # the first line is always the title
                    circ.title = line
                    continue
                elif len(line) == 0:
                    continue  # empty line is really empty after strip()
                line = join_lines(ffile, line) # let's assume we dont' have line connectors +
                if line[0] == "*":  # comments start with *
                    continue

                # directives are grouped together and evaluated after
                # we have the whole circuit.
                # subcircuits are grouped too, but processed first
                if line[0] == ".":
                    line_elements = line.split()
                    if line_elements[0] == '.subckt':
                        if within_subckt:
                            raise NetlistParseError("nested subcircuit declaration detected")
                        current_subckt_temp = current_subckt_temp + \
                            [(line, line_n)]
                        within_subckt = True
                    elif line_elements[0] == '.ends':
                        if not within_subckt:
                            raise NetlistParseError(".ENDS outside of .subckt")
                        within_subckt = False
                        subckts_list_temp.append(current_subckt_temp)
                        current_subckt_temp = []
                    elif line_elements[0] == '.include':
                        file_list.append(
                            parse_include_directive(line, netlist_wd))
                    elif line_elements[0] in {'.options', '.option'}:
                        option_directives.append((line, line_n))    
                    elif line_elements[0] == ".end":
                        break
                    elif line_elements[0] == ".plot":
                        postproc.append((line, line_n))
                    elif line_elements[0] in {".print", ".probe"}:
                        postproc.append((line, line_n))
                    elif line_elements[0] == ".save":
                        postproc.append((line, line_n))
                    elif line_elements[0] == '.four':
                        postproc.append((line, line_n))
                    elif line_elements[0] == '.fft':
                        postproc.append((line, line_n))
                    elif line_elements[0] == ".model":
                        model_directives.append((line, line_n))
                    elif line_elements[0] == ".includemodel":
                        model_directives.append((line, -1))
                        file_list.append(
                            parse_include_directive(line, netlist_wd))
                    elif line_elements[0] == ".param":
                        if within_subckt:
                            current_subckt_temp  = current_subckt_temp + \
                                [(line, line_n)]
                        else:
                            parameters.append((line, line_n)) # do nothing yet    
                    elif line_elements[0] == ".ic":
                        if within_subckt:
                            # current_subckt_temp = current_subckt_temp + \
                            # [(line, line_n)]
                            pass # do not handle ic within subckt definition due to port name mismatch
                        else:
                            directives.append((line, line_n)) # process as normal      
                    elif line_elements[0] == ".measure": # omit .measure derivative
                        pass 
                    elif line_elements[0] == ".preprocess": # omit .preprocess derivative
                        pass 
                    else:
                        directives.append((line, line_n))
                    continue

                if within_subckt:
                    current_subckt_temp  = current_subckt_temp + \
                        [(line, line_n)]
                else:
                    netlist_lines.append((line, line_n))
#                    pass
            if within_subckt:
                raise NetlistParseError(".ends not found")

            file_index = file_index + 1
            ffile = get_next_file_and_close_current(file_list, file_index)
            # print file_list

    except NetlistParseError as npe:
        (msg,) = npe.args
        if len(msg):
            printing.print_general_error(msg)
        printing.print_parse_error(line_n, line)
        # if not read_netlist_from_stdin:
            # ffile.close()
        raise NetlistParseError(msg)
    print("Readline time: {} s".format(time.time() - t))   
#    sys.exit(12)
    # if not read_netlist_from_stdin:
        # ffile.close()
    parse_options(option_directives)    
    models = parse_models(model_directives)

    # now we parse the subcircuits, we want a circuit.subckt object that holds the netlist code,
    # the nodes and the subckt name in a handy way.
    # We will create all the elements in the subckt every time it is
    # instantiated in the netlist file.
    subckts_dict = {}
    for subckt_temp in subckts_list_temp:
        subckt_obj = parse_sub_declaration(subckt_temp)
        if subckt_obj.name not in subckts_dict:
            subckts_dict.update({subckt_obj.name: subckt_obj})
        else:
            pass # if the subckt definition is included more than one time, keep only the first one (Alex)
#            raise NetlistParseError("subckt " + \
#                subckt_obj.name + " has been redefined")
    t = time.time()        
    circ += main_netlist_parser(circ, netlist_lines, subckts_dict, models)
    print("Element parse time: {} s".format(time.time() - t))
    circ.models = models
    circ.isnonlinear = circ.is_nonlinear()
    # assign vde_index and mem_index here
    vde_index = 0
    mem_index = 0  
    nidv, nidc = 0, 0 #numbers of independent voltage and current sources
    for elem in circ:
        if hasattr(elem, 'is_voltage_defined') and elem.is_voltage_defined:
            elem.vde_index = vde_index
            vde_index += 1
        elif hasattr(elem, 'is_memristor') and elem.is_memristor:
            elem.mem_index = mem_index
            mem_index += 1
        else:
            pass
        
        # calculating numbers of independent voltage and current sources
        if isinstance(elem, devices.VSource):
            nidv += 1
        elif isinstance(elem, devices.ISource):
            nidc += 1
        else:
            pass
            
             
    circ.nv = circ.get_nodes_number()
    circ.ni = vde_index
    circ.nx = mem_index
    circ.nidv = nidv
    circ.nidc = nidc
    circ.subckts_dict = subckts_dict
    # extract bsim3 device instances if bsim3 model is used
    if options.useBsim3:
        # bsim3list = bsim3.bsim3_instances(ptr_model=models['ptr_model'])
        plist, nlist = [], []
        pmodelname, nmodelname = "", ""
        mp, np, mn, nn = 0, 10, 0, 10
        max_node_num = circ.nv - 1
        temp = 27.0
        for elem in circ:
            if isinstance(elem, bsim3.bsim3_device):
                device_card = [elem.n1, elem.ng, elem.n2, elem.nb,
                               1, elem.device.W, elem.device.L,
                               1, 1, 1]
                if elem.bsim3_model.NPMOS == 1:
                    nlist += device_card
                    mn += 1
                    if len(nmodelname) == 0:
                        nmodelname = elem.bsim3_model.name 
                    elif nmodelname != elem.bsim3_model.name:
                        raise ValueError(" only one n-type model is allowed")
                elif elem.bsim3_model.NPMOS == -1:
                    plist += device_card
                    mp += 1
                    if len(pmodelname) == 0:
                        pmodelname = elem.bsim3_model.name
                    elif pmodelname != elem.bsim3_model.name:
                        raise ValueError(" only one p-type model is allowed")    
                else:
                    raise ValueError(" neither n nor p type mosfets")
                    
        ptr_plist = (ctypes.c_double * len(plist))(*plist)
        ptr_nlist = (ctypes.c_double * len(nlist))(*nlist)
        pmodelname = ctypes.c_char_p(pmodelname.encode('utf-8'))
        nmodelname = ctypes.c_char_p(nmodelname.encode('utf-8'))
        bsim.LoadBSIM3Instance.argtypes = [ctypes.POINTER(ctypes.c_double), 
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.c_char_p, ctypes.c_char_p,
                               ctypes.c_double, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int,
                               ctypes.c_double]
        bsim.LoadBSIM3Instance.restype =ctypes.c_double
        circ.bsim3list = bsim.LoadBSIM3Instance(ptr_plist, ptr_nlist,
                                pmodelname, nmodelname,
                                models['ptr_model'],
                                mp, np, mn, nn,
                                max_node_num, temp,
                                )
        
#    sys.exit(12)
    return (circ, directives, postproc)

def parse_directive_and_postprocess(filename, circ, read_netlist_from_stdin=False):
    
    if not read_netlist_from_stdin:
        ffile = open(filename, "r")
    else:
        buf = ""
        for aline in sys.stdin:
            buf += aline + "\n"
        ffile = StringIO(buf)

    file_list = [(ffile, "unknown", not read_netlist_from_stdin)]
    netlist_wd = os.path.split(filename)[0]
    file_index = 0
    directives = []
    option_directives = []
    model_directives = []
    postproc = []
    subckts_list_temp = []
#    netlist_lines = []
    current_subckt_temp = []
    within_subckt = False
    line_n = 0
    offset = len(circ)
            
    t_parse = time.time()   
    try:
        while ffile is not None:
            for line_n in range(offset):
                next(ffile)
            line_n += 1    
            while True:
                line = ffile.readline()
                if len(line) == 0:
                    break  # check for EOF
                line_n = line_n + 1
                line = line.strip().lower()
                if line_n == 1:
                    # the first line is always the title
                    circ.title = line
                    continue
                elif len(line) == 0:
                    continue  # empty line is really empty after strip()
#                line = join_lines(ffile, line)
                if line[0] == "*":  # comments start with *
                    continue

                # directives are grouped together and evaluated after
                # we have the whole circuit.
                # subcircuits are grouped too, but processed first
                if line[0] == ".":
                    line_elements = line.split()
                    if line_elements[0] == '.subckt':
                        if within_subckt:
                            raise NetlistParseError("nested subcircuit declaration detected")
                        current_subckt_temp = current_subckt_temp + \
                            [(line, line_n)]
                        within_subckt = True
                    elif line_elements[0] == '.ends':
                        if not within_subckt:
                            raise NetlistParseError(".ENDS outside of .subckt")
                        within_subckt = False
                        subckts_list_temp.append(current_subckt_temp)
                        current_subckt_temp = []
                    elif line_elements[0] == '.include':
                        file_list.append(
                            parse_include_directive(line, netlist_wd))
                    elif line_elements[0] in {'.options', '.option'}:
                        option_directives.append((line, line_n))    
                    elif line_elements[0] == ".end":
                        break
                    elif line_elements[0] == ".plot":
                        postproc.append((line, line_n))
                    elif line_elements[0] in {".print", ".probe"}:
                        postproc.append((line, line_n))
                    elif line_elements[0] == ".save":
                        postproc.append((line, line_n))
                    elif line_elements[0] == '.four':
                        postproc.append((line, line_n))
                    elif line_elements[0] == '.fft':
                        postproc.append((line, line_n))
                    elif line_elements[0] == ".model":
                        model_directives.append((line, line_n))
                    elif line_elements[0] == ".includemodel":
                        model_directives.append((line, -1))
                        file_list.append(
                            parse_include_directive(line, netlist_wd))
                    else:
                        directives.append((line, line_n))
                        
                    continue

                if within_subckt:
                    current_subckt_temp  = current_subckt_temp + \
                        [(line, line_n)]
                else:
#                    netlist_lines = netlist_lines + [(line, line_n)]
                    pass
                    
                
                
            if within_subckt:
                raise NetlistParseError(".ends not found")

            file_index = file_index + 1
            ffile = get_next_file_and_close_current(file_list, file_index)
            # print file_list

    except NetlistParseError as npe:
        (msg,) = npe.args
        if len(msg):
            printing.print_general_error(msg)
        printing.print_parse_error(line_n, line)
        # if not read_netlist_from_stdin:
            # ffile.close()
        raise NetlistParseError(msg)
        
    print("Parse only directives and postprocessing. Time used: {} s".format(time.time() - t_parse))    
    # if not read_netlist_from_stdin:
        # ffile.close()
    parse_options(option_directives)    
    models = parse_models(model_directives)    
    circ.models = models
    circ.isnonliear = circ.is_nonlinear()
    
    return (circ, directives, postproc)

def main_netlist_parser(circ, netlist_lines, subckts_dict, models):
    elements = []
    parse_function = {
        'c': lambda line: parse_elem_capacitor(line, circ),
        'd': lambda line: parse_elem_diode(line, circ, models),
        'e': lambda line: parse_elem_vcvs(line, circ),
        'f': lambda line: parse_elem_cccs(line, circ),
        'g': lambda line: parse_elem_vccs(line, circ),
        'h': lambda line: parse_elem_ccvs(line, circ),
        'i': lambda line: parse_elem_isource(line, circ),
        'k': lambda line: parse_elem_inductor_coupling(line, circ, elements),
        'l': lambda line: parse_elem_inductor(line, circ),
        'm': lambda line: parse_elem_mos(line, circ, models),
        'r': lambda line: parse_elem_resistor(line, circ),
        's': lambda line: parse_elem_switch(line, circ, models),
        'v': lambda line: parse_elem_vsource(line, circ),
        'x': lambda line: parse_sub_instance(line, circ, subckts_dict, models),
        'y': lambda line: parse_elem_memristor(line, circ, models),
        'b': lambda line: parse_elem_bsource(line, circ),
        'n': lambda line: parse_elem_mos(line, circ, models),
        'p': lambda line: parse_elem_mos(line, circ, models),
#        'y': lambda line: parse_elem_user_defined(line, circ)
    }
    try:
#        # determine the global starting vlaue of vde_index and mem_index
#        # needed when main_netlist_parser is called multiple times from subckt
#        vde_index_vec = [elem.vde_index for elem in circ if hasattr(elem, 'vde_index')]
#        vde_index = 0 if not vde_index_vec else max(vde_index_vec) + 1
#        mem_index_vec = [elem.mem_index for elem in circ if hasattr(elem, 'mem_index')]
#        mem_index = 0 if not mem_index_vec else max(mem_index_vec) + 1
        for line, line_n in netlist_lines:
            # elements: detect the element type and call the
            # appropriate parsing function
            # we always use normal convention V opposite to I
            # n1 is +, n2 is -, current flows from + to -
            try:
                elem = parse_function[line[0]](line)
                # no need to check is_voltage_defined as it will be set at element initialization (Alex)
#                if circuit.is_elem_voltage_defined(elem[0]): #check if the elem is voltage-defined. If so, also calculate and store its vde_index 
#                    elem[0].is_voltage_defined = True
##                    elem[0].vde_index = vde_index
##                    vde_index += 1
#                elif line[0] == 'y':
##                    elem[0].mem_index = mem_index
#                    elem[0].is_voltage_defined = False
##                    mem_index += 1
#                else:
#                    elem[0].is_voltage_defined = False
                elements += elem
            except KeyError:
                raise NetlistParseError("Parser: do not know how to parse" +
                                        " '%s' elements." % line[0])
                                
    #   Handle errors from individual parse functions
    except NetlistParseError as npe:
        (msg,) = npe.args
        if len(msg):
            printing.print_general_error(msg)
        printing.print_parse_error(line_n, line)
        raise NetlistParseError(msg)
        
#    sys.exit()    

    return elements


def get_next_file_and_close_current(file_list, file_index):
    if file_list[file_index - 1][2]:
        file_list[file_index - 1][0].close()
    if file_index == len(file_list):
        ffile = None
    else:
        ffile = open(file_list[file_index][1], "r")
        file_list[file_index][0] = ffile
    return ffile

def parse_models(models_lines):
    models = {}     
    for line, line_n in models_lines:
        if line_n == -1: # BSIM3 model file
            line_elements = line.split()
            ptr_model, model_labels = parse_includemodel_directive(line_elements[1])
            models.update({'ptr_model': ptr_model})
            for model_label in model_labels:
                model_parameters = {}
                model_parameters.update({'name': model_label})
                model_parameters.update({'TYPE': model_label[0]}) # rely on 1st letter of model_label to indicate np type
                model_parameters.update({'ptr_model': ptr_model})
                model_parameters.update({'para': {}})
                model_iter = bsim3.bsim3_mos_model(**model_parameters)
                models.update({model_label: model_iter})
        else:
            regex = re.compile("\s*[=]\s*")
            line = regex.sub("=", line)
            tokens = line.replace("(", "").replace(")", "").split()
            if len(tokens) < 3:
                raise NetlistParseError("parse_models(): syntax error in model" +
                                        " declaration on line " + str(line_n) +
                                        ".\n\t" + line)
    
            if options.Xyce_netlist:    
                model_label = tokens[1] # swap the type and label for compatibility with Xyce (Alex)
                model_type = tokens[2]
            else: # swap type and label. Revise Ahkab netlist
                model_label = tokens[1]
                model_type = tokens[2]
            model_parameters = {}
            for index in range(3, len(tokens)):
                if tokens[index][0] == "*":
                    break
                (label, value) = parse_param_value_from_string(tokens[index])
                model_parameters.update({label.upper(): value})
            if model_type == "ekv":
                model_iter = ekv.ekv_mos_model(**model_parameters)
                model_iter.name = model_label
            elif model_type == "mosq":
                model_iter = mosq.mosq_mos_model(**model_parameters)
                model_iter.name = model_label
            elif model_type == "diode" or model_type == 'd':
                model_parameters.update({'name': model_label})
                model_iter = diode.diode_model(**model_parameters)
            elif model_type == "memristor" or model_type == 'y':
                model_parameters.update({'name': model_label})
                model_iter = memristor.Yakopcic_model(**model_parameters)    
            elif model_type == "sw":
                model_parameters.update({'name': model_label})
                model_iter = switch.vswitch_model(**model_parameters)
            elif model_type == "csw":
                model_parameters.update({'name':model_label})
                model_iter = switch.iswitch_model(**model_parameters)
            elif options.useBsim3 and (model_type == "nmos" or model_type == "pmos"):
                if model_label in models:
                    models[model_label].para = model_parameters
                    # model_iter = models[model_label]
                else:
                    model_parameters.update({'name':model_label})
                    model_iter = bsim3.bsim3_mos_model_model(**model_parameters)  
            else:
                raise NetlistParseError("parse_models(): Unknown model (" +
                                        model_type + ") on line " + str(line_n) +
                                        ".\n\t" + line,)
            if model_label not in models:
                models.update({model_label: model_iter})
    return models


def parse_elem_resistor(line, circ):
    """Parses a resistor from the line supplied, adds its nodes to the circuit
    instance circ and returns a list holding the resistor element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit instance to which the resistor is to be connected.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.Resistor` element.

    """
    line_elements = line.split()
    if len(line_elements) < 4 or (len(line_elements) > 4 and not line_elements[4][0] in {"*", "$"} ):
        raise NetlistParseError("parse_elem_resistor(): malformed line")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    value = convert_units(line_elements[3])

    if value == 0:
        raise NetlistParseError("parse_elem_resistor(): ZERO-valued resistors are not allowed.")

    elem = devices.Resistor(part_id=line_elements[0], n1=n1, n2=n2, value=value)

    return [elem]


def parse_elem_capacitor(line, circ):
    """Parses a capacitor from the line supplied, adds its nodes to the circuit
    instance circ and returns a list holding the capacitor element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit to which the capacitor is to be connected.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.Capacitor` element.
    """
    line_elements = line.split()
    if len(line_elements) < 4 or \
       (len(line_elements) > 5 and not line_elements[5][0] in {"*", "$"} and
        not line_elements[4][0] in {"*", "$"}):
        raise NetlistParseError("parse_elem_capacitor(): malformed line")

    ic = None
    if len(line_elements) == 5 and not line_elements[4][0] == '*':
        (label, value) = parse_param_value_from_string(line_elements[4])
        if label == "ic":
            ic = convert_units(value)
        else:
            raise NetlistParseError("parse_elem_capacitor(): unknown parameter " + label)

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.Capacitor(part_id=line_elements[0], n1=n1, n2=n2,
                             value=convert_units(line_elements[3]), ic=ic)

    return [elem]


def parse_elem_inductor(line, circ):
    """Parses a inductor from the line supplied, adds its nodes to the circuit
    instance circ and returns a list holding the inductor element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit to which the inductor is to be connected.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.Inductor` element.
    """
    line_elements = line.split()
    if len(line_elements) < 4 or (len(line_elements) > 5 and not line_elements[6][0] == "*"):
        raise NetlistParseError("parse_elem_inductor(): malformed line")

    ic = None
    if len(line_elements) == 5 and not line_elements[4][0] == '*':
        (label, value) = parse_param_value_from_string(line_elements[4])
        if label == "ic":
            ic = convert_units(value)
        else:
            raise NetlistParseError("parse_elem_inductor(): unknown parameter " + label)

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.Inductor(part_id=line_elements[0], n1=n1, n2=n2,
                            value=convert_units(line_elements[3]), ic=ic)

    return [elem]


def parse_elem_inductor_coupling(line, circ, elements=[]):
    """Parses a inductor coupling from the line supplied,
    returns a list holding the inductor coupling element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit to which the inductor coupling is to be connected.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.InductorCoupling` element.
    """
    line_elements = line.split()
    if len(line_elements) < 4 or (len(line_elements) > 4 and not line_elements[5][0] == "*"):
        raise NetlistParseError("parse_elem_inductor_coupling(): malformed line")

    part_id = line_elements[0]
    L1 = line_elements[1]
    L2 = line_elements[2]

    try:
        Kvalue = convert_units(line_elements[3])
    except ValueError:
        (label, value) = parse_param_value_from_string(line_elements[3])
        if not label == "k":
            raise NetlistParseError("parse_elem_inductor_coupling(): unknown parameter " + label)
        Kvalue = convert_units(value)

    L1elem, L2elem = None, None

    for e in elements:
        if isinstance(e, devices.Inductor) and L1 == e.part_id:
            L1elem = e
        elif isinstance(e, devices.Inductor) and L2 == e.part_id:
            L2elem = e

    if L1elem is None or L2elem is None:
        error_msg = "parse_elem_inductor_coupling(): One or more coupled" + \
                    " inductors for %s were not found: %s (found: %s), %s (found: %s)." % \
                    (part_id, L1, L1elem is not None, L2, L2elem is not None)
        raise NetlistParseError(error_msg)

    M = math.sqrt(L1elem.value * L2elem.value) * Kvalue

    elem = devices.InductorCoupling(part_id=part_id, L1=L1, L2=L2, K=Kvalue,
                                    M=M)
    L1elem.coupling_devices.append(elem)
    L2elem.coupling_devices.append(elem)

    return [elem]


def parse_elem_vsource(line, circ):
    """Parses a voltage source from the line supplied, adds its nodes to the
    circuit instance and returns a list holding the element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the voltage source is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.VSource` element.
    """
    line_elements = line.replace(',', '').replace('(', ' ').replace(')', ' ').split() # remove , ( ) before split
    if len(line_elements) < 3:
        raise NetlistParseError("parse_elem_vsource(): malformed line")

    dc_value = None
    vac = None
    function = None

    index = 3
    while True:  # for index in range(3, len(line_elements)):
        if index >= len(line_elements):
            break
        if line_elements[index][0] == '*':
            break
        
        if len(line_elements) == 4: # Deal with vias (DC vsource of 0V) in IBM power grid netlists (Alex)
            line_elements = line_elements[:3] + ['type=vdc', '%s' % line_elements[3]]

        label, value = parse_param_value_from_string(line_elements[index])
        
        if label.isalpha() and value == '': # if the 'type' keyword is missing
            value = label
            label = 'type'

        if label == 'type':
            if value in ['vdc', 'dc']:
                param_number = 0
                dc_value = convert_units(line_elements[index + 1])
                index += 1
            elif value == 'vac':
                param_number = 0
            elif value == 'pulse':
                param_number = 7
            elif value == 'exp':
                param_number = 6
            elif value == 'sin':
                param_number = 5
            elif value == 'sffm':
                param_number = 5
            elif value == 'am':
                param_number = 5
            elif value == 'pwl':
                param_number = len(line_elements[index:-1])    
            elif value == 'dc':
                dc_value = convert_units(line_elements[index + 1])
                index += 2
                label, value = parse_param_value_from_string(line_elements[index])
                if label.isalpha() and value == '': # if the 'type' keyword is missing
                    value = label
                    label = 'type'
                    
                if value == 'pwl':
                    param_number = len(line_elements[index:-1])
                else:
                    param_number = 0
            else:
                raise NetlistParseError("parse_elem_vsource(): unknown signal" +
                                        "type %s" % value)
            if param_number and function is None:
                if value == 'pwl':
                    params = line_elements[index + 1:index + param_number + 1]
                    x = params[::2]
                    y = params[1::2]
                    idx_dup=[idx for idx, item in enumerate(x) if item in x[:idx]] # remove duplicates in x to make sure it is strictly increasing
                    for i in idx_dup[::-1]: 
                        x.pop(i)
                        y.pop(i)
                    function = parse_time_function(value, [x, y], 'voltage')
                else:    
                    function = parse_time_function(value, line_elements[index + 1:
                                                             index + param_number + 1], "voltage")
                index = index + param_number
                # continue
            elif function is not None:
                raise NetlistParseError("parse_elem_vsource(): only a time function can be defined.")
        elif label in ['vdc', 'dc']:
            dc_value = convert_units(value)
        elif label == 'vac':
            vac = convert_units(value)
        else:
            raise NetlistParseError("parse_elem_vsource(): unknown type %s" %
                                    label)
        index = index + 1

    if dc_value == None and function == None:
        raise NetlistParseError("parse_elem_vsource(): neither vdc nor a time function are defined.")

    # usual
    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.VSource(part_id=line_elements[0], n1=n1, n2=n2,
                           dc_value=dc_value, ac_value=vac)

    if function is not None:
        elem.is_timedependent = True
        elem._time_function = function

    return [elem]


def parse_elem_isource(line, circ):
    """Parses a current source from the line supplied, adds its nodes to the
    circuit instance and returns a list holding the current source element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the current source is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.ISource` element.
    """
    line_elements = line.replace(',', '').replace('(', ' ').replace(')', ' ').split() # remove , ( ) before split
    if len(line_elements) < 3:
        raise NetlistParseError("parse_elem_isource(): malformed line")
        
    if len(line_elements) == 4: # Deal with vias (DC vsource of 0V) in IBM power grid netlists (Alex)
        line_elements = line_elements[:3] + ['type=idc', 'idc=%s' % line_elements[3]]    

    dc_value = None
    iac = None
    function = None

    index = 3
    while True:  # for index in range(3, len(line_elements)):
        if index >= len(line_elements):
            break
        if line_elements[index][0] == '*':
            break
        if is_number(line_elements[index]) == True and is_number(line_elements[index+1]) == False:
            index += 1
            continue
            

        (label, value) = parse_param_value_from_string(line_elements[index])
        
        if label.isalpha() and value == '': # if the 'type' keyword is missing
            value = label
            label = 'type'
            

        if label == 'type':
            if value == 'idc':
                param_number = 0
            elif value == 'iac':
                param_number = 0
            elif value == 'pulse':
                param_number = 7
            elif value == 'exp':
                param_number = 6
            elif value == 'sin':
                param_number = 5
            elif value == 'sffm':
                param_number = 5
            elif value == 'am':
                param_number = 5
            elif value == 'pwl':
                param_number = len(line_elements[index:-1])       
            elif value == 'dc':
                dc_value = convert_units(line_elements[index + 1])
                index += 2
                label, value = parse_param_value_from_string(line_elements[index])
                if label.isalpha() and value == '': # if the 'type' keyword is missing
                    value = label
                    label = 'type'
                    
                if value == 'pwl':
                    param_number = len(line_elements[index:-1])
                else:
                    param_number = 0    
            else:
                raise NetlistParseError("parse_elem_isource(): unknown signal type.")
            if param_number and function is None:
                if value == 'pwl':
                    params = line_elements[index + 1:index + param_number + 1]
                    x = [convert_units(elem) for elem in params[::2]]
                    y = [convert_units(elem) for elem in params[1::2]]
                    idx_dup=[idx for idx, item in enumerate(x) if item in x[:idx]] # remove duplicates in x to make sure it is strictly increasing
                    for i in idx_dup[::-1]: 
                        x.pop(i)
                        y.pop(i)
                    function = parse_time_function(value, [x, y], 'current')
                    circ.GTP = circ.GTP + x
                    # plt.figure()
                    # plt.plot(x, y)
                    # plt.show()
                else:  
                    function = parse_time_function(value,
                                               line_elements[index + 1:
                                                             index + param_number + 1],
                                                   "current")
                index = index + param_number
            elif function is not None:
                raise NetlistParseError("parse_elem_isource(): only a time function can be defined.")
        elif label == 'idc':
            dc_value = convert_units(value)
        elif label == 'iac':
            iac = convert_units(value)
        else:
            raise NetlistParseError("parse_elem_isource(): unknown type "+label)
        index = index + 1

    if dc_value == None and function == None:
        raise NetlistParseError("parse_elem_isource(): neither idc nor a time function are defined.")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.ISource(part_id=line_elements[0], n1=n1, n2=n2,
                           dc_value=dc_value, ac_value=iac)

    if function is not None:
        elem.is_timedependent = True
        elem._time_function = function

    return [elem]

def parse_elem_bsource(line, circ):
    """Parses a nonlinear dependent source (bsource) from the line
    supplied, adds its nodes to the circuit instance and returns a
    list holding the bsource element.

    Syntax::

        BX N+ N- VNAME {EXPRESSION}

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the CCCS is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.FISource` element.
    """

    line_elements = line.split()
    if len(line_elements) < 4 or (len(line_elements) > 4
       and not line_elements[4][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    func = line_elements[3]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)
    
    if func[0].upper() == 'I':
        func = func.split('{', 1)[1].split('}')[0]
        # assume v(...) is the just applied bias (specific to cross sim) 
        # TODO: allow to include other variables (Alex)
        func = func.replace('v(' + ext_n1 + ',' + ext_n2 + ')', 'v') \
                    .replace('sgn', 'sign').replace('abs', 'Abs')
#        func = func.replace('v(' + ext_n1 + ',' + ext_n2 + ')', 'v')\
#                   .replace('sgn', 'np.sign').replace('exp', 'np.exp')\
#                   .replace('abs', 'np.abs')
#        func = utilities.str2numpy_func(func)
        elem = devices.BISource(part_id=line_elements[0], n1=n1, n2=n2,
                                func=func)
    else:
        raise NetlistParseError("nonlinear dependent vsource not implemented yet ")     

    return [elem]

def parse_elem_diode(line, circ, models=None):
    """Parses a diode from the line supplied, adds its nodes to the circuit
    instance and returns a list holding the diode element.

    Diode syntax:

    ::

        DX N+ N- <MODEL_LABEL> <AREA=xxx>

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the diode will be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.diode.Diode` element.
    """
    # sarebbe bello implementare anche: <IC=VD> <TEMP=T>
    Area = None
    T = None
    ic = None
    off = False

    line_elements = line.split()
    if len(line_elements) < 4:
        raise NetlistParseError("")

    model_label = line_elements[3]

    for index in range(4, len(line_elements)):
        if line_elements[index][0] == '*':
            break
        param, value = parse_param_value_from_string(line_elements[index])

        
        if not value == '':
            value = convert_units(value)
            if param == "area":
                Area = value
            elif param == "t":
                T = value
            elif param == "ic":
                ic = value
            elif param == "off":
                if not len(value):
                    off = True
                else:
                    off = convert_boolean(value)
            else:
                raise NetlistParseError("parse_elem_diode(): unknown parameter " + param)
        else:
            pass
            # print('ignore unknown diode parameters')

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    if model_label not in models:
        raise NetlistParseError("parse_elem_diode(): Unknown model id: " + model_label)
    elem = diode.diode(part_id=line_elements[0], n1=n1, n2=n2, model=models[
                       model_label], AREA=Area, ic=ic, off=off)
    return [elem]

def parse_elem_memristor(line, circ, models=None):
    """Parses a memristor from the line supplied, adds its nodes to the circuit
    instance and returns a list holding the memristor element.

    Diode syntax:

    ::

#        DX N+ N- <MODEL_LABEL> <AREA=xxx>
        ymemristor mx N+ N- <MODEL_LABEL>

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the memristor will be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.memristor.Diode` element.
    """
    # sarebbe bello implementare anche: <IC=VD> <TEMP=T>
    Area = None
    T = None
    ic = None
    off = False

    line_elements = line.split()
    if len(line_elements) < 4:
        raise NetlistParseError("")

    model_label = line_elements[4] # spcial for memristor (Alex)

    for index in range(5, len(line_elements)):
        if line_elements[index][0] == '*':
            break
        param, value = parse_param_value_from_string(line_elements[index])

        value = convert_units(value)
        if param == "area":
            Area = value
        elif param == "t":
            T = value
        elif param == "ic":
            ic = value
        elif param == "off":
            if not len(value):
                off = True
            else:
                off = convert_boolean(value)
        else:
            raise NetlistParseError("parse_elem_memristor(): unknown parameter " + param)

    ext_n1 = line_elements[2]
    ext_n2 = line_elements[3]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    if model_label not in models:
        raise NetlistParseError("parse_elem_memristor(): Unknown model id: " + model_label)
    elem = memristor.memristor(part_id='memristor!'+ line_elements[1], n1=n1, n2=n2, model=models[
                       model_label], AREA=Area, ic=ic, off=off)
    return [elem]

def parse_elem_mos(line, circ, models):
    """Parses a MOS transistor from the line supplied, adds its nodes to the
    circuit instance and returns a list holding the element.

    MOS syntax:

    ::
        MX ND NG NS KP=xxx Vt=xxx W=xxx L=xxx type=n/p <LAMBDA=xxx>

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit to which the element will be added.

    **Returns:**

    elements_list : list
        A list containing a MOS element.
    """
    #(self, nd, ng, ns, kp=1.0e-14, w, l, mos_type='n', lambd=0, type_of_elem="mosq")

    line_elements = line.split()
    if len(line_elements) < 6:
        raise NetlistParseError("parse_elem_mos(): required parameters are missing.")
        # print "MX ND NG NS model_id W=xxx L=xxx"

    model_label = line_elements[5]
    if model_label == '1' and line[0] == 'n':
        model_label = 'nch1' if options.useBsim3 else 'nmos' 
    elif model_label == '1' and line[0] == 'p':
        model_label = 'pch1' if options.useBsim3 else 'pmos'
    # kp = None
    w = None
    l = None
    # mos_type = None
    # vt = None
    m = 1
    n = 1
    idx = 0
    # lambd = 0 # va is supposed infinite if not specified
    for index in range(6, len(line_elements)):
        if line_elements[index][0] == '*':
            break
        if idx > 1: # only allow w and l parameters
            break
        param, value = parse_param_value_from_string(line_elements[index])        
        param_list = ['w', 'l']
        if value == '': # specific for SMORES 1k_cell example
            value = param
            param = param_list[idx]
            idx += 1
            
        if param == "w":
            w = convert_units(value)
        elif param == "l":
            l = convert_units(value)
        elif param == "m":
            m = convert_units(value)
        elif param == "n":
            n = convert_units(value)
        else:
            pass
#            raise NetlistParseError("parse_elem_mos(): unknown parameter " + param)

    if (w is None) or (l is None):
        raise NetlistParseError('parse_elem_mos(): required parameter ' +
                                'w'*(w is None) + ' and '*
                                (w is None and l is None) + 'l'*(l is None)+
                                'missing.')
        # print "MX ND NG NS W=xxx L=xxx <M=xxx> <N=xxx>"

    ext_nd = line_elements[1]
    ext_ng = line_elements[2]
    ext_ns = line_elements[3]
    ext_nb = line_elements[4]
    nd = circ.add_node(ext_nd)
    ng = circ.add_node(ext_ng)
    ns = circ.add_node(ext_ns)
    nb = circ.add_node(ext_nb)

    if model_label not in models:
        raise NetlistParseError("parse_elem_mos(): Unknown model ID: " + model_label)

    elem = None
    if isinstance(models[model_label], ekv.ekv_mos_model):
        elem = ekv.ekv_device(line_elements[0], nd, ng, ns, nb, w, l,
                              models[model_label], m, n)
    elif isinstance(models[model_label], mosq.mosq_mos_model):
        elem = mosq.mosq_device(line_elements[0], nd, ng, ns, nb, w, l,
                                models[model_label], m, n)
    elif isinstance(models[model_label], bsim3.bsim3_mos_model):
        elem = bsim3.bsim3_device(line_elements[0], nd, ng, ns, nb, w, l,
                                models[model_label], m, n)    
    else:
        raise NetlistParseError("parse_elem_mos(): Unknown MOS model type: " + model_label)

    return [elem]


def parse_elem_vcvs(line, circ):
    """Parses a voltage-controlled voltage source (VCVS) from the line
    supplied, adds its nodes to the circuit instance circ and returns a
    list holding the VCVS element.

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the VCVS is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.EVSource` element.
    """
    line_elements = line.split()
    if len(line_elements) < 6 or (len(line_elements) > 6 and not line_elements[6][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    ext_sn1 = line_elements[3]
    ext_sn2 = line_elements[4]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)
    sn1 = circ.add_node(ext_sn1)
    sn2 = circ.add_node(ext_sn2)

    elem = devices.EVSource(part_id=line_elements[0], n1=n1, n2=n2, sn1=sn1,
                            sn2=sn2, value=convert_units(line_elements[5]))

    return [elem]


def parse_elem_ccvs(line, circ):
    """Parses a current-controlled voltage source (CCVS) from the line supplied,
    adds its nodes to the circuit instance and returns a list holding
    the CCVS element.

    CCVS syntax:

    ::

       HXXX N1 N2 VNAME VALUE


    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the CCVS is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.HVSource` element.
    """
    #  0    1  2  3     4
    line_elements = line.split()
    if len(line_elements) < 5 or (len(line_elements) > 5 and not
                                  line_elements[5][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.HVSource(part_id=line_elements[0], n1=n1, n2=n2,
                            source_id=line_elements[3],
                            value=convert_units(line_elements[4]))

    return [elem]


def parse_elem_vccs(line, circ):
    """Parses a voltage-controlled current source (VCCS) from the line
    supplied, adds its nodes to the circuit instance and returns a
    list holding the VCCS element.

    Syntax:

    ::

        GX N+ N- NC+ NC- VALUE

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the VCCS is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.GISource` element.
    """
    line_elements = line.split()
    if len(line_elements) < 6 or (len(line_elements) > 6
       and not line_elements[6][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    ext_sn1 = line_elements[3]
    ext_sn2 = line_elements[4]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)
    sn1 = circ.add_node(ext_sn1)
    sn2 = circ.add_node(ext_sn2)

    elem = devices.GISource(part_id=line_elements[0], n1=n1, n2=n2, sn1=sn1,
                            sn2=sn2, value=convert_units(line_elements[5]))

    return [elem]


def parse_elem_cccs(line, circ):
    """Parses a current-controlled current source (CCCS) from the line
    supplied, adds its nodes to the circuit instance and returns a
    list holding the CCCS element.

    Syntax::

        FX N+ N- VNAME VALUE

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the CCCS is to be inserted.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.FISource` element.
    """

    line_elements = line.split()
    if len(line_elements) < 5 or (len(line_elements) > 5
       and not line_elements[5][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    source_id = line_elements[3]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = devices.FISource(part_id=line_elements[0], n1=n1, n2=n2,
                            source_id=source_id,
                            value=convert_units(line_elements[4]))

    return [elem]


def parse_elem_switch(line, circ, models=None):
    """Parses a switch device from the line supplied, adds its nodes to
    the circuit instance and returns a list holding the switch element.

    General syntax::

        SW1 n1 n2 ns1 ns2 model_label

    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance
        The circuit in which the switch is to be connected.
    models : dict, optional
        The currently defined models.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.switch.switch_device` element.
    """
    line_elements = line.split()
    if len(line_elements) < 6 or (len(line_elements) > 6 and not line_elements[6][0] == "*"):
        raise NetlistParseError("")

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    ext_sn1 = line_elements[3]
    ext_sn2 = line_elements[4]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)
    sn1 = circ.add_node(ext_sn1)
    sn2 = circ.add_node(ext_sn2)

    model_label = line_elements[5]
    elem = None

    if model_label not in models:
        raise NetlistParseError("Unknown model id: " + model_label)
    if isinstance(models[model_label], switch.vswitch_model):
        elem = switch.switch_device(
            n1, n2, sn1, sn2, models[model_label], part_id=line_elements[0])
    else:
        raise NetlistParseError("Unknown MOS model type: " + model_label)

    return [elem]


def parse_elem_user_defined(line, circ):
    """Parses a user defined element.

    In order for this to work, you should write a module that supplies the
    elem class.

    Syntax:
    Y<X> <n1> <n2> module=<module_name> type=<type> [<param1>=<value1> ...]

    This method will attempt to load the module <module_name> and it will
    then look for a class named <type>.

    An object will be instatiated with the following arguments:
    n1, n2, param_dict, get_int_id_func, convert_units_func
    Where:
    n1: is the anode of the element
    n2: is the cathode
    param_dict: is a dictionary, its elements are {param1:value1, ...}
    get_int_id_func, convert_units_func are two function that may be used
    in the __init__ method, if needed.
    get_int_id_func: a function that gives back the internal name of a node
    convert_units_func: utility function to convert eg 1p -> 1e-12

    See ideal_oscillators.py for a reference implementation.
    **Parameters:**

    line : string
        The netlist line.
    circ : circuit instance.
        The circuit to which the element will be added.

    **Returns:**

    elements_list : list
        A list containing a :class:`ahkab.devices.HVSource` element.

    Parameters:
    line: the line
    circ: the circuit instance.

    Returns: [userdef_elem]
    """
    line_elements = line.split()

    if len(line_elements) < 4:
        raise NetlistParseError("")

    param_dict = {}
    for index in range(3, len(line_elements)):
        if line_elements[index][0] == '*':
            break

        param, value = parse_param_value_from_string(line_elements[index])

        if param not in param_dict:
            param_dict.update({param: value})
        else:
            raise NetlistParseError(param + " already defined.")

    if "module" in param_dict:
        module_name = param_dict.pop("module", None)
    else:
        raise NetlistParseError("module name is missing.")

    if module_name in circuit.user_defined_modules_dict:
        module = circuit.user_defined_modules_dict[module_name]
    else:
        try:
            fp, pathname, description = imp.find_module(module_name)
            module = imp.load_module(module_name, fp, pathname, description)
        except ImportError:
            raise NetlistParseError("module " + module_name + " not found.")
        circuit.user_defined_modules_dict.update({module_name: module})

    if "type" in param_dict:
        elem_type_name = param_dict.pop("type", None)
    else:
        raise NetlistParseError("type of element is missing.")

    try:
        elem_class = getattr(module, elem_type_name)
    except AttributeError:
        raise NetlistParseError("module doesn't have elem type: " + \
            elem_type_name)

    ext_n1 = line_elements[1]
    ext_n2 = line_elements[2]
    n1 = circ.add_node(ext_n1)
    n2 = circ.add_node(ext_n2)

    elem = elem_class(n1, n2, param_dict, circ.add_node,
                      convert_units, part_id=line_elements[0])

    selfcheck_result, error_msg = elem.check()
    if not selfcheck_result:
        raise NetlistParseError("module: " + module_name + " elem type: " + elem_type_name + " error: " +\
            error_msg)
    #   TODO fixme non so sicuro che sia una buona idea

    return [elem]


def parse_time_function(ftype, line_elements, stype):
    """Parses a time function of type ftype from the line_elements supplied.

    **Parameters:**

    ftype : str
        One among ``"pulse"``, ``"exp"``, ``"sin"``, ``"sffm"`` or ``"am"``.
    line_elements : list of strings
        The tokens describing the time function. The list mustn't hold the
        ``"type=<ftype>"`` element
    stype : str
        Set this to "current" for current sources, "voltage" for voltage sources

    See :class:`ahkab.time_functions.pulse`, :class:`ahkab.time_functions.sin`,
    :class:`ahkab.time_functions.exp`, :class:`ahkab.time_functions.sffm` and
    :class:`ahkab.time_functions.am` for more.

    **Returns:**

    time_function : object
        A time-function instance
    """
    if not ftype in time_fun_specs:
        raise NetlistParseError("Unknown time function: %s" % ftype)
    prot_params = list(copy.deepcopy(time_fun_specs[ftype]['tokens']))

    fun_params = {}
    for i in range(len(line_elements)):
        token = line_elements[i]
        if token[0] == "*":
            break
        if is_valid_value_param_string(token):
            (label, value) = token.split('=')
        else:
            label, value = None, token
        assigned = False
        for t in prot_params:
            if (label is None and t['pos'] == i) or label == t['label']:
                fun_params.update({t['dest']: convert(value, t['type'])})
                assigned = True
                break
        if assigned:
            prot_params.pop(prot_params.index(t))
            continue
        else:
            raise NetlistParseError("Unknown .%s parameter: pos %d (%s=)%s" % \
                                     (ftype.upper(), i, label, value))

    missing = []
    for t in prot_params:
        if t['needed']:
            missing.append(t['label'])
    if len(missing):
        raise NetlistParseError("%s: required parameters are missing: %s" % (ftype, " ".join(line_elements)))
    # load defaults for unsupplied parameters
    for t in prot_params:
        fun_params.update({t['dest']: t['default']})

    fun = time_functions[ftype](**fun_params)
    fun._type = "V" * \
        (stype.lower() == "voltage") + "I" * (stype.lower() == "current")
    return fun


def convert_units(string_value):
    """Converts a value conforming to SPICE's syntax to ``float``.

    Quote from the SPICE3 manual:

        A number field may be an integer field (eg 12, -44), a floating point
        field (3.14159), either an integer or a floating point number followed
        by an integer exponent (1e-14, 2.65e3), or either an integer or a
        floating point number followed by one of the following scale factors:

        T = 1e12, G = 1e9, Meg = 1e6, K = 1e3, mil = 25.4x1e-6, m = 1e-3, u =
        1e-6, n = 1e-9, p = 1e-12, f = 1e-15

    :raises ValueError: if the supplied string can't be interpreted according
    to the above.

    **Returns:**

    num : float
        A float representation of ``string_value``.
    """

    if type(string_value) is float:
        return string_value  # not actually a string!
    if not len(string_value):
        raise NetlistParseError("")

    index = 0
    string_value = string_value.strip().upper()
    while(True):
        if len(string_value) == index:
            break
        if not (string_value[index].isdigit() or string_value[index] == "." or
                string_value[index] == "+" or string_value[index] == "-" or
                string_value[index] == "E"):
            break
        index = index + 1
    if index == 0:
        # print string_value
        raise ValueError("Unable to parse value: %s" % string_value)
        # return 0
    numeric_value = float(string_value[:index])
    multiplier = string_value[index:]
#    if len(multiplier) == 0:
#        pass # return numeric_value
#    elif multiplier == "T":
#        numeric_value = numeric_value * 1e12
#    elif multiplier == "G":
#        numeric_value = numeric_value * 1e9
#    elif multiplier == "K":
#        numeric_value = numeric_value * 1e3
#    elif multiplier == "M" or multiplier == "MS":
#        numeric_value = numeric_value * 1e-3
#    elif multiplier == "U" or multiplier == "US":
#        numeric_value = numeric_value * 1e-6
#    elif multiplier == "N" or multiplier == "NS":
#        numeric_value = numeric_value * 1e-9
#    elif multiplier == "P" or multiplier == "PS":
#        numeric_value = numeric_value * 1e-12
#    elif multiplier == "F" or multiplier == "FS":
#        numeric_value = numeric_value * 1e-15
#    elif multiplier == "MEG":
#        numeric_value = numeric_value * 1e6
#    elif multiplier == "MIL":
#        numeric_value = numeric_value * 25.4e-6
#    elif multiplier == "S":
#        numeric_value = numeric_value
#    else:
#        numeric_value = numeric_value # for unknown multiplier keep the numeric value (Alex)
##        raise ValueError("Unknown multiplier %s" % multiplier)
    if multiplier in unit_conversion_table.keys():
        numeric_value = numeric_value * unit_conversion_table[multiplier]
    else:
        pass    
    return numeric_value


def parse_postproc(circ, postproc_direc):
    postproc_list = []
    valid_var_names = ['v', 'i', 'n']
    for line, line_n in postproc_direc:
        if not line[0] == ".":
            continue

        try:
            line_elements = line.split()
            # plot
            if line_elements[0] == ".plot":
                plot_postproc = {}
                plot_postproc["type"] = "plot"
                plot_postproc["analysis"] = line_elements[1]
                if not (plot_postproc["analysis"] == "tran" or
                        plot_postproc["analysis"] == "pss" or
                        plot_postproc["analysis"] == "ac" or
                        plot_postproc["analysis"] == "dc"
                        ):
                    printing.print_general_error("Plotting is unsupported for" +
                                                 "analysis type " +
                                                 plot_postproc["analysis"])

                graph_labels = ""
                for glabel in line_elements[2:]:
                    graph_labels = graph_labels + " " + glabel

                l2l1 = plotting._split_netlist_label(graph_labels)

                if plot_postproc["analysis"] == "ac":
                    l2l1ac = []
                    for l2, l1 in l2l1:
                        if l1 is not None:
                            l1 = "|%s|" % (l1, )
                        else:
                            l1 = None
                        if l2 is not None:
                            l2 = "|%s|" % (l2, )
                        else:
                            l2 = None
                        l2l1ac.append((l2, l1))
                    l2l1 = l2l1ac
                plot_postproc["l2l1"] = l2l1
                postproc_list.append(plot_postproc)
            #print
            elif line_elements[0] in {".print", ".probe"}:
                print_postproc = {}
                print_postproc["type"] = "print"
                idx = 1
                print_postproc["analysis"] = line_elements[idx]
                if not (print_postproc["analysis"] == "tran" or
                        print_postproc["analysis"] == "pss" or
                        print_postproc["analysis"] == "ac" or
                        print_postproc["analysis"] == "dc"
                        ):
                    print_postproc["analysis"] = 'tran' # if analysis type not provides, 
                                                        # set it to 'tran' and read variables starting fomr elements[1]
#                    printing.print_general_error("Printing is unsupported for" +
#                                                 "analysis type " +
#                                                 print_postproc["analysis"])
                else:
                    idx += 1
                var_labels = []
                for vlabel in line_elements[idx:]:
#                    vlabel = vlabel.replace('(', '').replace(')', '').upper()
                    name = vlabel[0]
                    if name in valid_var_names:
                        var_labels += [vlabel]
                    else:
                        pass
                print_postproc["printvar"] = var_labels
                postproc_list.append(print_postproc)    
            # fourier
            elif line_elements[0] == ".four":
                if type(convert_units(line_elements[1])) is not float:
                    raise NetlistParseError('postprocessing(): fourier' +
                                            ' fundamental \'%s\'?' %
                                            line_elements[1])
                fpv = {'type':'four',
                       'fund':convert_units(line_elements[1])}
                variables = []
                for le in line_elements[2:]:
                    if le[0] == '*':
                        break
                    variables += [plotting._split_netlist_label(le)[0]]
                fpv.update({'variables':tuple(variables)})
                postproc_list.append(fpv)
            elif line_elements[0] == '.fft':
                fpv = {'type': 'fft'}
                params = list(copy.deepcopy(fft_specs['fft']['tokens']))
                for i in range(len(line_elements[1:])):
                    token = line_elements[i + 1]
                    if token[0] == "*":
                        break
                    if is_valid_value_param_string(token):
                        label, value = token.split('=')
                    else:
                        label, value = None, token
                    assigned = False
                    for t in params:
                        if (label is None and t['pos'] == i) or label == t['label']:
                            fpv.update({t['dest']: convert(value, t['type'])})
                            assigned = True
                            break
                    if assigned:
                        params.pop(params.index(t))
                        continue
                    else:
                        raise NetlistParseError("Unknown .%s parameter: pos %d (%s=)%s" % \
                                                 (fpv[type].upper(), i, label, value))
                # is there anything required which is missing?
                missing = []
                for t in params:
                    if t['needed']:
                        missing.append(t['label'])
                if len(missing):
                    raise NetlistParseError("Required parameters are missing: %s" %
                                            (" ".join(line_elements)))
                # load defaults for unsupplied parameters
                for t in params:
                    fpv.update({t['dest']: t['default']})
                fpv.update({'label':
                            tuple(plotting._split_netlist_label(fpv['label'])[0])})
                postproc_list.append(fpv)
            else:
                raise NetlistParseError("Unknown postproc directive %s." %
                                        line_elements[0])
        except NetlistParseError as npe:
            (msg,) = npe.args
            if len(msg):
                printing.print_general_error(msg)
            printing.print_parse_error(line_n, line)
            raise NetlistParseError(msg)
    return postproc_list

def parse_options(option_directives): # added by Alex
    for line, line_n in option_directives:
        if line[0] != '.':
            continue
        regex = re.compile("\s*[=]\s*") # remove space at the two ends of '='
        line = regex.sub("=", line)
        line_elements = line.split()
        token = line_elements[1]
        if is_valid_value_param_string(token):
            (label, value) = token.split('=')
        else:
            label, value = None, token
        try:
            value = float(value)
        except ValueError:
            pass
        if label is not None:
            if hasattr(options, label.lower()):
                setattr(options, label.lower(), value)
            elif hasattr(options, label.upper()):
                setattr(options, label.upper(), value)
            else:
                print("option line %s is ignored" % line)
        else:
            print("option line %s is ignored" % line)
    return

def parse_ics(directives):
    ics = []
    for line, line_n in directives:
        if line[0] != '.':
            continue
#        line_elements = line.split()
        if line[:3] == '.ic':
            icdict = parse_ic_directive(line)
            if not ics:
                ics += [icdict]
            else:    
                for ic in ics:
                    ic_label = list(icdict.keys())[0]
                    if list(ic.keys())[0] == ic_label:
                        ic[ic_label].update(icdict[ic_label])
    return ics


def parse_analysis(circ, directives):
    """Parses the analyses.

    **Parameters:**

    circ: circuit class instance
        The circuit description
    directives: list of tuples
        The list should be assembled as ``(line, line_number)``.

    Both of them are returned by ``parse_circuit()``

    **Returns:**

    a list of the analyses
    """
    an = []
    for line, line_n in directives:
        if line[0] != '.' or line[:3] == '.ic':
            continue
#        line_elements = line.split()
        an_single = parse_single_analysis(line)
        # identify breakpoints from pwl and pulse sources for transient simulation
        if an_single['type'] in {'tran', 'pss'}:            
            breakpoints = []
            for elem in circ:
                if isinstance(elem, devices.VSource) or isinstance(elem, devices.ISource):
                    if isinstance(elem._time_function, pwl):
#                        x = [e for e in elem._time_function.x if e not in tuple(breakpoints + [0])]
                        x = [e for e in elem._time_function.x if (np.min(np.abs(e - np.array(breakpoints + [0]))) > options.hmin) 
                             and (e < (an_single['tstop'] + options.hmin))]
                        breakpoints += x
                    if isinstance(elem._time_function, pulse):  
                        td, tr, tf, pw, per = attrgetter('td', 'tr', 'tf', 'pw', 'per')(elem._time_function)
                        if an_single['type'] == 'pss':
                            tstop = per
                        else:
                            tstop = an_single['tstop']
                        bkpt0 = np.unique([td, td + tr, td + tr + pw, td + tr + pw + tf])
                        num_per = int(np.ceil(tstop / per))
                        for i in range(num_per):
                            bkpt = [e + i * per for e in bkpt0]
                            x = [e for e in bkpt if 
                                 (np.min(np.abs(e - np.array(breakpoints + [0]))) > options.hmin 
                                 and e <= tstop)]
                            breakpoints += x
            an_single['breakpoints'] = np.array(breakpoints)
            an_single['breakpoints'].sort()
        an += [an_single]
    return an


def parse_temp_directive(line):
    """Parses a TEMP directive:

    The syntax is::

        .TEMP <VALUE>

    """
    line_elements = line.split()
    for token in line_elements[1:]:
        if token[0] == "*":
            break
        value = convert_units(token)

    return {"type": "temp", "temp": value}


def parse_single_analysis(line):
    """Parses an analysis

    **Parameters:**

    line : str
        The netlist line from which an analysis statement is to be parsed.

    **Returns:**

    an : dict
        A dictionary with its parameters as keys.

    :raises NetlistParseError: if the analysis is not parsed correctly.
    """
    line_elements = line.split()
    an_type = line_elements[0].replace(".", "").lower()
    if not an_type in specs:
        raise NetlistParseError("Unknown directive: %s" % an_type)
    params = list(copy.deepcopy(specs[an_type]['tokens']))

    an = {'type': an_type}
    for i in range(len(line_elements[1:])):
        token = line_elements[i + 1]
        if token[0] == "*":
            break
        if is_valid_value_param_string(token):
            (label, value) = token.split('=')
        else:
            label, value = None, token
        assigned = False
        for t in params:
            if (label is None and t['pos'] == i) or label == t['label']:
                an.update({t['dest']: convert(value, t['type'])})
                assigned = True
                break
        if assigned:
            params.pop(params.index(t))     #pop the assigned paramters
            continue
        else:
            raise NetlistParseError("Unknown .%s parameter: pos %d (%s=)%s" % \
                                     (an_type.upper(), i, label, value))

    missing = []
    for t in params:
        if t['needed']:
            missing.append(t['label'])
    if len(missing):
        raise NetlistParseError("Required parameters are missing: %s" %
                                (" ".join(line_elements)))
    # load defaults for unsupplied parameters
    for t in params:
        an.update({t['dest']: t['default']})

    # ad-hoc code for tran ... :(
    if an['type'] == 'tran':
        if type(an['uic']) == str: 
            an['uic'] = 3
        uic = int(an.pop('uic'))
        if uic == 0:
            an['x0'] = None
        elif uic == 1:
            an['x0'] = 'op'
        elif uic == 2:
            an['x0'] = 'op+ic'
        elif uic == 3:
            an['x0'] = 'tran'
            pass  # already set by ic_label
        else:
            raise NetlistParseError("Unknown UIC value: %d" % uic)
    # ... and pz :(
    if an['type'] == 'pz':
        an.update({'x0':'op'})

    return an

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

def is_valid_value_param_string(astr):
    """Has the string a form like ``<param_name>=<value>``?

    .. note::

        No spaces.

    **Returns:**

    ans : a boolean
        The answer to the above question.
    """
    if type(astr) == str:
        work_astr = astr.strip()
        if work_astr.count("=") == 1:
            ret_value = True
        else:
            ret_value = False
    else:
        ret_value = False
    return ret_value


def convert(astr, rtype, raise_exception=False):
    """Convert a string to a different representation

    **Parameters:**

    astr : str
        The string to be converted.
    rtype : type
        One among ``float``, if a ``float`` sould be parsed from ``astr``,
        ``bool``, for parsing a boolean or ``str`` to get back a string (no
        parsing).
    raise_exception : boolean, optional
        Set this flag to ``True`` if you wish for this function to raise
        ``ValueError`` if parsing fails.

    **Returns:**

    ret : object
        The parsed data.
    """
    if rtype == float:
        try:
            ret = convert_units(astr)
        except ValueError as msg:
            if raise_exception:
                raise ValueError(msg)
            else:
                ret = astr
    elif rtype == str:
        ret = astr
    elif rtype == bool:
        ret = convert_boolean(astr)
    elif rtype == list:
        ret = [convert(elem, float) for elem in astr]
    elif raise_exception:
        raise ValueError("Unknown type %s" % rtype)
    else:
        ret = astr
    return ret


def parse_param_value_from_string(astr, rtype=float, raise_exception=False):
    """Search the string for a ``<param>=<value>`` couple and returns a list.

    **Parameters:**

    astr : str
        The string to be converted.
    rtype : type
        One among ``float``, if a ``float`` sould be parsed from ``astr``,
        ``bool``, for parsing a boolean or ``str`` to get back a string (no
        parsing).
    raise_exception : boolean, optional
        Set this flag to ``True`` if you wish for this function to raise
        ``ValueError`` if parsing fails.

    **Returns:**

    ret : object
        The parsed data. If the conversion fails and ``raise_exception`` is not
        set, a ``string`` is returned.

    * If ``rtype`` is ``float`` (the type), its default value, the method will
      attempt converting ``astr`` to a float. If the conversion fails, a string
      is returned.
    * If set ``rtype`` to ``str`` (again, the type), a string will always be
      returned, as if the conversion failed.

    This prevents ``'0'`` (str) being detected as ``float`` and converted to 0.0,
    ending up being a new node instead of the reference.

    Notice that in ``<param>=<value>`` there is no space before or after the equal sign.

    **Returns:**

    alist : ``[param, value]``
        where ``param`` is a string and ``value`` is parsed as described.
    """
    if not is_valid_value_param_string(astr):
        return (astr, "")
    p, v = astr.strip().split("=")
    v = convert(v, rtype, raise_exception=False)
    return p, v


class NetlistParseError(Exception):
    """Netlist parsing exception."""
    pass


def convert_boolean(value):
    """Converts the following strings to a boolean:
    yes, 1, true to True
    no, false, 0 to False

    raises NetlistParserException

    Returns: boolean
    """
    if value == 'no' or value == 'false' or value == '0' or value == 0:
        return_value = False
    elif value == 'yes' or value == 'true' or value == '1' or value == 1:
        return_value = True
    else:
        raise NetlistParseError("invalid boolean: " + value)

    return return_value


def parse_ic_directive(line):
    """Parses an ic directive and assembles a dictionary accordingly.
    """
    line = line.replace(' = ', '=')
    line_elements = line.split()
    ic_dict = {}
#    name = None
    name = 'tran'
    for token in line_elements[1:]:
        if token[0] == "*":
            break
        
        (label, value) = parse_param_value_from_string(token)
        if label == "name" and name is None:
            name = value
            continue

        # the user should have specified either something like:
        # V(node)=10u
        # or something like:
        # I(Vtest)=100e-6
        ic_dict.update({label: convert_units(value)})
        # We may decide to check if the node exists and/or if the syntax
        # is correct and raise NetlistParseError if needed.

    if name is None:
        raise NetlistParseError("The 'name' parameter is missing")

#    return ic_dict # return ic_dict instead of a name-value pair (Alex)
#    return ic_dict
    return {name: ic_dict}


def parse_sub_declaration(subckt_lines):
    """Returns a circuit.subckt instance that holds the subckt
    information, ready to be instantiated/called.
    """
    index = 0
    netlist_lines = []
    connected_nodes_list = []
    params_dict = {}
    for line, line_n in subckt_lines:
        if index == 0:
            line = line.replace(' = ', '=')
            line_elements = line.split()
            if line_elements[0] != '.subckt':
                raise RuntimeError("BUG? parse_sub_declaration() \
                called on non-subckt text. (line" + str(line_n) + ")")
            name = line_elements[1]
            idx = 0
            for node_name in line_elements[2:]:
                if node_name[0] == '0':
                    raise NetlistParseError("subckt " + name + \
                        " has a connection node named '0' (line" + str(
                            line_n) + ")")
                if node_name[0] == '*':
                    break
                elif options.Xyce_netlist and node_name.upper() == 'PARAMS:': # parse parameters definition in the first line
                    for token in line_elements[2 + idx + 1:]:
                        (label, value) = parse_param_value_from_string(token)
                        params_dict.update({label: value})
                    break
                else:
                    connected_nodes_list = connected_nodes_list + [node_name]
                    idx += 1
        else:
            netlist_lines = netlist_lines + [(line, "")]
        index = index + 1
    subck_inst = circuit.subckt(name, netlist_lines, connected_nodes_list, params_dict)
    return subck_inst


def parse_sub_instance(line, circ, subckts_dict, models=None):
    """Parses a subckt call/instance.

    1. Gets name and nodes connections
    2. Looks in subckts_dict for a matching subckts_dict[name]
    3. Builds a circuit wrapper
    4. Calls main_netlist_parser() on the subcircuit code
       (with the wrapped circuit)

    Returns: a elements list

    """
    line = line.replace(' = ', '=')
    line_elements = line.split()
    if len(line_elements) < 2:
        raise NetlistParseError("")

    param_value_dict = {}
    name = None
    if options.Xyce_netlist: # read parameters for subckt declaration and store in param_dict (Alex) 
#        name_idx = -1
        param_idx = [idx for idx, token in enumerate(line_elements) if token.upper() == 'PARAMS:'] #location of "PARAMs:"
        if param_idx == []:
            name_idx = len(line_elements) - 1 # name is the last label if no "PARAMS:" found
        else:
            name_idx = param_idx[0] - 1 # otherwise it is in front of "PARAMS:"
        param_value_dict.update({'name': line_elements[name_idx]}) # revision for Xyce 
        name = param_value_dict['name']
        subckt = subckts_dict[name]
        connection_nodes_dict = {}
        parameters_dict = subckt.params_dict
#        node_last = name_idx if name_idx != -1 else len(line_elements) - 1 
        for index in range(1, name_idx): # parse node info as usual up to "name" label
            if line_elements[index][0] == '*':
                break
    
            param, value = parse_param_value_from_string(line_elements[index],
                                                         rtype=str)
            if value != '':
                connection_nodes_dict.update({param: value})
            else:
                value = param
                param = subckt.connected_nodes_list[index-1]
                connection_nodes_dict.update({param: value})
        for index in range(name_idx + 2, len(line_elements)): # parse parameter info after "params:" keyword
            param, value = parse_param_value_from_string(line_elements[index]) # parameter values must be float
            if value == '':
                raise NetlistParseError("")
            else:
                if param in parameters_dict:
                    parameters_dict[param] = value
    else:
        param_idx = [idx for idx, token in enumerate(line_elements) if token.upper() == 'PARAMS:'] #location of "PARAMs:"
        if param_idx == []:
            name_idx = len(line_elements) - 1 # name is the last label if no "PARAMS:" found
        else:
            name_idx = param_idx[0] - 1 # otherwise it is in front of "PARAMS:"
        param_value_dict.update({'name': line_elements[name_idx]}) # revision for Xyce 
        if "name" not in param_value_dict:
            raise NetlistParseError("missing 'name' in subckt call")
        if param_value_dict['name'] not in subckts_dict:
            raise NetlistParseError("subckt " + \
                param_value_dict['name'] + " is unknown")
    
        name = param_value_dict['name']
        subckt = subckts_dict[name]
        connection_nodes_dict = {}
        parameters_dict = subckt.params_dict
        for index in range(1, len(line_elements) - 1):
            if line_elements[index][0] == '*':
                break
    
            param, value = parse_param_value_from_string(line_elements[index],
                                                         rtype=str)
            if value != '':
                connection_nodes_dict.update({param: value})
            else:
                value = param
                param = subckt.connected_nodes_list[index-1]
                connection_nodes_dict.update({param: value})
#            param_value_dict.update({param:value})
#        for param, value in param_value_dict.items():
#            if param == 'name':
#                continue
#            
#            if param in subckt.connected_nodes_list:
#                connection_nodes_dict.update({param: value})
#            else:
#                raise NetlistParseError("unknown node " + param)

    # check all nodes are connected
    for node in subckt.connected_nodes_list:
        if node not in connection_nodes_dict:
            raise NetlistParseError("unconnected subckt node " + node)

    if options.Xyce_netlist:
        subckt_partid = line_elements[0]
    else:
        subckt_partid = line_elements[0][1:] # LTSpice style. remove [1:] for Xyce style
    wrapped_circ = circuit._circuit_wrapper(circ, connection_nodes_dict, parameters_dict, 
                                            subckt.name, subckt_partid) 
    wrapped_circ.parameterize(subckt.code)
    elements_list = main_netlist_parser(wrapped_circ, wrapped_circ.code,
                                        subckts_dict, models)

    # Every subckt adds elements with the _same description_ (elem.part_id[1:])
    # We modify it so that each description is unique for every instance
    for element in elements_list:
#        element.part_id = element.part_id[0] + "-" + wrapped_circ.prefix + element.part_id[1:]
        # element.part_id = wrapped_circ.prefix + element.part_id # Xyce style                  
        element.part_id = wrapped_circ.prefix + element.part_id # LTSpice style by Alex
        if isinstance(element, devices.InductorCoupling):
            element.L1 = wrapped_circ.prefix + element.L1
            element.L2 = wrapped_circ.prefix + element.L2
    return elements_list


def parse_include_directive(line, netlist_wd):
    """.include <filename> [*comments]
    """
#    line_elements = line.split()
    if sys.platform == 'win32':
        if './' in line:
            line_elements = line.split("./") # for relative path only (Alex)
        else:
            line_elements = line.split(" ")
    else:
        line_elements = line.split("\"") # the file name often given in as a path in double quotes
    line_elements = [elem for elem in line_elements if elem != '']
    if not len(line_elements) > 1 or \
            (len(line_elements) > 2 and not line_elements[2][0] == '*'):
        raise NetlistParseError("")

    path = line_elements[1]
    if not os.path.isabs(path):
        # the user did not specify the full path.
        # the path is then assumed to be relative to the netlist location
        path = os.path.join(netlist_wd, path)
    if not utilities.check_file(path):
        raise RuntimeError("")

    # fnew = open(path, "r")

    return [None, path, True]

def parse_includemodel_directive(filename):
    
    dll_file = "BSIM3_DLL_new.dll"
    bsim = ctypes.cdll.LoadLibrary(dll_file)
    filename = ctypes.c_char_p(filename.encode('utf-8'))
    bsim.LoadBSIM3Model.argtypes = [ctypes.c_char_p]
    bsim.LoadBSIM3Model.restype = ctypes.c_double
    ptr_model = bsim.LoadBSIM3Model(filename)
    
    bsim.ListModelName.argtypes =[ctypes.c_double]
    bsim.ListModelName.restype = ctypes.POINTER(ctypes.c_char_p)
    bsim.ListModelName(ptr_model)
    model_label = bsim.ListModelName(ptr_model)

    bsim.getcount.argtypes =[ctypes.c_double]
    bsim.getcount.restype = ctypes.c_int
    counter = bsim.getcount(ptr_model)
    
    model_labels = []
    for i in range(counter):
        model_labels.append(model_label[i].decode('utf-8'))
        # print(model_label[i])
    
    return ptr_model, model_labels

def join_lines(fp, line):
    """Read the lines coming up in the file. Each line that starts with '+' is added to the
    previous line (line continuation rule). When a line not starting with '+' is found, the
    file is rolled back and the line is returned.
    """
    while True:
        last_pos = fp.tell()
        next = fp.readline()
        next = next.strip().lower()
        if not next:
            break
        elif next[0] == '+':
            line += ' ' + next[1:]
        else:
            fp.seek(last_pos)
            break
    return line
