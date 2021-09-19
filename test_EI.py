# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:06:36 2019

@author: Administrator
True"""
import os
import numpy as np
import scipy.io as spio
import scipy as sp
#from scipy.sparse import linalg
from operator import itemgetter
import time
import sys
#import pypardiso
# from pypardiso import PyPardisoSolver
# from pypardiso import spsolve
import ahkab as ak

testcase = 'dyn'
ak.options.testcase = testcase
if testcase == 'dyn':
    netlistfile = '.\\dyn.sp'
    ak.options.Xyce_netlist = False
    ak.options.transient_no_step_control = True
    ak.options.useBsim3 = False
    ak.options.default_tran_method = "GEAR2"
elif testcase == 'pkg_dyn':
    netlistfile = 'D:\\Onedrive\\OneDrive - The Hong Kong Polytechnic University\\Alex\\Research\\Circuit Simulation\\GIGA\\package_spice_cases\\package_spice_cases\\chip_package_1L1K1R\\pkg_dyn.sp'
    ak.options.Xyce_netlist = False
    ak.options.transient_no_step_control = True
    ak.options.useBsim3 = False
    ak.options.default_tran_method = "EI"
elif testcase == 'pkg_dyn_big':
    netlistfile = 'D:\\Onedrive\\OneDrive - The Hong Kong Polytechnic University\\Alex\\Research\\Circuit Simulation\\GIGA\\package_spice_cases\\package_spice_cases\\chip_package_10L1K10R\\pkg_dyn_10L1K10R.sp'
    ak.options.Xyce_netlist = False
    ak.options.transient_no_step_control = True
    ak.options.useBsim3 = False
    ak.options.default_tran_method = "EI"     
elif testcase == 'pkg_dyn_big_current':
    netlistfile = 'D:\\Onedrive\\OneDrive - The Hong Kong Polytechnic University\\Alex\\Research\\Circuit Simulation\\GIGA\\prof_chen_giga\\chip_package_10L1K10R\\pkg_chip_tri_current_junc_ahakb.sp'
    ak.options.Xyce_netlist = False
    ak.options.transient_no_step_control = True
    ak.options.useBsim3 = False
    ak.options.default_tran_method = "EI"      
elif testcase == '1k':
    ak.options.Xyce_netlist = False
    netlistfile = '1k_cell.ckt' 
    # ak.options.transient_no_step_control = False
    ak.options.transient_no_step_control = False
    ak.options.useBsim3 = False
else:
    #netlistfile = 'Yakopcic.spice'
    ak.options.Xyce_netlist = True
    
    #netlistfile = 'homework.spice'
    #netlistfile = 'C:\\Research\\Circuit simulation\\Cross Sim\\netlist\\ahkab.cir'
    #netlistfile = 'C:\\Research\\Circuit simulation\\Cross Sim\\examples\\examples\\output\\xyce.cir'
    netlistfile = 'C:\\Research\\Circuit simulation\\Cross Sim\\netlist_tmp\\xyce.cir'
    #netlistfile = 'ibmpg5.sp'
    ak.options.transient_no_step_control = False
if ak.options.transient_no_step_control == False:
    identifier = '_cs_adaptive'
else:
    identifier = '_cs_fixed'
#identifier = '_4e-10'
outfile = os.path.splitext(netlistfile)[0] + '_' + ak.options.default_tran_method + identifier


ak.options.use_sparse = True
ak.options.nr_damp_first_iters = False
ak.options.transient_prediction_as_x0 = False
ak.options.transient_use_aposteriori_step_control = True
ak.options.use_standard_solve_method = True
ak.options.use_gmin_stepping = False
ak.options.use_source_stepping = False
ak.options.lin_sol_method = 'splu'
#ak.options.lin_sol_method = 'GMRES'
ak.options.mintimestepsbp = 1
ak.options.lin_GMRES_tol = 1e-6
ak.options.reparse = 2 # 0：ｎｏ　ｒｅｐａｒｓｅ，　１： reparse only directives, 2: full reparse

# ak.options.default_tran_method = "EI"
# ak.options.default_tran_method = "IMPLICIT_EULER"
# ak.options.default_tran_method = "TRAP"
ak.options.ei_reg = True
# ak.options.ei_reg = True
#ak.options.ei_newton = False
ak.options.ei_newton = True
ak.options.ei_use_jac22 = True
ak.options.ei_max_m = 50
scl = 1e-0
ak.options.vea = 1e-4 * scl
#: Voltage relative tolerance.
ak.options.ver = 1e-3 * scl
#: Current absolute tolerance.
ak.options.iea = 1e-4 * scl
#: Current relative tolerance.
ak.options.ier = 1e-3 * scl
results = ak.main(netlistfile, outfile=outfile, verbose=6)


print("complete simulation")
#sys.exit(1)
#import ahkab as ak
if 0:
    import matplotlib.pyplot as plt
    result_file= outfile + '.tran'
    with open(result_file) as rf:
        data = np.loadtxt(rf.readlines()[:-1], skiprows=1, dtype=None)
    plt.figure()
    t = data[:,0]
    I = data[:,2]
    if testcase == 'C':
        ref = sp.io.loadmat('colpitts_vnd0.mat')
        t0 = ref['t0'][0]
        V0 = ref['V0'][0]
        V = data[:,2]
        plt.plot(t0, V0, '-o', t, V, '-*', ms=4, lw=1.25)
        plt.gca().legend(('ref', 'exp'))
    elif testcase == '1k':
        V = data[:,951]
        plt.plot(t, V, '-o', ms=4, lw=1.25) 
    elif testcase == 'inv':
        V = data[:,3]
        plt.plot(t, V, '-o', ms=4, lw=1.25)    
    else:
        V = data[:,13]
        plt.plot(t, V, '-o', ms=4, lw=1.25)
    
    plt.show()
#sp.io.savemat('Colpitts_vnd0.mat', {'t0': t, 'V0': V})

