# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:57:16 2019

@author: Administrator
"""

      if iteration >= 100:
            D = D * (gamma / dt)
#            M0 = (D + mna + J).toarray()
#            rhs0 = D.dot(x - xtold) + mna.dot(x) + Tx + T
#            dx0 = np.linalg.solve(M0, -rhs0)
            
            x1 = x[idx1]
            x2 = x[idx2]
            Cs = D[idx1, :][:, idx1].toarray()
#            Cs = Cs * (gamma / dt)
            n1= len(Cs)
            Csinv = np.linalg.inv(Cs)
            G11 = mna[idx1, :][:, idx1].toarray()
            G12 = mna[idx1, :][:, idx2].toarray()
            G21 = mna21.toarray()
            G22 = mna22.toarray()
            G22inv = np.linalg.inv(G22)
#            del G22
            Gs = G11 - G12.dot(G22inv.dot(G21))
#            
#            del  G11, G21
            
#            Txs = Tx[idx1] - G12.dot(G22inv.dot(Tx[idx2]))
#            Ts = T[idx1] - G12.dot(G22inv.dot(T[idx2]))
#            Js1 = J11.toarray() - G12.dot(G22inv.dot(J21.toarray()))
#            Js2 = J12.toarray() - G12.dot(G22inv.dot(J22.toarray()))
#            Ms11 = Cs + Gs + Js1
#            Ms22 = G22 + J22
#            Ms = np.block([[Ms11, Js2], [G21 + J21, Ms22]])
#            rhss = np.zeros((mna_size, 1))
#            rhss[idx1] = Cs.dot(x[idx1] - xtold[idx1]) + Gs.dot(x[idx1]) + Txs + Ts
#            rhss[idx2] = G21.dot(x1) + G22.dot(x2) + Tx[idx2] + T[idx2]
#            dxs = np.linalg.solve(Ms, -rhss)
#            
            rhs2 = G21.dot(x1) + Tx[idx2] + T[idx2]
            x2p = np.linalg.solve(G22, -(G21.dot(x1) + Tx[idx2] + T[idx2]))
            xp = x.copy()
            xp[idx2] = x2p
            _, Txp = dc_analysis.generate_J_and_Tx(circ, xp, time, nojac=True)
            Txsp = Txp[idx1] - G12.dot(G22inv.dot(Txp[idx2]))
            Ts = T[idx1] - G12.dot(G22inv.dot(T[idx2]))
            M21 = G21 + J21
            M22 = G22 + J22
            Jg1 = -G22inv.dot(M21)
            Js1 = J11 + J12.dot(Jg1)
            Js2 = J21 + J22.dot(Jg1)
            Js = Js1 - G12.dot(G22inv.dot(Js2))
            Ms = Cs + Gs + Js
            rhss = Cs.dot(x[idx1] - xtold[idx1]) + Gs.dot(x[idx1]) + Txsp + Ts
            dxs1 = np.linalg.solve(Ms, -rhss) 
            dx0 = np.zeros((mna_size, 1))
            dx0[idx1] = dxs1
            rhss2 = G21.dot(x1) + G22.dot(x2) + Tx[idx2] + T[idx2] + M21.dot(dxs1)
            dxs2 = np.linalg.solve(M22, -rhss2) 
            dx0[idx2] = dxs2
##            rhss2 = rhss[idx2]
##            dxs2 = np.linalg.solve(Ms22, -rhss2)
#            Jp11 = Jp[idx1, :][:, idx1]
#            Jp12 = Jp[idx1, :][:, idx2]
#            Jp21 = Jp[idx2, :][:, idx1]
#            Txsp = Txp[idx1] - G12.dot(G22inv.dot(Txp[idx2]))
#            rhss1 = Cs.dot(x[idx1] - xtold[idx1]) + Gs.dot(x[idx1]) + Txsp + Ts
#            Jsp1 = Jp11.toarray() - G12.dot(G22inv.dot(Jp21.toarray()))
#            Msp11 = Cs + Gs + Jsp1
#            dxs1 = np.linalg.solve(Msp11, -rhss1)
#            xp[idx1] += dxs1
#            dxsnew = np.zeros((mna_size, 1))
#            dxsnew[idx1] = dxs1
#            dxsnew[idx2] = dxs2
            
            As = -Csinv.dot(Gs)
            expAs = sp.linalg.expm(As)
            phiAs = (expAs - np.eye(n1)).dot(-np.linalg.inv(Gs))
            Mse = np.eye(n1) + phiAs.dot(Js)
            v1 = -(Txsp + Ts)
            phi1 = phiAs.dot(v1)
            phi0 = expAs.dot(xtold[idx1])
            rhse = x1 - (phi0 + phi1)
            dxse1 = np.linalg.solve(Mse, -rhse) 
            dx = np.zeros((mna_size, 1))
            dx[idx1] = dxse1
            rhss2 = G21.dot(x1) + G22.dot(x2) + Tx[idx2] + T[idx2] + M21.dot(dxse1)
            dxse2 = np.linalg.solve(M22, -rhss2) 
            dx[idx2] = dxse2
#            As = -Csinv.dot(G11)
#            del Gs
            J2 = np.array([[0, 1], [0, 0]])
#            W1 = W[idx1, :]
#            W2 = W[idx2, :]
#            Ws = W1 - G12.dot(G22inv.dot(W2))
            
            M11 = np.eye(n1) + tmp.dot(J11.toarray())
            M12 = tmp.dot(G12 + J12.toarray())
            M = np.block([[M11, M12],[G21 + J21.toarray(), G22 + J22.toarray()]])
            
            
            rhs1 = x1 - (phi0 + phi1)
            rhs2 = G21.dot(x1) + G22.dot(x2) + Tx[idx2] + T[idx2]
            rhs = np.zeros((mna_size, 1))
            rhs[idx1] = rhs1
            rhs[idx2] = rhs2
            dx = np.linalg.solve(M, -rhs)
            idx = 3247
            v = Ws[:, 1]
            gv = np.linalg.solve(Gs, v)
            tmp1 = expAs.dot(gv)
#            Ast = np.block([[As, Csinv.dot(Ws)], [np.zeros((2, len(idx1))), J2 * dt]])
#            eAst = sp.linalg.expm(Ast)
#            vt = np.block([[xtold[idx1]], [np.array([[0], [1]])]])
#            y = eAst.dot(vt)
#        w = np.linalg.inv(np.eye(5,5) - gamma * Ast).dot(vt)