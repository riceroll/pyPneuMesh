import time
import json

import scipy.optimize as optimize
import tqdm
import pathlib
import numpy as np
import torch
import polyscope as ps

import multiprocessing

class JointGenerator(object):
    def __init__(self, model, jointGeneratorParam):
        self.model = model
        
        self.sphereRadius = jointGeneratorParam['sphereRadius']
        self.spherePadding = jointGeneratorParam['spherePadding']
        self.channelRadius = jointGeneratorParam['channelRadius']
        self.channelMargin = jointGeneratorParam['channelMargin']
        
        self.numPointsOnEdge = jointGeneratorParam['numPointsOnEdge']
        self.wContracting = jointGeneratorParam['wContracting']
        self.wRepelling = jointGeneratorParam['wRepelling']
        self.wRepellingMultiplier = jointGeneratorParam['wRepellingMultiplier']
        self.seqLengthMultiplier = jointGeneratorParam['seqLengthMultiplier']
        self.h = jointGeneratorParam['h']
        self.beta = jointGeneratorParam['beta']     # momentum
        self.decay = jointGeneratorParam['decay']   # gradient decay
        self.numSteps = jointGeneratorParam['numSteps']

        self.wRepellingChanging = self.wRepelling
        self.segLength = (self.sphereRadius - self.spherePadding) / (self.numPointsOnEdge - 1)
    
    def getJointGeneratorParam(self):
        jointGeneratorParam = {
            'sphereRadius': self.sphereRadius,
            'spherePadding': self.spherePadding,
            'channelRadius': self.channelRadius,
            'channelMargin': self.channelMargin,
        
            'numPointsOnEdge': self.numPointsOnEdge,
            'wContracting': self.wContracting,
            'wRepelling': self.wRepelling,
            'wRepellingMultiplier': self.wRepellingMultiplier,
            'seqLengthMultiplier': self.seqLengthMultiplier,
            'h': self.h,
            'beta': self.beta,
            'decay': self.decay,
            'numSteps': self.numSteps
        }
        
        return jointGeneratorParam
        
    def save(self, folderDir, name):
        jointGeneratorParam = self.getJointGeneratorParam()
        folderPath = pathlib.Path(folderDir)
        np.save(str(folderPath.joinpath('{}.jointgeneratorparam'.format(name))), jointGeneratorParam)
    
    def generateJoints(self):
        # output
        #   vtP: [iVertex, [iStep, iPoint, xyz] ]
        #   vE: [iVertex, iPointInEdge(0/1)]
        # positions of points represent the absolute positions of the channels at every time step
        
        vtP = []
        vEOnEdge = []
        vEIntersection = []
        V = []
        
        numV = len(self.model.v0)
        # numV = 1
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            vtPvEevEi = list(tqdm.tqdm(p.imap(self.generateJoint, np.arange(0, numV)), total=numV))
        
        # vtPvEevEi = []
        # for i in range(numV):
            # tP, Ee, Ei = self.generateJoint(i)
            # vtPvEevEi.append([tP, Ee, Ei])
        
        for iv, tPEeEi in enumerate(vtPvEevEi):
            vtP.append(tPEeEi[0])
            vEOnEdge.append(tPEeEi[1])
            vEIntersection.append(tPEeEi[2])
            V.append(self.model.v0[iv])
        
        return vtP, vEOnEdge, vEIntersection, V
    
    def generateJointsChannels(self, vtP, vE, vEIntersection, V):
        # returns
        #   vChannelP: [idVertex, idChannel, iPoint, xyz]
        #   V: [idVertex, xyz]
        
        vP = [tP[-1] for tP in vtP]
        vE = vE
        vEIntersection = vEIntersection
        vChannelP = []
        
        for iv in range(len(V)):
            P = vP[iv].tolist()
            E = vE[iv].tolist()
            
            ChannelP = []      # [iChannel, iPoint]
            while len(E):
                channelE = [E.pop()]   # [iEdge]  # a list of order edges
                while True:     # keep merging into channelIp
                    merged = False
                    iPLeft = channelE[0][0]
                    iPRight = channelE[-1][-1]
                    
                    for ie1, e1 in enumerate(E):
                        if e1[0] == iPLeft:
                            channelE = [e1[::-1]] + channelE
                        elif e1[0] == iPRight:
                            channelE = channelE + [e1]
                        elif e1[1] == iPLeft:
                            channelE = [e1] + channelE
                        elif e1[1] == iPRight:
                            channelE = channelE + [e1[::-1]]
                        else:
                            continue    # this e1 not connected
                        
                        # connected
                        del E[ie1]
                        merged = True
                        break
                    
                    if not merged:  # finished a channelE
                        channelP = [P[e[0]] for e in channelE]     # convert to channelP
                        channelP.append(P[channelE[-1][1]])
                        ChannelP.append(channelP)       # store into ChannelP
                        break
                        
            vChannelP.append(ChannelP)
            
        return vChannelP, V
        
    def export(self, vChannelP, V, outDir):
        vChannelP = [[[[p[0], p[1], p[2]] for p in POfTheChannel] for POfTheChannel in channelP] for channelP in vChannelP]
        V = [v.tolist() for v in V]
        
        jointsDict = {
            'vChannelP': vChannelP,
            'V': V,
            'channelRadius': self.channelRadius,
            'sphereRadius': self.sphereRadius
        }

        js = json.dumps(jointsDict)
        
        with open(outDir, 'w') as oFile:
            oFile.write(js)
    
    def exportJoints(self, folderDir, name, vtP, vE, vEIntersection, V):
        folderPath = pathlib.Path(folderDir)
        outDir = folderPath.joinpath('{}.jointsdict'.format(name))
        # vtP, vE, vEIntersection, V = self.generateJoints()
        
        vChannelP, V = self.generateJointsChannels(vtP, vE, vEIntersection, V)
        
        self.export(vChannelP, V, outDir)
        
    def generateJoint(self, iv):
        # output tP: [iTime, iPoint, dimension]
        #   and E: [iEdgeContracting, iPOnEdge(0/1)]
        # positions of points represent the relative positions of the channels in a joint at every time step
        
        v = self.model.v0[iv]
        ies0 = np.arange(len(self.model.e))[self.model.e[:, 0] == iv]
        ies1 = np.arange(len(self.model.e))[self.model.e[:, 1] == iv]
        ies = np.hstack([ies0, ies1])     # all ids of edges connecting to v
        
        P = np.vstack([
            self.__generatePointsOnEdge(ie, iv)
            for ie in ies
        ])   # (ne x numPointsOnEdge) * 3
        ps0 = torch.tensor(P, dtype=torch.float64)
        P = ps0.clone().requires_grad_()
        
        ipsOnEdges = []     # ne x numPointsOnEdge [0, 1, 2], [3, 4, 5,], ...
        for i in ies:
            tmp = np.arange(self.numPointsOnEdge)
            tmp += len(ipsOnEdges * self.numPointsOnEdge)
            ipsOnEdges.append(tmp)
            
        esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esChannelRepellingEnergy, psSphereRepellingEnergy, psAnchorConstraint, pssIntersectionConstraint = self.__getConstraints(ies, ipsOnEdges)


        # # scipy minimize
        # P = P.detach().numpy()
        # tP = [P.copy()]
        #
        # shape0 = P.shape[0]
        # shape1 = P.shape[1]
        #
        # def func(PNumpy):
        #     P = torch.tensor(PNumpy)
        #     P = P.reshape([shape0, shape1])
        #     e = self.__getEnergy(P, ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy,
        #                          esRepellingEnergy, psSphereRepellingEnergy)
        #     return e.numpy()
        #
        # def jac(PNumpy):
        #     P = torch.tensor(PNumpy, requires_grad=True)
        #     P = P.reshape([shape0, shape1])
        #     e = self.__getEnergy(P, ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy,
        #                          esRepellingEnergy, psSphereRepellingEnergy)
        #
        #     gP = torch.autograd.grad(e, P)[0]
        #     gP[psAnchorConstraint] *= 0
        #
        #     return gP.detach().numpy().reshape(-1)
        #
        # def callback(x):
        #     # pass
        #     print(func(x))
        #     # tP.append(x.copy().reshape(shape0, shape1))
        #
        # # print('ehhe', func(P))
        # res = optimize.minimize(func, P.reshape(-1), jac=jac, method="SLSQP", callback=callback, options={'disp':True, 'ftol': 0, 'eps': 1e-5, 'maxiter': 500})
        # tP.append(res.x.reshape([shape0, shape1]))
        # # print(func(tP[-1]))


        # gradient descent
        tP = [P]
        for i in tqdm.tqdm(range(self.numSteps)):
            # print(esChannelRepellingEnergy)
            
            e = self.__getEnergy(tP[-1], ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy,
                                 esChannelRepellingEnergy, psSphereRepellingEnergy)
            
            # print(e.item())
            gP = torch.autograd.grad(e, tP[-1])[0]
            
            # self.wContracting *= 0.999
            self.segLength *= self.seqLengthMultiplier
            self.wRepellingChanging *= self.wRepellingMultiplier
            
            if i == 0:
                gPNoise = (torch.rand(gP.shape) -0.5) * gP.abs().mean() * 1
                gP += gPNoise
            
            gP[psAnchorConstraint] *= 0
            
            momentum = 0 if len(tP) < 2 else tP[-1] - tP[-2]

            # self.h *= np.sqrt(1.0 / (1 + self.decay * i))
            PNext = tP[-1] - self.h * gP + self.beta * momentum
            
            for key in pssIntersectionConstraint:
                PNext[np.array(list(pssIntersectionConstraint[key]))] = PNext[np.array(list(pssIntersectionConstraint[key]))].mean(0)
            
            tP.append(PNext)

        tP = [P.detach().numpy() for P in tP]

        return tP, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy
    
    def __getConstraints(self, ies, ipsOnEdges):
        esContractingOnEdgeEnergy = []
        esContractingAtIntersectionEnergy = []
        esChannelRepellingEnergy = []
        psSphereRepellingEnergy = []
        psAnchorConstraint = []
        pssIntersectionConstraint = {}
    
        for i0 in range(len(ies)):
            for ipOnEdge0 in range(self.numPointsOnEdge):
                ip0 = ipsOnEdges[i0][ipOnEdge0]
                if ipOnEdge0 != self.numPointsOnEdge - 1:  # except for the outer most point
                    ipNext = ipsOnEdges[i0][ipOnEdge0 + 1]
                    esContractingOnEdgeEnergy.append([ip0, ipNext])  # contracting energy
                    
                    psSphereRepellingEnergy.append(ip0)
                else:
                    psAnchorConstraint.append(ip0)  # anchor energy
        
            for ipOnEdge0 in range(self.numPointsOnEdge):
                ip0 = ipsOnEdges[i0][ipOnEdge0]
            
                for i1 in range(len(ies)):  # all other edges
                    if i1 == i0:
                        continue
                    if self.model.edgeChannel[ies[i0]] == self.model.edgeChannel[ies[i1]]:    # same channel
                        continue
                        
                    for ipOnEdge1 in range(self.numPointsOnEdge):  # all points on other edges
                        ip1 = ipsOnEdges[i1][ipOnEdge1]
                        esChannelRepellingEnergy.append([ip0, ip1])  # repelling energy
    
        for i0 in range(len(ies)):
            for i1 in range(len(ies) - i0 - 1):
                i1 = i1 + i0 + 1
                
                if self.model.edgeChannel[ies[i0]] == self.model.edgeChannel[ies[i1]]:  # same channel
                    iChannel = self.model.edgeChannel[ies[i0]]
                    if iChannel not in pssIntersectionConstraint:
                        pssIntersectionConstraint[iChannel] = set()

                    pssIntersectionConstraint[iChannel].add(ipsOnEdges[i1][0])
                    pssIntersectionConstraint[iChannel].add(ipsOnEdges[i0][0])
                    
                    esContractingAtIntersectionEnergy.append(
                        [ipsOnEdges[i0][0], ipsOnEdges[i1][0]])  # contraction between the two center points
                else:
                    pass
            
        esContractingOnEdgeEnergy = np.array(esContractingOnEdgeEnergy)
        esContractingAtIntersectionEnergy = np.array(esContractingAtIntersectionEnergy)
        esChannelRepellingEnergy = np.array(esChannelRepellingEnergy)
        psSphereRepellingEnergy = np.array(psSphereRepellingEnergy)
        psAnchorConstraint = np.array(psAnchorConstraint)
        
        
        # breakpoint()
        
        return esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esChannelRepellingEnergy, psSphereRepellingEnergy, psAnchorConstraint, pssIntersectionConstraint

    def __getEnergy(self, ps, ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esRepellingEnergy, psSphereRepellingEnergy):
        cee = self.__contractingEnergy(ps, esContractingOnEdgeEnergy)
        # cie = self.__contractingEnergy(ps, esContractingAtIntersectionEnergy)
        cre = self.__channelRepellingEnergy(ps, esRepellingEnergy, self.channelRadius, self.channelMargin)
        # print('re', re.item())
        sre = self.__sphereRepellingEnergy(ps, psSphereRepellingEnergy, self.sphereRadius, self.spherePadding)
        # print(esRepellingEnergy)
        
        # print(cee.item(), cre.item(), sre.item())
        
        return self.wContracting * cee + self.wRepelling * cre + self.wRepelling * sre

    def __contractingEnergy(self, ps, es):
        if len(es) == 0:
            return 0
        ps0 = ps[es[:, 0]]
        ps1 = ps[es[:, 1]]
        ce = ((torch.norm(ps0 - ps1, dim=1) - self.segLength) ** 2).sum()
        return ce
        
    @staticmethod
    def __channelRepellingEnergy(ps, es, radius, margin):
        if len(es) == 0:
            return 0
        ps0 = ps[es[:, 0]]
        ps1 = ps[es[:, 1]]
        d = torch.norm(ps0 - ps1, dim=1)
        re = - (-radius * 2 - margin + d)[ d < radius * 2 + margin].sum()
        # print(re.item())
        
        return re
    
    @staticmethod
    def __sphereRepellingEnergy(ps, ips, radius, padding):
        d = torch.norm(ps, dim=1)
        se = ( (d - (radius-padding) )**2)[d > radius - padding].sum()
        
        return se
    
    def __generatePointsOnEdge(self, ie, iv):
        # a numPointsOnEdge points, the first point is from the center, the last is the end of the port
        
        if self.model.e[ie][0] == iv:
            ivNeighbor = self.model.e[ie][1]
        else:
            ivNeighbor = self.model.e[ie][0]
            
        vNeighbor = self.model.v0[ivNeighbor] - self.model.v0[iv]
        v = self.model.v0[iv] - self.model.v0[iv]
        
        vecUnit = (vNeighbor - v) / np.linalg.norm(vNeighbor - v)   # from v to vNeighbor
        pOut = vecUnit * (self.sphereRadius - self.spherePadding) + v
        pCenter = v.copy()
        points = []
        
        for ip in range(self.numPointsOnEdge):
            tOut = ip / (self.numPointsOnEdge - 1)
            tCenter = 1 - tOut
            p = tOut * pOut + tCenter * pCenter
            
            points.append(p)
        points = np.array(points)
        
        assert(points.shape == (self.numPointsOnEdge, 3))
        
        return points
    
    def animate(self, Ps, E, EIntersection, speed=1.0):
        # animate the generation process import polyscope as ps
        try:
            ps.init()
        except:
            pass
        ps.set_up_dir('z_up')
        
        P = Ps[0]
        cs = ps.register_curve_network('channels', P, E)

        csIntersection = ps.register_curve_network('channelsIntersection', P, EIntersection)
        pp = ps.register_point_cloud('p', P)
        
        t0 = time.time()
        
        def callback():
            t = (time.time() - t0) * speed
            
            if t // self.h < len(Ps):
                iStep = int(t // self.h)
            else:
                iStep = len(Ps) - 1
            print(iStep)
            cs.update_node_positions(Ps[iStep])
            csIntersection.update_node_positions(Ps[iStep])
            pp.update_point_positions(Ps[iStep])
            

        ps.set_user_callback(callback)
        ps.show()

