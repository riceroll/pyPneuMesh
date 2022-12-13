import time
import json

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
        self.wContractingOnEdge = jointGeneratorParam['wContractingOnEdge']
        self.wRepelling = jointGeneratorParam['wRepelling']
        self.wContractingAtIntersection = jointGeneratorParam['wContractingAtIntersection']
        self.h = jointGeneratorParam['h']
        self.numSteps = jointGeneratorParam['numSteps']
    
    def getJointGeneratorParam(self):
        jointGeneratorParam = {
            'sphereRadius': self.sphereRadius,
            'spherePadding': self.spherePadding,
            'channelRadius': self.channelRadius,
            'channelMargin': self.channelMargin,
        
            'numPointsOnEdge': self.numPointsOnEdge,
            'wContractingOnEdge': self.wContractingOnEdge,
            'wContractingAtIntersection': self.wContractingAtIntersection,
            'wRepelling': self.wRepelling,
            'h': self.h,
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
        vE = []
        vEIntersection = []
        V = []
        
        numV = len(self.model.v0)
        numV = 1
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            vtPvEevEi = list(tqdm.tqdm(p.imap(self.generateJoint, np.arange(numV)), total=numV))
        
        # vtPvE = []
        # for i in range(len(self.model.v0)):
        #     tP, E = self.generateJoint(i)
        #     vtPvE.append([tP, E])
        
        for iv, tPEeEi in enumerate(vtPvEevEi):
            vtP.append(tPEeEi[0])
            vE.append(tPEeEi[1])
            vEIntersection.append(tPEeEi[2])
            V.append(self.model.v0[iv])
        
        return vtP, vE, vEIntersection, V
    
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
    
    def generateJointsAndExport(self, folderDir, name):
        folderPath = pathlib.Path(folderDir)
        outDir = folderPath.joinpath('{}.jointsdict'.format(name))
        vtP, vE, vEIntersection, V = self.generateJoints()
        
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
            
        esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esRepellingEnergy, psSphereRepellingEnergy, psAnchorConstraint = self.__getConstraints(ies, ipsOnEdges)
        
        tP = [P]
        for i in tqdm.tqdm(range(self.numSteps)):
            e = self.__getEnergy(tP[-1], ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esRepellingEnergy, psSphereRepellingEnergy)
            
            gP = torch.autograd.grad(e, tP[-1])[0]
            gP[psAnchorConstraint] *= 0
            PNext = tP[-1] - self.h * gP
            tP.append(PNext)
        
        tP = [P.detach().numpy() for P in tP]
        
        return tP, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy
    
    def __getConstraints(self, ies, ipsOnEdges):
        esContractingOnEdgeEnergy = []
        esContractingAtIntersectionEnergy = []
        esChannelRepellingEnergy = []
        psSphereRepellingEnergy = []
        psAnchorConstraint = []
    
        for i in range(len(ies)):
            for ipOnEdge in range(self.numPointsOnEdge):
                ip = ipsOnEdges[i][ipOnEdge]
                if ipOnEdge != self.numPointsOnEdge - 1:  # except for the outer most point
                    ipNext = ipsOnEdges[i][ipOnEdge + 1]
                    esContractingOnEdgeEnergy.append([ip, ipNext])  # contracting energy
                    
                    psSphereRepellingEnergy.append(ip)
                else:
                    psAnchorConstraint.append(ip)  # anchor energy
        
            for ipOnEdge in range(self.numPointsOnEdge):
                ip = ipsOnEdges[i][ipOnEdge]
            
                for ie2 in range(len(ies)):  # all other edges
                    if ie2 == i:
                        continue
                    for ipOnEdge2 in range(self.numPointsOnEdge):  # all points on other edges
                        ip2 = ipsOnEdges[i][ipOnEdge2]
                        esChannelRepellingEnergy.append([ip, ip2])  # repelling energy
    
        for i0 in range(len(ies)):
            for i1 in range(len(ies) - i0 - 1):
                i1 = i1 + i0 + 1
                
                if self.model.edgeChannel[i0] == self.model.edgeChannel[i1]:  # same channel
                    esContractingAtIntersectionEnergy.append(
                        [ipsOnEdges[i0][0], ipsOnEdges[i1][0]])  # contraction between the two center points
                else:
                    pass
            
        esContractingOnEdgeEnergy = np.array(esContractingOnEdgeEnergy)
        esContractingAtIntersectionEnergy = np.array(esContractingAtIntersectionEnergy)
        esChannelRepellingEnergy = np.array(esChannelRepellingEnergy)
        psSphereRepellingEnergy = np.array(psSphereRepellingEnergy)
        psAnchorConstraint = np.array(psAnchorConstraint)
        
        return esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esChannelRepellingEnergy, psSphereRepellingEnergy, psAnchorConstraint

    def __getEnergy(self, ps, ps0, esContractingOnEdgeEnergy, esContractingAtIntersectionEnergy, esRepellingEnergy, psSphereRepellingEnergy):
        cee = self.__contractingEnergy(ps, esContractingOnEdgeEnergy)
        cie = self.__contractingEnergy(ps, esContractingAtIntersectionEnergy)
        re = self.__channelRepellingEnergy(ps, esRepellingEnergy, self.channelMargin)
        se = self.__sphereRepellingEnergy(ps, psSphereRepellingEnergy, self.sphereRadius, self.spherePadding)
        return self.wContractingOnEdge * cee + self.wContractingAtIntersection * cie + self.wRepelling * re

    @staticmethod
    def __contractingEnergy(ps, es):
        if len(es) == 0:
            return 0
        ps0 = ps[es[:, 0]]
        ps1 = ps[es[:, 1]]
        ce = torch.norm(ps0 - ps1, dim=1).sum()
        return ce
        
    @staticmethod
    def __channelRepellingEnergy(ps, es, margin):
        ps0 = ps[es[:, 0]]
        ps1 = ps[es[:, 1]]
        d = torch.norm(ps0 - ps1, dim=1)
        re = ( (d - margin) ** 2)[d < margin].sum()
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
        pOut = vecUnit * self.sphereRadius + v
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
    
    def animate(self, Ps, E, speed=1.0):
        # animate the generation processimport polyscope as ps
        try:
            ps.init()
        except:
            pass
        ps.set_up_dir('z_up')
        
        P = Ps[0]
        cs = ps.register_curve_network('channels', P, E)
    
        t0 = time.time()
        
        def callback():
            t = (time.time() - t0) * speed
            if t // self.h < len(Ps):
                iStep = int(t // self.h)
            else:
                iStep = len(Ps) - 1
                
            cs.update_node_positions(Ps[iStep])
            

        ps.set_user_callback(callback)
        ps.show()
    
        
    
    
        
