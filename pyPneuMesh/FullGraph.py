import numpy as np

from pyPneuMesh.Model import Model

class FullGraph(object):
    def __init__(self, model):
        # self.model = model
        #
        # self.numChannels = model.numChannel
        # self.nV = model.nV
        # self.nE = model.nE
        #
        # self.ivsSub = ivsSub  # mapping from original vertex indices to new indices of the subGraph
        # self.iesSub = iesSub  # mapping from original edge indices to new indices of the subGraph
        # self.esSub = esSub  # ne x 2, edges of the subGraph
        # self.channels = np.ones(self.nE) * -1  # ne, indices of channels of edges
        # self.contractions = np.zeros(
        #     self.nE)  # ne, int value of contractions, Model.contractionLevels number of types of contractions
        #
        # self.esChannels = []
        #
        # self.incM = np.zeros([self.nV, self.nE])  # vertex-edge incidence matrix
        # for ie, e in enumerate(self.esSub):
        #     self.incM[e[0], ie] = 1
        #     self.incM[e[1], ie] = 1
        #
        # self.ieAdjList = []  # nE x X, each row includes indices of adjacent edges of ie, np.array
        # for ie in range(self.nE):
        #     iv0 = self.esSub[ie, 0]
        #     iv1 = self.esSub[ie, 1]
        #     iesConnected0 = np.where(self.incM[iv0] == 1)[0]
        #     iesConnected1 = np.where(self.incM[iv1] == 1)[0]
        #     iesConnected = np.concatenate([iesConnected0, iesConnected1])
        #     self.ieAdjList.append(iesConnected)
        #
        # self.init()
        pass

    def init(self):
        self.contractions = np.random.randint(0, Model.contractionLevels, self.contractions.shape)

        # randomly choose numChannel edges and assign channels
        dice = np.arange(self.nE)
        np.random.shuffle(dice)
        ies = dice[:self.numChannels]
        for ic, ie in enumerate(ies):
            self.channels[ie] = ic

        # grow channels to fill the entire graph
        while (self.channels == -1).any():
            iesToGrow = []
            for ie in range(self.nE):
                iesConnected = self.ieAdjList[ie]
                if (self.channels[iesConnected] == -1).any():
                    iesToGrow.append(ie)

            ie = np.random.choice(iesToGrow)
            iesConnected = self.ieAdjList[ie]
            np.random.shuffle(iesConnected)
            for ieConnected in iesConnected:
                if self.channels[ieConnected] == -1:
                    self.channels[ieConnected] = self.channels[ie]

    def channelConnected(self, ic):
        # check if channel ic is interconnected
        nEic = (self.channels == ic).sum()  # number of edges of channel ic

        for ie in range(self.nE):
            if self.channels[ie] == ic:
                break

        queue = [ie]  # ies in the queue
        visited = set()  # ies visited

        while queue:
            ie = queue.pop(0)
            visited.add(ie)

            iv0 = self.esSub[ie, 0]
            iv1 = self.esSub[ie, 1]
            iesConnected0 = np.where(self.incM[iv0] == 1)[0]
            iesConnected1 = np.where(self.incM[iv1] == 1)[0]
            iesConnected = np.concatenate([iesConnected0, iesConnected1])
            for ie in iesConnected:
                if ie not in visited and self.channels[ie] == ic:
                    queue.append(ie)

        return nEic == len(visited)

    def mutate(self):
        # mutate one digit of contractions and one edge channel
        self.contractions[np.random.choice(len(self.contractions))] = np.random.randint(Model.contractionLevels)

        ies = np.arange(self.nE)
        np.random.shuffle(ies)
        for ie in ies:
            iesConnected = self.ieAdjList[ie]

            icOld = self.channels[ie]
            icsConnected = self.channels[iesConnected].tolist()
            icsConnected = set(icsConnected)
            icsConnected.remove(icOld)
            icsConnected = np.array(list(icsConnected))
            if len(icsConnected):
                np.random.shuffle(icsConnected)

            for icNew in icsConnected:
                self.channels[ie] = icNew  # change the channel of the edge
                if self.channelConnected(icOld):  # if the changed channel is still connected
                    return True  # mutation finished
                else:
                    self.channels[ie] = icOld  # revert channel change
        print('mutation failed')
        return False
