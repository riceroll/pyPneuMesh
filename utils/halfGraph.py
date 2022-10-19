import numpy as np


class HalfGraph(object):
    def __init__(self):
        self.ins_o = []  # nn, original indices of nodes in a halfgraph
        self.edges = []  # indices of two incident nodes, ne x 2, ne is the number of edges in a halfgraph
        self.ies_o = []  # ne, original indices of edges in a halfgraph
        self.channels = []  # ne, indices of channels
        self.contractions = []  # ne, int value of contractions
        self.esOnMirror = []  # ne, bool, if the edge in on the mirror plane

    def add_edges_from(self, esInfo):
        # input:
        # esInfo: [(iv0, iv1, {'ie': , 'channel': , 'contraction': })]

        ne = len(esInfo)
        self.ies_o = np.zeros(ne, dtype=int)
        self.edges = np.zeros([ne, 2], dtype=int)
        self.channels = np.zeros(ne, dtype=int)
        self.contractions = np.zeros(ne, dtype=int)
        self.esOnMirror = np.zeros(ne, dtype=bool)

        for i, eInfo in enumerate(esInfo):
            ie = eInfo[2]['ie']  # original index of the edge
            self.ies_o[i] = ie

            self.ins_o.append(eInfo[0])
            self.ins_o.append(eInfo[1])
            self.channels[i] = int(eInfo[2]['channel'])
            self.contractions[i] = int(eInfo[2]['contraction'])
            self.esOnMirror[i] = bool(eInfo[2]['onMirror'])

        self.ins_o = sorted(list(set(self.ins_o)))

        for i, eInfo in enumerate(esInfo):
            in0 = np.where(self.ins_o == eInfo[0])[0][0]
            in1 = np.where(self.ins_o == eInfo[1])[0][0]
            self.edges[i, 0] = in0
            self.edges[i, 1] = in1

    def iesIncidentMirror(self):
        # indices of edges on / connecting to the mirror plane
        ies = np.arange(len(self.edges))[self.esOnMirror.copy()]
        ins_onMirror = self.incidentNodes(ies)
        ies_IncidentMirror = self.incidentEdges(ins_onMirror)
        return ies_IncidentMirror

    def iesNotMirror(self):
        ies = np.arange(len(self.edges))[~self.esOnMirror]
        return ies

    def incidentNodes(self, ies):
        # get the incident nodes of a set of edges
        ins = np.array(list(set(self.edges[ies].reshape(-1))))
        return ins

    def incidentEdges(self, ins):
        # get the incident edges of a set of noes
        isin = np.isin(self.edges, ins)
        ifEdgesIncident = isin[:, 0] + isin[:, 1]
        ies = np.arange(len(self.edges))[ifEdgesIncident]
        return ies

    def iesAroundChannel(self, ic, unassigned=True):
        # breakpoint()
        # get ies incident but not belonged to the channel ic and not assigned
        boolEsChannel = self.channels == ic  # bool, es in the channel
        ins = self.incidentNodes(np.arange(len(self.edges))[boolEsChannel])

        isin = np.isin(self.edges, ins)
        boolEsIncidentChannel = isin[:, 0] + isin[:, 1]
        boolEsAroundChannel = boolEsIncidentChannel * (~boolEsChannel)
        boolEsUnassigned = self.channels == -1
        if unassigned:
            bools = boolEsAroundChannel * boolEsUnassigned
        else:
            bools = boolEsAroundChannel
        if True in bools:
            return np.arange(len(self.edges))[bools]
        else:
            return None

    def channelConnected(self, ic):
        iesUnvisited = np.arange(len(self.edges))[self.channels == ic].tolist()  # ies of the channel

        if len(iesUnvisited) == 0:  # no graph exist at all
            return False

        ie = iesUnvisited.pop()

        queue = [ie]

        while len(queue) != 0:
            ie = queue.pop(0)

            ins = self.incidentNodes([ie])
            iesAdjacent = self.incidentEdges(ins)

            for ieAdjacent in iesAdjacent:
                if ieAdjacent in iesUnvisited:
                    iesUnvisited.remove(ieAdjacent)
                    queue.append(ieAdjacent)

        if len(iesUnvisited) != 0:
            return False
        else:
            return True
