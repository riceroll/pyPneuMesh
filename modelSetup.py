
class ModelSetup:
    def __init__(self):
        self.inFileDir = None          # str, the directory of the input json file
        self.edgeChannel = None     # np.array, int, [nEdge, ] : which channel each edge belongs to
        self.maxContractionLevel = None  # np.array, int, [nEdge, ] : which contraction level the edge is, 0 ~ 3
        self.tasks = []     # component: {'actions': np.array, nChannel * nActions,
                                        # 'target': list, target function, 'score': float, score of the target function}
    
    def fitness(self):
        fitness = 0
        for task in self.tasks:
            actions = task['actions']
            taskFitness = 0
            for target in task['targets']:
                taskFitness += target(self.inFileDir, self.edgeChannel, self.maxContractionLevel, actions)
            fitness += taskFitness
        return fitness
        