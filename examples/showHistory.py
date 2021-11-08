import sys
from utils.GA import GeneticAlgorithm, loadHistory

if len(sys.argv) > 1:
    hs_dir = sys.argv[1]    # directory to the history file .hs
else:
    hs_dir = '/Users/Roll/Desktop/pyPneuMesh/output/GA_117-5:27:38/g1455_1.52,1.00,-1.97,-1.95.hs'

history = loadHistory(hs_dir)
history.plot(False)


