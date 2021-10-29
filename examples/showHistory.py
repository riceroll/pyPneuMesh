import sys
from utils.GA import GeneticAlgorithm, loadHistory, plot

if len(sys.argv) > 1:
    hs_dir = sys.argv[1]    # directory to the history file .hs
else:
    hs_dir = './output/_GA_72-4-14-36_pillbugnodir/g2495_0.76,1.00.hs'

plot(loadHistory(hs_dir))


