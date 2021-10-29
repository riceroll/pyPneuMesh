# TrussBot
The simulation and multi-objective optimization framework for trussBot.

### Installation
```bash
pip install -r requirements.txt
```

### Examples
1. To optimize a pill-bug to move forward and stay in one direction,
```bash
ipython
run examples/pillBugForward.py
```

2. To visualize a result,
```bash
ipython
run utils/visualizer.py ./output/_GA_72-4-14-36_pillbugnodir/g2495_0.76,1.00/3.63,0.98
```
or 
```bash
ipython
run examples/showActions.py  ./output/_GA_72-4-14-36_pillbugnodir/g2495_0.76,1.00/3.63,0.98
```

3. To visualize the training graph from a history,
```bash
ipython
run examples/showHistory.py  ./output/_GA_630-2-27-41_pillbug/g315_1.54,1.00,0.99,0.99.hs
```


### Testing

```bash
ipython
run test/test.py [all] [<moduleName>] [plot] [unmute]  
```
`all`: test all functions \
`<moduleName>`: test specific testing module \
`plot`: enable plotting \
`unmute`: enable printing info

