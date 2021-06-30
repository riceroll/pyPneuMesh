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
run utils/visualizer.py ./output/pillBugIn_626-0:22:29.json
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

