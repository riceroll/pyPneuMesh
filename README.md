## TrussBot
The simulation and optimization framework for trussBot.

### Dependency
- numpy - 1.20.0
- pathos - 0.2.5
- open3d - 0.10.0.0

### Examples
To optimize a fox to move towards +x direction:
```bash
python examples/train.py --iFile fox --nPop 100 --nGen 1000 --nWorkers 8 --numChannels 3 --numActions 4 --targets moveForward
```

To run a simulation with given file
```bash
python examples/test --iFile fox --testing 1 --visualize 1
```

