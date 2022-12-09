import polyscope as ps
from pathos.multiprocessing import ProcessPool as Pool

import model

def test(i):
    m = model.Model()
    v = m.step(200000)
    return v

with Pool(8) as p:
    p.map(test, [1] * 8)

for i in range(8):
    test(1)

# ps.init()
# n = ps.register_curve_network('c', v, e)
# ps.show()

# v = m.step(200000)
# n.update_node_positions(v)
# ps.show()




