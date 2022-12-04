import taichi as ti

ti.init(arch=ti.cpu)

n = 128
# x is an n x n field consisting of 3D floating-point vectors
# representing the mass points' positions
x = ti.Vector.field(3, dtype=float, shape=(n, n))
# v is an n x n field consisting of 3D floating-point vectors
# representing the mass points' velocities
v = ti.Vector.field(3, dtype=float, shape=(n, n))


@ti.kernel
def test():
    for i, j in x:
        print(i, j)
        
test()