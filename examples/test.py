from utils.GA import GeneticAlgorithm
import argparse
import multiprocessing

from utils.moo import MOO
import torch
from utils.model import Model
import time

import numpy as np

import taichi as ti
ti.init(arch=ti.cpu)


model = Model('./data/config.json')
model.load('./data/table2.json')
model.updateL()
model.lMax = model.l * 0.8


# t0 = time.time()
# for i in range(10000):
#     model.step()
# print(time.time() - t0)
#
# t0 = time.time()
# for i in range(10000):
#     model.step2()
# print(time.time() - t0)
#
#
# v = torch.tensor(model.v, dtype=torch.float32, requires_grad=True)
# e = torch.tensor(model.e, dtype=torch.long)
#
# t0 = time.time()
# for i in range(10000):
#     l0 = (v[e[:, 0]] - v[e[:, 1]]).norm(dim=1, keepdim=True)
#     # l0.sum().backward()
# dt0 = time.time() - t0
# print(dt0)
#
# v = model.v.copy()
# e = model.e.copy()
#
# t0 = time.time()
# for i in range(10000):
#     l1 = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
# dt1 = time.time() - t0
# print(dt1)
#
# v = ti.Vector.field(3, dtype=ti.float32, shape=(model.v.shape[0], ), needs_grad=True)
# v.from_numpy(model.v)
# e = ti.Vector.field(2, dtype=ti.int32, shape=(model.e.shape[0], ))
# e.from_numpy(model.e)
# l2 = ti.field(dtype=ti.float32, shape=model.e.shape[0])
#
#
# @ti.kernel
# def test():
#     for i in range(10000):
#         for iE in range(e.shape[0]):
#             l2[iE] = (v[e[iE][0]] - v[e[iE][1]]).norm()
#
# test()

#
# @ti.kernel
# def test1():
#     for i in range(10000):
#         with ti.ad.Tape(loss):  # take gradient is not allowed in kernel
#             for iE in range(e.shape[0]):
#                 l2[iE] = (v[e[iE][0]] - v[e[iE][1]]).norm()
#                 loss[None] += l2[i]
#
# test1()

# v = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
# v.fill(1.0)
# loss = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
#
# @ti.kernel
# def test2():
#     ti.loop_config(serialize=True)
#     for i in range(5):
#         loss[None] = 0.5 * v[None] ** 2
#         v[None] -= v.grad[None]
#         # v[None] -= 1
#
# print(v[None], v.grad[None])    # output: 1.0 0.0
# with ti.ad.Tape(loss):
#     test2()
# print(v[None], v.grad[None])    # output: 1.0 5.0
#
# v = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
# v.fill(1.0)
# loss = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
#
# @ti.kernel
# def test3():
#     ti.loop_config(serialize=True)
#     for i in range(5):
#         loss[None] = 0.5 * v[None] ** 2
#         # v[None] -= v.grad[None]
#         v[None] -= 1
#
# print(v[None], v.grad[None])    # 1.0 0.0
# with ti.ad.Tape(loss):
#     test3()
# print(v[None], v.grad[None])    # -4.0 -20.0
#
#
# def test5(n):
#     ti.init(arch=ti.cpu, cpu_max_num_threads=4)
#
#     v = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
#     v_grad = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
#     v_nograd = ti.field(dtype=ti.float32, shape=(), needs_grad=False)
#     v.fill(1.0)
#     v_grad.fill(1.0)
#     v_nograd.fill(1.0)
#     loss = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
#     loss.grad[None] = 1
#
#     @ti.kernel
#     def test4():
#         ti.loop_config(serialize=False, parallelize=4)
#         for i in range(8):
#             for j in range(10000000):
#                 for k in range(10):
#
#                     loss[None] += 0.5 * v[None] ** 2
#                     v[None] = v[None] - v.grad[None]
#                     v_grad[None] = v[None] - v.grad[None]
#                     v_nograd[None] = v[None] - v.grad[None]
#
#     t0 = time.time()
#     test4.grad()
#     print(time.time() - t0)


v = ti.Vector.field(3, dtype=ti.float64, shape=len(model.v))
v.from_numpy(model.v)
lMax = ti.field( dtype=ti.float64, shape=len(model.lMax))
lMax.from_numpy(model.lMax)
vel = ti.Vector.field(3, dtype=ti.float64, shape=len(model.v))
f = ti.Vector.field(3, dtype=ti.float64, shape=len(model.v))

@ti.kernel
def test():
    ti.loop_config(serialize=True)
    for iLoop in range(10000):
        for i in model.e:
            iv0 = model.e[i, 0]
            iv1 = model.e[i, 1]
            v[iv0] = v[iv1]
            
            f[i] = - x[i] * 0.5
            x[i] += f[i]

test()










