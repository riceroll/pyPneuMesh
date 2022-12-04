from utils.model import Model
import time
import sys

model = Model('./data/config.json')
model.load('./data/table2.json')
model.updateL()
model.lMax = model.l


Model.h = 0.0005
# t0 = time.time()
for i in range(int(sys.argv[1])):
    model.step()

# print(model.v)

print(model.l)

# t1 = time.time()
# print(t1 - t0)



