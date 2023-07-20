import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import tqdm

sns.set_theme(style="white")

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['font.size'] = 7


folder = '/Users/Roll/Desktop/dataPlotting/'

colors = [
    '#F21365',
    '#66C8D2',
    '#F1B807',
    '#F5AB55',
    '#F27A5E',
    '#63A5BF'
]

table_8_0 = 'table_64_0_2.npy'
table_8_1 = 'table_64_1_2.npy'
table_8_2 = 'table_64_2_2.npy'

def plotCurve(x, y, variances=None, title='hehe', x_axis='iteration', y_axis='performance', x_interval=25, n_yticks=10, color='b', label='fitness'):
    
    plt.plot(x, y, label=label, color=color, linewidth=2.0)  # Add a label
    
    # Add the shaded variance
    if variances is not None:
        upper = y + variances
        lower = y - variances
        plt.fill_between(x, lower, upper, color=color, alpha=.1)
    
    plt.tick_params(left="on", bottom="on", length=4, width=1.0)
    
    plt.xticks(np.arange(min(x), max(x), x_interval))
    y_locs = np.arange(min(y), max(y) + (max(y) - min(y)) * 0.2, ( max(y) - min(y)) / n_yticks)
    plt.yticks(y_locs, [f'{loc:.3f}' for loc in y_locs])
    
    # Change the frame width
    ax = plt.gca()  # Get the current axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.0)  # Set the line width to 2.0
    
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    
    plt.legend(frameon=False)
    
def getRCD(ratings: np.ndarray) -> (np.ndarray, np.ndarray):
    if len(ratings) == 0:
        return np.array([]), np.array([])
    # get R
    ratings = ratings.reshape(len(ratings), -1)
    ratingsCol = ratings.reshape([ratings.shape[0], 1, ratings.shape[1]])
    ratingsRow = ratings.reshape([1, ratings.shape[0], ratings.shape[1]])
    dominatedMatrix = (ratingsCol <= ratingsRow).all(2) * (ratingsCol < ratingsRow).any(2)
    
    Rs = np.ones(len(ratings), dtype=int) * -1
    R = 0
    while -1 in Rs:
        nonDominated = (~dominatedMatrix[:, np.arange(len(Rs))[Rs == -1]]).all(1) * (Rs == -1)
        Rs[nonDominated] = R
        R += 1
    
    # get CD
    CDs = np.zeros(len(ratings))
    R = 0
    while R in Rs:
        ratingsSurface = ratings[Rs == R]
        CDMatrix = np.zeros_like(ratingsSurface, dtype=np.float)
        sortedIds = np.argsort(ratingsSurface, axis=0)
        CDMatrix[sortedIds[0], np.arange(len(ratingsSurface[0]))] = np.inf
        CDMatrix[sortedIds[-1], np.arange(len(ratingsSurface[0]))] = np.inf
        ids0 = sortedIds[:-1, :]
        ids1 = sortedIds[1:, :]
        distances = ratingsSurface[ids1, np.arange(len(ratingsSurface[0]))] - ratingsSurface[
            ids0, np.arange(len(ratingsSurface[0]))]
        
        if ((ratingsSurface.max(0) - ratingsSurface.min(0)) > 0).all():
            CDMatrix[sortedIds[1:-1, :], np.arange(len(ratingsSurface[0]))] = \
                (distances[1:] + distances[:-1]) / (ratingsSurface.max(0) - ratingsSurface.min(0))
        else:
            CDMatrix[sortedIds[1:-1, :], np.arange(len(ratingsSurface[0]))] = np.inf
        CDsSurface = CDMatrix.mean(1)
        CDs[Rs == R] = CDsSurface
        R += 1
    
    return Rs, CDs

def toNumpyFilled(dic):
    arr = np.empty(max(dic.keys()) + 1)
    arr[:] = np.nan
    
    for k, v in dic.items():
        arr[k] = v
    
    # Forward fill function to replace nan
    def numpy_ffill(arr):
        mask = np.isnan(arr)
        idxs = np.where(~mask,np.arange(mask.shape[0]),0)
        np.maximum.accumulate(idxs, out=idxs)
        out = arr[idxs]
        
        out[out == np.nan] = dic[min(dic.keys())]
        
        return out
    
    return numpy_ffill(arr)


def plot(name0, index, title, label, x_axis, y_axis, color, name1=None, name2=None):
    from pymoo.indicators.hv import HV
    
    def getVolumes(x, y, data):
        for iScore in tqdm.tqdm(x):
            score = data['scores'][iScore]
            
            hv = HV(ref_point=(0, 0, 10, 0))
            volume = hv(-score)
            
            y[iScore] = volume
        return y
    
    if index == -1:
        data = np.load(folder + name0, allow_pickle=True).all()
        x0 = sorted(list(data['scores'].keys()))
        y0 = dict()
        y0 = getVolumes(x0, y0, data)
        
        data = np.load(folder + name1, allow_pickle=True).all()
        x1 = sorted(list(data['scores'].keys()))
        y1 = dict()
        y1 = getVolumes(x1, y1, data)
        
        data = np.load(folder + name2, allow_pickle=True).all()
        x2 = sorted(list(data['scores'].keys()))
        y2 = dict()
        y2 = getVolumes(x2, y2, data)
        
    else:
        data = np.load(folder + name0, allow_pickle=True).all()
        x0 = data['scores'].keys()
        y0 = [data['scores'][xx][index] for xx in x0]
        y0 = {xx: data['scores'][xx][index] for xx in x0}
        
        data = np.load(folder + name1, allow_pickle=True).all()
        x1 = data['scores'].keys()
        y1 = [data['scores'][xx][index] for xx in x1]
        y1 = {xx: data['scores'][xx][index] for xx in x1}
        
        data = np.load(folder + name2, allow_pickle=True).all()
        x2 = data['scores'].keys()
        y2 = [data['scores'][xx][index] for xx in x2]
        y2 = {xx: data['scores'][xx][index] for xx in x2}
    
    
    y0 = toNumpyFilled(y0)[:1000]
    y1 = toNumpyFilled(y1)[:1000]
    y2 = toNumpyFilled(y2)[:1000]
    
    x = np.arange(1000)
    y = np.array([y0, y1, y2]).mean(0)
    variance = np.array([y0, y1, y2]).std(0) ** 2
    
    plotCurve(x, y0, None, title=title,
              x_axis=x_axis, y_axis=y_axis, x_interval=100, n_yticks=10, color=color, label=label)

    plotCurve(x, y1, None, title=title,
              x_axis=x_axis, y_axis=y_axis, x_interval=100, n_yticks=10, color=color, label=label)

    plotCurve(x, y2, None, title=title,
              x_axis=x_axis, y_axis=y_axis, x_interval=100, n_yticks=10, color=color, label=label)

def plot2(name0, index, title, label, x_axis, y_axis, color, name1=None, name2=None):
    data = np.load(folder + name0, allow_pickle=True).all()
    
    iLast = sorted(list(data['scores'].keys()))[-1]
    
    score = data['scores'][iLast]
    
    from pymoo.indicators.hv import HV
    
    hv = HV(ref_point=(0, 0, 10, 0))
    volume = hv(-score)
    print(volume)
    
    #
    for key in sorted(list(data['scores'].keys())):
        score = data['scores'][key]
        
        from pymoo.indicators.hv import HV
        
        hv = HV(ref_point=(0, 0, 10, 0))
        volume = hv(-score)
        print(key, volume)
    
    
    
    # return R, CD
    

plot(table_8_0, -1, title='4-legged robot locomotion', x_axis='iterations', y_axis='displacement(m)', color=colors[0], label='hehe', name1=table_8_1, name2=table_8_2)

plt.show()

