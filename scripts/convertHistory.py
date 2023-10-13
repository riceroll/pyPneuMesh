import numpy as np
from pathlib import Path
import tqdm

# lobster
# folderDir = "/mnt/disks/pneumesh/pneumesh_self/pyPneuMesh/scripts/trainLobster/output/2023-02-02_16-23-13"

# elitePools = {1 : 'scripts/trainTable_2/output/2023-03-16_17-13-07/',
#               2: 'scripts/trainTable_2/output/2023-05-18_18-32-23/',
#               3: 'scripts/trainTable_2/output/2023-05-18_19-30-10/'
# }

# elitePools = {1 : 'scripts/trainTable_8/output/2023-03-16_17-16-25/',
#               2: 'scripts/trainTable_8/output/2023-04-26_05-52-20/',
#               3: 'scripts/trainTable_8/output/2023-04-26_05-53-40/'
# }

# elitePools = {1 : 'scripts/trainTable_16/output/2023-02-23_19-02-40/',
#               2: 'scripts/trainTable_16/output/2023-04-26_05-51-48/',
#               3: 'scripts/trainTable_16/output/2023-04-26_20-37-43/'
# }

# elitePools = {1 : 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
#               2: 'scripts/trainTable_32/output/2023-05-12_21-16-26/',
#               3: 'scripts/trainTable_32/output/2023-05-15_17-07-16/'
# }

# elitePools = {1 : 'scripts/trainTable_64/output/2023-03-22_18-01-24/',
#               2: 'scripts/trainTable_64/output/2023-05-10_17-48-32/',
#               3: 'scripts/trainTable_64/output/2023-05-11_17-31-50/'
# }


folders = {
    
    # 'table_2_0': 'scripts/trainTable_2/output/2023-03-16_17-13-07/',
    # 'table_2_1': 'scripts/trainTable_2/output/2023-05-18_18-32-23/',
    # 'table_2_2': 'scripts/trainTable_2/output/2023-05-18_19-30-10/',
    
    # 'table_8_0': 'scripts/trainTable_8/output/2023-03-16_17-16-25/',
    # 'table_8_1': 'scripts/trainTable_8/output/2023-04-26_05-52-20/',
    # 'table_8_2': 'scripts/trainTable_8/output/2023-04-26_05-53-40/',
    
    # 'table_16_0': 'scripts/trainTable_16/output/2023-02-23_19-02-40/',
    # 'table_16_1': 'scripts/trainTable_16/output/2023-04-26_05-51-48/',
    # 'table_16_2': 'scripts/trainTable_16/output/2023-04-26_20-37-43/',
    
    # 'table_32_0': 'scripts/trainTable_32/output/2023-04-06_04-05-10/',
    # 'table_32_1': 'scripts/trainTable_32/output/2023-05-12_21-16-26/',
    # 'table_32_2': 'scripts/trainTable_32/output/2023-05-15_17-07-16/',
    #
    # 'table_64_0': 'scripts/trainTable_64/output/2023-03-22_18-01-24/',
    # 'table_64_1': 'scripts/trainTable_64/output/2023-05-10_17-48-32/',
    # 'table_64_2': 'scripts/trainTable_64/output/2023-05-11_17-31-50/',
    
    # 'lobster': 'scripts/trainLobster/output/2023-02-01_21-14-46'
    # 'lobster': 'scripts/trainLobster/output/2023-02-02_16-23-13'
    
    "helmet": '/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainHelmet4_static/output/2023-09-04_16-42-01'
    
}

allScores = dict()

for name in folders:
    print(name)
    
    folderDir = folders[name]
    folderPath = Path(folderDir)
    
    for filePath in folderPath.iterdir():
        
        # get objective
        if filePath.suffix == '.txt':
            with open(str(filePath)) as iFile:
              content = iFile.read()
            
            # objectives
            with open(str(filePath)) as iFile:
                lines = iFile.readlines()
                line = lines[4]
                objectives = line.split('|')
                objs = []
                for objective in objectives:
                    subObjectives = objective.split()
                    objs.append(subObjectives)
    
    scores = dict()
    
    for filePath in tqdm.tqdm(folderPath.iterdir()):
        
        if filePath.suffix == '.npy' and filePath.name[0] == 'E':
            iEpoch = int(filePath.name.split('.')[0].split('_')[1])
            
            try:
                data = np.load(filePath, allow_pickle=True).all()
            
            except Exception as e:
                print(e)
                continue
            
            score = np.array([e['score'] for e in data['elitePoolMOODict']])
            
            if len(score) == 0:
                continue
            
            scores[iEpoch] = score
    
    print('len_scores: ', len(scores))
    
    # for iIter in scores:
    #     print(iIter, scores[iIter].shape)
    
    scores = np.array([scores[i] for i in list(scores.keys())])
    
    epochs = content.split('Training ElitePool')[1:-1]
    gens = [epoch.split('gen:')[-2].split('\n')[1:-1] for epoch in epochs]
    # gen = np.array([   [float(i) for i in gene.split()[1:-1]] for gene in gen])
    
    scores2 = dict()
    
    for i, epoch in enumerate(epochs):
      elitePoolString = epoch.split('GenePool:')[1].split('ElitePool')[1].split('\n')[0]
      if len(elitePoolString) < 30: # elitePool not updated
        continue
    
      means = np.array([float(subString.replace(' ', '').split('/')[0]) for subString in elitePoolString.split('        ')[1].split('    ')])
      maxs = np.array([float(subString.replace(' ', '').split('/')[1]) for subString in elitePoolString.split('        ')[1].split('    ')])
    
      gen = epoch.split('gen:')[-2].split('\n')[1:-1]
      gen = np.array([[float(i) for i in gene.split()[1:-1]] for gene in gen])
      scores2[i] = gen
    
      if(  not((means - gen.mean(0)).sum() < 0.001   )):
        print(epoch[:20])
        print(means, maxs)
        print(gen.mean(0), gen.max(0))
      print()
    
    # import re
    #
    # scores = dict()
    #
    # pattern = '\nElitePool:(.*?)\n'
    # eliteMatches = re.findall(pattern, content)
    # pattern = '\nGenePool:(.*?)\n'
    # geneMatches = re.findall(pattern, content)
    # geneMatches = []
    #
    # for matches in [geneMatches, eliteMatches]:
    #
    #   if len(scores) > 5:
    #     break
    #   for match in matches:
    #     iter = int(match[:12].replace(' ', ''))
    #     match = match[12:]
    #
    #     # print(iter)
    #     items = match.split('    ')
    #     if len(items) > 0 and items[0] != '':
    #
    #       scores[iter] = []
    #
    #       for iObject, item in enumerate(items):
    #         item = item.replace(' ', '')
    #         if len(item) == 0:
    #           continue
    #         mean = float(item.split('/')[0].replace(' ', ''))
    #         max = float(item.split('/')[1].replace(' ', ''))
    #         # print(mean, max)
    #
    #         # if iter == 78:
    #           # breakpoint()
    #
    #         scores[iter].append(mean)
    #         scores[iter].append(max)
    
    # scores = np.array([scores[i] for i in list(scores.keys())])
    
    # print(objs)
    # print(scores.shape)
    

    data = {
        'objectives': objs,
        'scores': scores2
    }

    outDir = '/Users/Roll/Desktop/MetaTruss-Submission/Plots/dataPlotting/'

    np.save(outDir + name + '.npy', data)
    print(outDir + name + '.npy')




