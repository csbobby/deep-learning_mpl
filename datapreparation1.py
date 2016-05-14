import numpy as np
import os

def feature_loader(datasets,features):
    """
    fea = np.loadtxt(datasets,delimiter= ' ')
    target = np.loadtxt(features,delimiter= '\t')
    num_samples = len(fea)
    shuffleidx = range(num_samples)                                             
    np.random.shuffle(shuffleidx) 
    sampleRatio = 0.9                                                           
    sample_boundary = int(num_samples*sampleRatio)                              
    train_fea = fea[shuffleidx[:sample_boundary]]                               
    train_target = target[shuffleidx[:sample_boundary]]                         
    test_fea = fea[shuffleidx[sample_boundary:]]                                
    test_target = target[shuffleidx[sample_boundary:]]
    datadir = os.path.split(datasets)[0]
    return datadir,train_fea,train_target,test_fea,test_target
    """
    f = open(datasets)
    fea = []
    i = 0
    for line in f:
        i+= 1
        if i <= 5000: 
            items = line.strip().split(' ')
            l = []
            for item in items:
                l.append(float(item))
            fea.append(l)
    f.close
    fea = np.array(fea)

    f1 = open(features)
    target = []
    j = 0
    for line in f1:
        j += 1
        if j <= 5000:
            items = line.strip().split('\t')
            l = []
            for item in items:
                l.append(int(item))
            target.append(l)
    f1.close
    target = np.array(target)
    target = (np.arange(19)==target[:]).astype(int)


    
    num_samples = len(fea)
    shuffleidx = range(num_samples)                                             
    np.random.shuffle(shuffleidx) 
    sampleRatio = 0.9                                                           
    sample_boundary = int(num_samples*sampleRatio)                              
    train_fea = fea[shuffleidx[:sample_boundary]]                               
    train_target = target[shuffleidx[:sample_boundary]]                         
    test_fea = fea[shuffleidx[sample_boundary:]]                                
    test_target = target[shuffleidx[sample_boundary:]]
    datadir = os.path.split(datasets)[0]
    return datadir,train_fea,train_target,test_fea,test_target

