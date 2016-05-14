import pandas as pd
import numpy as np

from mlp_training import mlp_training
from testing import testing
from predicting import predicting
from IterationGraph import *
import sys
sys.path.insert(0, '../')
import datapreparation1 as dp

if __name__ == "__main__":
    print 'start'
    did = 0
    fid = 2
    datasets=['3W','40W','68W','100W'];    datasets=datasets[did]# MSRA:3W 68W VSO_CC:40W 100W
    features=['ALL','USER','DL','VISUAL'];    features=features[fid]
    resulttitle='features'
    sname=features
    model_save_filename = None
    model_save_collection = {}

    
    ## step 1: load data
    print "step 1: load data..."
    data_path = '/root/tangmeng/pcaData4Caffe_30000-500/list_train_input_ALLIN_3W_USER.txt'
    target_path = '/root/tangmeng/pcaData4Caffe_30000-500/list_train_label_ALLIN_3W.txt'
    datadir,X_train,y_train,X_test,y_test = dp.feature_loader(data_path,target_path)
    #results and evaluation fnames
    resultsfname = datadir+'_'+'results_ALLIN_'+datasets+'_'+features+'.txt'
    evaluationfname = datadir+'_'+'evaluation_ALLIN_'+datasets+'_'+features+'.txt'
    
    print 'training...'
    input_dim = len(X_train[0])
	model,loss,val_loss = mlp_training(X_train,y_train,input_dim)
    history_path = '/root/tangmeng/%s_history.txt'%features 
    graph_out_path = '/root/tangmeng/Graph/%s_loss.png'%features
	IterationGraph(loss,val_loss,history_path,graph_out_path,features)

        
    print 'testing...'
    spearmanr_corr,objective_score = testing(model,X_test,y_test,resultsfname,evaluationfname)
        
    preds = predicting(model,X_test,resultsfname)


