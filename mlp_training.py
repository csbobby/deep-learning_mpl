from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras import callbacks as cb
#from keras.utils.visualize_util import plot
#from IPython.display import SVG
#from keras.utils.visualize_util import to_graph

import pandas as pd
import numpy as np
from keras.optimizers import SGD
def mlp_training(X_train,y_train,input_dim,features,datadir,history_path):
	#settings
	param={}
	param['nb_epoch']= 2000
	param['batch_size']=16
	param['validation_split']=0.1
	print 'traning',param

	# Here's a Deep Dumb MLP (DDMLP)
	#create a model
	model = Sequential()
	model.add(Dense(output_dim=1024, input_dim=input_dim, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(19, init='uniform'))
	model.add(Activation('softmax'))

	
	# we'll use MSE (mean squared error) for the loss, and RMSprop as the optimizer
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)


    #SVG(to_graph(model).create(prog='dot',format='svg'))
    #from keras.utils.visualize_util import plot
    #plot(model,'model.png')
	
	#from keras.utils.dot_utils import Grapher
	#grapher = Graph()
	#grapher.fit(model, 'model.png')


	#if the evalidation error decreased after one epoch, save the model 
	checkpointer = cb.ModelCheckpoint(filepath=datadir+features+".hdf5", verbose=1, save_best_only=True)
	# the callback function for logging loss 
	class LossHistory(cb.Callback):  
		def on_train_begin(self, logs={}):  
			self.losses = []
		def on_batch_end(self, batch, logs={}):  
			self.losses.append(logs.get('loss'))  
	# define a callback object
	history = LossHistory()  
	
	
	print("start train process...")
	hist = model.fit(X_train, y_train,nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], \
	    validation_split=param.get('validation_split'), show_accuracy=True, verbose=0 \
        , callbacks=[checkpointer,history]
        )
	
	loss = hist.history.get('loss')
	val_loss = hist.history.get('val_loss')
	
		
	return model,loss,val_loss

