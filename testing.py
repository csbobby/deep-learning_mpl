from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
from scipy import stats


def testing(model,X_test,y_test,resultsfname,evaluationfname):
	print("Evaluate the results...")

	preds = model.predict_classes(X_test, verbose=0)
	def write_preds(preds, fname):
		pd.DataFrame({"SampleID": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
	write_preds(preds, resultsfname)

        y_list = []
        preds_list = []
        print preds
        for i in range(len(preds)):
            y_list.append(y_test[i][0])
            preds_list.append(preds[i])
	spearmanr_corr = stats.pearsonr(y_list, preds_list)
	print "Pearson Correlation",spearmanr_corr
	
	objective_score = model.evaluate(X_test, y_test, batch_size=32)
	print "objective_score",objective_score

	return spearmanr_corr,objective_score

