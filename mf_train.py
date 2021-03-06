#here, we train the matrix factorization model, in addition to storing the imputed data matrix

import h2o
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator as GLRM
import pandas as pd
import numpy as np

h2o.init(max_mem_size=4)
h2o.remove_all()

SEED=2018

matrix = h2o.import_file(path='data\\big_df.csv')
#print(matrix.columns)
model = GLRM(k=1, 
		loss='Absolute', 
		regularization_x='L2', 
		regularization_y='L2',
		init='svd')
		#ignore_const_cols=False)
#, transform='STANDARDIZE')
#max_iterations=5)#, transform='STANDARDIZE')
print('model built!')
model.train(training_frame=matrix)
print('model trained!')	
model.show()

model.summary()

preds = model.predict(matrix)

preds = h2o.as_list(preds, use_pandas=True)

preds.to_csv('data\\recon_big_matrix.csv', index=False)











