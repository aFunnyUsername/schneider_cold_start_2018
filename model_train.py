#script to train both the regression neural network and the matrix factorization model
import numpy as np
from numpy.random import seed
import pandas as pd

import time

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto

RANDOM_SEED = 2018
seed(RANDOM_SEED)
#set_random_seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = Session(config=ConfigProto(gpu_options=gpu_options))
#===============================================================================================================
ts_data = pd.read_csv('data\\tot_ts_reorg.csv')
meta = pd.read_csv('data\\meta_reorg.csv')

meta_matrix = meta.iloc[:, 1:].values
unique_buildings = np.unique(meta_matrix, axis=0)
unique_building_dict = {}
for i in range(unique_buildings.shape[0]):
	unique_building_dict[i] = unique_buildings[i, :]




"""
vector_list = []
for i, row in enumerate(meta['series_id']):
	full_vector = meta.iloc[i, 1:].values
	vector_list.append(full_vector)	
meta['full_vector'] = vector_list	

unique_ints = []
for vector in meta['full_vector']:
	for key, value in unique_building_dict.items():
		if np.array_equal(vector, value):
			unique_ints.append(key)
meta['unique_building'] = unique_ints
meta = meta.drop('full_vector', axis=1)
meta.to_csv('data\\meta_reorg_with_unique.csv', index=False)
exit()
"""
#---------------------------------------------------------------------------------------------------------------

seasonal_window = 24
#this is the number of columns of the meta data matrix.  Note, this is AFTER one hot encoding, and grabbing the
#daily isOff info
meta_factors = 8
dense_1_nodes = 256#[seasonal_window, 32, 64, 128, 256, 512]
dense_2_nodes = 0#[0, seasonal_window, 32, 64, 128, 256, 512]
#epochs = 500
epochs = 500
#epochs = 3000
val_split = 0.2
#learning_rates = [0.001, 0.01, 0.1, 1, 10]
#momentums = [0.5, 0.6, 0.7, 0.8, 0.9]
learning_rate = 0.01
momentum = 0.8
decay = learning_rate / epochs

NAME = f"{epochs}-dense1-{dense_1_nodes}-{int(time.time())}"
print(NAME)
tensorboard = TensorBoard(log_dir=f'dense_logs/{NAME}')
#meta Regression model
meta_input = Input(shape=(meta_factors,), name='meta_input')
#NOTE, look into the use_bias parameter here as well
dense_1 = Dense(dense_1_nodes, activation='relu')(meta_input)
if dense_2_nodes == 0:
	output = Dense(seasonal_window, name='output_always_24')(dense_1)
else:
	dense_2 = Dense(dense_2_nodes, activation='relu')(dense_1)
	output = Dense(seasonal_window, name='output_always_24')(dense_2)

#compile model
model = Model(inputs=meta_input, outputs=output)
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum)
model.compile(optimizer=sgd, loss='mae')

print(model.summary())

X = meta.drop('series_id', axis=1).values
Y = ts_data.values
Y = Y.T

#FIT THE MODEL HERE

history = model.fit(X, Y, 
		batch_size=32, epochs=epochs, 
		verbose=1, validation_split=val_split,
		callbacks=[tensorboard], shuffle=True,
		steps_per_epoch=None)

model_fp = f'models\\{NAME}.sav'
model.save(model_fp)

#get residuals from training on just metadata
#these will be used to impute the full residual matrix
preds = model.predict(X)
resids = Y - preds

resids_df = pd.DataFrame(resids.T, columns=ts_data.columns)

resids_df.to_csv('data\\resids_from_nn.csv', index=False)

#===============================================================================================================
#Here, we will train the matrix factorization component
#We will be imputing the missing values in the residual dataframe from the neural network

resids_df = pd.read_csv('data\\resids_big_matrix.csv')





