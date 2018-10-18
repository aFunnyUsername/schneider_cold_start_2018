from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import time
import datetime
from datetime import datetime, timedelta

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto

data_path = Path('.\\data')

def read_init_data(path, csv, meta_flag=False):
	#if it's one of the non meta files we want to parse timestamps, otherwise not	
	if meta_flag == False:	
		data = pd.read_csv(data_path / csv,
				index_col=0, parse_dates=['timestamp'])
	else:
		data = pd.read_csv(data_path / csv)
	return data 

consumption_train = read_init_data(data_path, 'consumption_train.csv')
cold_start_test = read_init_data(data_path, 'cold_start_test.csv')
submission_format = read_init_data(data_path, 'submission_format.csv')
meta_data = read_init_data(data_path, 'meta.csv', meta_flag=True)

my_submission = submission_format.copy()
pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
pred_window_to_num_pred_days = {'hourly': 1, 'daily': 7, 'weekly': 14}

#===============================================================================================================
#Here, we will process the meta data 
#We will onehot encode the surface area column and the base temperature column

def encode_physical_meta(df, attribute):
	onehot_df = pd.get_dummies(df[attribute], prefix=attribute)
	return onehot_df

surface_onehot = encode_physical_meta(meta_data, 'surface')
#here, we drop the "base_temperature_low" column since it is a binary feature, so we only need the 1 or 0 
#for whether or not it is high
base_temp_onehot = encode_physical_meta(meta_data, 'base_temperature').drop('base_temperature_low', axis=1)
#drop the original columns
meta_data.drop(['surface', 'base_temperature'], axis=1, inplace=True)

meta_data = pd.concat([meta_data, surface_onehot, base_temp_onehot], axis=1)
#===============================================================================================================
#Data Reorganization for Cold Start Only
#NOTE, we'll also get the prediction window meta vectors prepared here
#Consider both the training and testing data as a multivariate time series, with 1358 series, each with (at most)
#672 time points.

#We are given daily information in the metadata for each building: the is_day_off column.  This means that we 
#really have a meta data vector for each DAY rather than for each building.  

#The longest span of time for any one building is 4 weeks and in order to fit these 4 weeks into the matrix 
#without wrapping values is to have five weeks worth of "weekdays".

#Consider each weekday as an integer from 0-6 with 0 representing Monday, and 6 representing Sunday. 
#This sequence would repeat 5 times for 5 weeks worth of data. 
#In populating this data matrix, we will check the .weekday() return for the first timestamp of the series, and
#start that building's series on the first column with that number representation.  The rest of the points will
#fill out from there.

#NOTE, we will use all of the training data available - consumption_train and cold_start_test

tot = consumption_train.append(cold_start_test)

def reorganize_consumption(df, meta_df, seasonal_window, 
		column_name, ts_col_name, ser_id, period_split):
	#for the chained assignment warning that we dgaf about
	pd.options.mode.chained_assignment = None
	data = df[column_name].values

	#scale data
	scaler = StandardScaler()
	data = scaler.fit_transform(data.reshape(-1, 1)).ravel()
	timestamps = df[ts_col_name].reset_index().drop('index', axis=1)
	consum_reorg = pd.DataFrame()

	day_count = 1
	hour_count = 0

	isOff_df = meta_df[[
		'monday_is_day_off', 
		'tuesday_is_day_off',
		'wednesday_is_day_off',
		'thursday_is_day_off',
		'friday_is_day_off',
		'saturday_is_day_off',
		'sunday_is_day_off']]
	#get non isOff data columns
	#print(meta_df)
	new_meta = meta_df.loc[:, ['series_id', 'base_temperature_high',
		'surface_xx-small', 'surface_x-small', 'surface_small',
		'surface_medium',
		'surface_large', 'surface_x-large', 'surface_xx-large']]
	#print(new_meta)	
	combined_meta = pd.DataFrame()
	for i in range(0, len(data)):
		if i % seasonal_window == 0:
			#add new day column to dataframe
			next_time = day_count * seasonal_window	
			consum_reorg[f'{ser_id}_{period_split}{day_count}'] = data[hour_count:next_time]

			#get appropriate meta_data
			weekday = timestamps.loc[i, 'timestamp'].weekday()
			#check if this weekday is on or off.
			#NOTE true means the building is off (isOff = True) so we will add a 0 to this column in that case and 
			#a 1 in the other
			if isOff_df.iloc[0, weekday] == True:
				new_meta['building_isOn'] = 0
			else:
				new_meta['building_isOn'] = 1
			if combined_meta.empty:
				combined_meta = new_meta
			else:
				combined_meta = combined_meta.append(new_meta)
			combined_meta = combined_meta.reset_index().drop('index', axis=1)
			day_count += 1
			hour_count += seasonal_window
	#print(consum_reorg)
	#print(combined_meta)
	#combined_meta.to_csv('test_csv.csv')
	return consum_reorg, combined_meta, scaler, isOff_df


seasonal_window = 24


#create the reorganized all train dataset

def check_is_empty(df, new_df, meta_or_ts):
	if df.empty:
		return new_df
	else:
		if meta_or_ts == 'ts':
			return pd.concat([df, new_df], axis=1)
		elif meta_or_ts == 'meta':
			return df.append(new_df)


def get_reorganized_dataframes(df):
	full_ts = pd.DataFrame()
	full_meta = pd.DataFrame()
	full_future_meta = pd.DataFrame()

	for series_id, group_df in df.groupby('series_id'):
		print(series_id)
		meta_df = meta_data[meta_data['series_id'] == series_id]
		#print(meta_df)
		reorg_ts, meta, scaler, isOff_df = reorganize_consumption(group_df, 
				meta_df, seasonal_window, 
				'consumption', 'timestamp', series_id, 'day')
		
		#Get future meta data if this is a cold start series
		if series_id in cold_start_test['series_id'].unique():

			pred_df = my_submission[my_submission['series_id'] == series_id].reset_index()

			pred_window = pred_df.prediction_window.unique()[0]
			num_preds = pred_window_to_num_preds[pred_window]
			num_pred_days = pred_window_to_num_pred_days[pred_window] 

			timestamp = datetime.strptime(str(pred_df.loc[0, 'timestamp']), '%Y-%m-%d %H:%M:%S')
			day = timestamp.weekday()
			future_timestamps = list()
			weekdays = list()
			future_timestamps.append(timestamp)
			weekdays.append(day)	
			#if num_pred_days is 1, we won't go through this loop
			#if 7, we will add the next 6 days
			#if 14, we will get the next 13 days
			#these daily weekday() integers are used for the isOff information
			for day in range(1, num_pred_days):
				timestamp = timestamp + timedelta(days=1)
				day = timestamp.weekday()	
				future_timestamps.append(timestamp)
				weekdays.append(day)
			#starting meta frame
			future_meta = meta.loc[0:0, ['series_id', 
				'base_temperature_high', 'surface_xx-small', 
				'surface_x-small', 'surface_small', 
				'surface_medium', 'surface_large', 
				'surface_x-large','surface_xx-large']]
			isOn = np.empty([num_pred_days, 1])
			for i, weekday in enumerate(weekdays):
				if isOff_df.iloc[0, weekday] == True:
					isOn[i] = int(0)
				else:
					isOn[i] = int(1)
			future_meta = pd.concat([future_meta] * num_pred_days, ignore_index=True)
			future_meta['building_isOn'] = isOn.astype(int)
			#print(future_meta)
			full_future_meta = check_is_empty(full_future_meta, future_meta, 'meta')

		full_ts = check_is_empty(full_ts, reorg_ts, 'ts')
		full_meta = check_is_empty(full_meta, meta, 'meta')
	return full_ts, full_meta, full_future_meta, scaler


#full_ts, full_meta, full_future_meta, scaler = get_reorganized_dataframes(tot)
#full_ts.to_csv(data_path / 'tot_ts_reorg.csv', index=False)
#full_meta.to_csv(data_path / 'meta_reorg.csv', index=False)
#full_future_meta.to_csv(data_path / 'future_meta_reorg.csv', index=False)

full_ts = pd.read_csv(data_path / 'tot_ts_reorg.csv')
full_meta = pd.read_csv(data_path / 'meta_reorg.csv')
full_future_meta = pd.read_csv(data_path / 'future_meta_reorg.csv')

all_meta = full_meta.append(full_future_meta)
meta_matrix = all_meta.iloc[:, 1:].values
unique_buildings = np.unique(meta_matrix, axis=0)
unique_building_dict = {}
for i in range(unique_buildings.shape[0]):
	unique_building_dict[i] = unique_buildings[i, :]


def add_unique_building(df):
	vector_list = []
	for i, row in enumerate(df['series_id']):
		full_vector = df.iloc[i, 1:].values
		vector_list.append(full_vector)	
	df['full_vector'] = vector_list	

	unique_ints = []	
	for vector in df['full_vector']:
		for key, value in unique_building_dict.items():
			if np.array_equal(vector, value):
				unique_ints.append(key)
	df['unique_building'] = unique_ints
	df = df.drop('full_vector', axis=1)
	return df
	
#full_meta = add_unique_building(full_meta)
#full_future_meta = add_unique_building(full_future_meta)

#full_meta.to_csv(data_path / 'meta_reorg_with_unique.csv', index=False)
#full_future_meta.to_csv(data_path / 'future_meta_reorg_with_unique.csv', index=False)

full_meta = pd.read_csv(data_path / 'meta_reorg_with_unique.csv')
full_future_meta = pd.read_csv(data_path / 'future_meta_reorg_with_unique.csv')
#===============================================================================================================
#Model Training / Residual Grabbing

seasonal_window = 24
#this is the number of columns of the meta data matrix.  Note, this is AFTER one hot encoding, and grabbing the
#daily isOff info
meta_factors = 9
dense_1_nodes = 256#[seasonal_window, 32, 64, 128, 256, 512]
dense_2_nodes = 0#[0, seasonal_window, 32, 64, 128, 256, 512]
#epochs = 500
epochs = 250
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

X = full_meta.drop(['series_id', 'unique_building'], axis=1).values
Y = full_ts.values
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
preds_df = pd.DataFrame(preds.T, columns=full_ts.columns)
resids_df = pd.DataFrame(resids.T, columns=full_ts.columns)

resids_by_unique = pd.DataFrame(resids)
resids_by_unique = pd.concat([resids_by_unique, full_meta['unique_building']], axis=1)
resids_by_unique.to_csv(data_path / 'resids_by_unique.csv')
#resids_by_unique = pd.read_csv(data_path / 'resids_by_unique.csv')
print(resids_by_unique)
resids_by_unique_grouped = resids_by_unique.groupby('unique_building')
mean_resids_by_unique = resids_by_unique_grouped.mean()
hour_list = [number for number in range(0, 24)]
means = mean_resids_by_unique[hour_list].values

unique_building_resids = {}
for i, day in enumerate(means):
	unique_building_resids[i] = day
print(unique_building_resids)
preds_df.to_csv('preds_df_TESTING.csv')
resids_df.to_csv('resids_df_TESTING.csv')

#===============================================================================================================
#Model Predictions - NOTE still need to put MF in

prediction_indicator = pd.read_csv('data\\preds_indicator.csv')
mf_component = pd.read_csv('data\\recon_big_matrix.csv')
mf_component.columns = prediction_indicator.columns

def generate_forecast(num_pred_days, meta, bias, model, scaler):

	preds_scaled = np.zeros((num_pred_days, seasonal_window)) 
	
	for i in range(num_pred_days):
		yhat = model.predict(X[i, :].reshape(1, meta_factors))
		#print(yhat)
		#full_yhat = yhat + bias[i]
		#print(bias[i])
		#print(full_yhat)	
		#preds_scaled[i] = full_yhat
		preds_scaled[i] = yhat	
	hourly_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(num_pred_days, seasonal_window)
	return hourly_preds

model_name = NAME
model = load_model(f'models\\{model_name}.sav')
NAME = f'{model_name}+mf_TEST'

for ser_id, pred_df in my_submission.groupby('series_id'):
	print(f'series_id: {ser_id}')

	pred_window = pred_df.prediction_window.unique()[0]
	num_preds = pred_window_to_num_preds[pred_window]
	num_pred_days = pred_window_to_num_pred_days[pred_window] 

	#get the meta data for the prediction windows:
	
	#NOTE, this is not all necessary, since we already trained on the cold_x and y data. 
	#I just want the meta data in a preprocessed format, but the rest of this code should get updated at
	#some point...
	cold_start = cold_start_test[cold_start_test['series_id'] == ser_id]	
	garb1, garb2, garb3, scaler = get_reorganized_dataframes(cold_start)

	meta = full_future_meta[full_future_meta['series_id'] == ser_id]
	
	X = meta.drop(['series_id', 'unique_building'], axis=1).values
	
	bias = []
	for value in meta['unique_building']:
		bias.append(unique_building_resids[value])

	preds = generate_forecast(num_pred_days, meta, bias, model, scaler)

	reduced_preds = []
	if pred_window == 'hourly':
		reduced_preds = preds.T
	else:
		for i in range(preds.shape[0]):	
			day_sum = np.sum(preds[i, :])
			#daily sums in a list
			reduced_preds.append(day_sum)
		if pred_window == 'weekly':
			week1_sum = np.sum(reduced_preds[:int(num_pred_days / 2)])
			week2_sum = np.sum(reduced_preds[int(num_pred_days / 2):])
			reduced_preds = [week1_sum, week2_sum]
			
	#store result in submission DataFrame
	ser_id_mask = my_submission.series_id == ser_id
	my_submission.loc[ser_id_mask, 'consumption'] = reduced_preds
	#print(my_submission[my_submission['series_id'] == ser_id])
my_submission.to_csv(f'predictions\\{NAME}_bias_added.csv')






