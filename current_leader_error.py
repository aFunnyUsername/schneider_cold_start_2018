import numpy as np
import pandas as pd
from pathlib import Path

import time
import datetime
from datetime import datetime, timedelta

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from numpy.random import seed
#from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto
RANDOM_SEED = 2018
seed(RANDOM_SEED)
#set_random_seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = Session(config=ConfigProto(gpu_options=gpu_options))
#---------------------------------------------------------------
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
#Split into a train set and a test set, the latter of which will be a simulated prediction
train_split = len(consumption_train) * 0.5

all_train = consumption_train.ix[:train_split - 1,:]
train_test = consumption_train.ix[train_split:,:].reset_index().drop('index', axis=1)

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
	#full_future_meta = pd.DataFrame()

	for series_id, group_df in df.groupby('series_id'):
		print(series_id)
		meta_df = meta_data[meta_data['series_id'] == series_id]
		#print(meta_df)
		reorg_ts, meta, scaler, isOff_df = reorganize_consumption(group_df, 
				meta_df, seasonal_window, 
				'consumption', 'timestamp', series_id, 'day')
		
		#Get future meta data if this is a cold start series
		"""if series_id in cold_start_test['series_id'].unique():

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
			full_future_meta = check_is_empty(full_future_meta, future_meta, 'meta')"""

		full_ts = check_is_empty(full_ts, reorg_ts, 'ts')
		full_meta = check_is_empty(full_meta, meta, 'meta')
	return full_ts, full_meta, scaler


"""full_all_train_ts, full_all_train_meta, full_train_scaler = get_reorganized_dataframes(all_train)
full_all_train_ts.to_csv('data\\train_test\\full_all_train_ts.csv', index=False)
full_all_train_meta.to_csv('data\\train_test\\full_all_train_meta.csv', index=False)
pre_full_train_test_ts, pre_full_train_test_meta, train_test_scaler = get_reorganized_dataframes(train_test)
pre_full_train_test_ts.to_csv('data\\train_test\\pre_full_train_test_ts.csv', index=False)
pre_full_train_test_meta.to_csv('data\\train_test\\pre_full_train_test_meta.csv', index=False)"""
#NOTE, the full_all_train is ready for training, but the pre_full_train_test needs to be split into cold_start
#and prediction windows
#===============================================================================================================
full_all_train_ts = pd.read_csv('data\\train_test\\full_all_train_ts.csv')
full_all_train_meta = pd.read_csv('data\\train_test\\full_all_train_meta.csv')


#differnt cold start and prediction windows to simulate per series
cold_starts = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336] 
prediction_windows = [336, 168, 24]

cold_start_count = 0
pred_window_count = 0

#build the simulated submission
for series_id, series_data in train_test.groupby('series_id'):
	print(series_id)
	series_data = series_data.reset_index()
	cold = cold_starts[cold_start_count]
	pred = prediction_windows[pred_window_count]
	cold_start_window = series_data.loc[:cold-1, 'consumption']
	pred_window = series_data.loc[cold:(cold+pred)-1, 'consumption']
	break









