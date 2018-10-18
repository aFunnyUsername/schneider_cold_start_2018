from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import time
import datetime
from datetime import datetime, timedelta

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
	
full_meta = add_unique_building(full_meta)
full_future_meta = add_unique_building(full_future_meta)

full_meta.to_csv(data_path / 'meta_reorg_with_unique.csv', index=False)
full_future_meta.to_csv(data_path / 'future_meta_reorg_with_unique.csv', index=False)









