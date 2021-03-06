#Now it is finally time to make predictions using our model!
from reorg_and_save import get_reorganized_dataframes, read_init_data
from pathlib import Path

import pandas as pd
import numpy as np

import time
import datetime
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from tensorflow.keras.models import Model, load_model
from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto
from numpy.random import seed
#from tensorflow import set_random_seed, GPUOptions, Session, ConfigProto
RANDOM_SEED = 2018
seed(RANDOM_SEED)
#set_random_seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = Session(config=ConfigProto(gpu_options=gpu_options))

data_path = Path('.\\data')
#load all files and models needed
model_name = '500-dense1-256-1539731294'
model = load_model(f'models\\{model_name}.sav')
#scaler = joblib.load('models\\scalers\\standard_scaler.sav')

consumption_train = read_init_data(data_path, 'consumption_train.csv')
cold_start_test = read_init_data(data_path, 'cold_start_test.csv')
submission_format = read_init_data(data_path, 'submission_format.csv')
meta_data = read_init_data(data_path, 'meta.csv', meta_flag=True)




prediction_indicator = pd.read_csv('data\\preds_indicator.csv')
mf_component = pd.read_csv('data\\recon_big_matrix.csv')
mf_component.columns = prediction_indicator.columns
seasonal_window = 24
meta_factors = 8

def generate_forecast(num_pred_days, meta, residuals, model, scaler):

	preds_scaled = np.zeros((num_pred_days, seasonal_window)) 
	pred_X = meta.astype(int)
	
	for i in range(num_pred_days):
		X = pred_X[i, :].reshape(1, meta_factors)
		yhat = model.predict(X)
		full_yhat = yhat + residuals[i]
		print(residuals[i])	
		preds_scaled[i] = full_yhat
		preds_scaled[i] = yhat
		
	hourly_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(num_pred_days, seasonal_window)
	return hourly_preds




my_submission = submission_format.copy()

pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
pred_window_to_num_pred_days = {'hourly': 1, 'daily': 7, 'weekly': 14}

model.reset_states()

#NOTE, we will want to have some indication of the mf model hyper parameters here
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
	ts_data, meta, scaler = get_reorganized_dataframes(cold_start)
	cold_X = meta.drop('series_id', axis=1).values
	cold_Y = ts_data.values
	cold_Y = cold_Y.T
	

	
	isOff_df = meta_data[meta_data['series_id'] == ser_id][[
		'monday_is_day_off', 
		'tuesday_is_day_off',
		'wednesday_is_day_off',
		'thursday_is_day_off',
		'friday_is_day_off',
		'saturday_is_day_off',
		'sunday_is_day_off']]


	#get meta data for prediction window
	pred_df.reset_index(inplace=True)
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
	future_meta = cold_X[0,:-1].reshape(1, -1)
	isOn = np.empty([num_pred_days, 1])
	for i, weekday in enumerate(weekdays):
		if isOff_df.iloc[0, weekday] == True:
			isOn[i] = int(0)
		else:
			isOn[i] = int(1)
	future_meta = np.repeat(future_meta, num_pred_days, axis=0)
	future_meta = np.concatenate((future_meta, isOn), axis=1)
	#print(future_meta)
	
	#here, get the residuals from the matrix
	pred_loc_list = prediction_indicator[str(ser_id)][prediction_indicator[str(ser_id)] == -100].index.tolist()
	resids = mf_component.loc[pred_loc_list, str(ser_id)].values
	resids_reorg = resids.reshape(int(len(resids) / seasonal_window), seasonal_window)
	
	preds = generate_forecast(num_pred_days, future_meta, resids_reorg, model, scaler)
	break
	#preds = generate_forecast(num_pred_days, future_meta, model, scaler)

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
#my_submission.to_csv(f'predictions\\{NAME}_avg_residuals_added.csv')
	








