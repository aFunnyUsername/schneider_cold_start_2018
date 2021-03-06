#Script to reorganize the NhatxThat residual matrix into the TxN (840x1353) matrix for the MF component.

import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime, timedelta

resids_daily = pd.read_csv('data\\resids_from_nn.csv')
consumption_train = pd.read_csv('data\\consumption_train.csv',
		index_col=0, parse_dates=['timestamp'])
cold_start_test = pd.read_csv('data\\cold_start_test.csv',
		index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv('data\\submission_format.csv',
		index_col='pred_id', parse_dates=['timestamp'])

#2 things need to happen in this loop:
#1. get the .weekday() value for the first day of data.  This will be the starting point in the big matrix for 
#each series.
#2. find out if the series is a cold_start_test series and if so, get the prediction window and save those time
#steps in a mask relative to the big series.

my_submission = submission_format.copy()

pred_window_to_num_pred_days = {'hourly': 1, 'daily': 7, 'weekly': 14}

total_series = 1383
seasonal_window = 24
num_weeks = 5
num_hours = num_weeks * 7 * seasonal_window

zero_arr = np.zeros((num_hours, total_series))
zero_arr[:] = np.nan
pred_ind_arr = np.zeros((num_hours, total_series))
pred_ind_arr[:] = np.nan

pred_flag = False
prev_id = 0
col_count = 0
series_resids_list = []

resids_df = pd.DataFrame()

num_columns = len(resids_daily.columns)

def get_values(new_list, series):
	this_list = series.tolist()
	for entry in this_list:
		new_list.append(entry)
	return new_list

for i, column in enumerate(resids_daily.columns):
	ser_id = int(column[:6])
	print(ser_id)
	if i < (num_columns - 1):
		next_id = int(resids_daily.columns[i + 1][:6])
	else:
		next_id = None
	
	if ser_id == next_id:
		series_resids_list = get_values(series_resids_list, resids_daily[column])
		#print(series_resids_list)
		#print(len(series_resids_list))
	else:
		series_resids_list = get_values(series_resids_list, resids_daily[column])

		if resids_df.empty:	
			resids_df[ser_id] = series_resids_list

		else:

			if len(series_resids_list) != len(resids_df):
				pred_window = my_submission[my_submission['series_id'] == ser_id]['prediction_window'].unique()[0]
				pred_days = pred_window_to_num_pred_days[pred_window]
				len_cur_list = len(series_resids_list)
				for i in range(len_cur_list, len_cur_list + (pred_days * 24)):
					series_resids_list.append('pred')
				for i in range(len(series_resids_list), len(resids_df)):
					series_resids_list.append(np.nan)
				resids_df[ser_id] = series_resids_list

			else:
				resids_df[ser_id] = series_resids_list
		del series_resids_list[:]

preds_ind_df = pd.DataFrame(index=resids_df.index, columns=resids_df.columns)
preds_ind_df = resids_df[resids_df == 'pred']
preds_ind_df.replace('pred', -100, inplace=True)
#preds_ind_df.to_csv('data\\preds_indicator.csv', index=False)
resids_df.replace('pred', np.nan, inplace=True)
#resids_df.to_csv('data\\resids_by_series.csv', index=False)

pred_flag = False

for i, column in enumerate(resids_df.columns):
	series_id = int(column)
	print(series_id)	

	if series_id in consumption_train['series_id'].unique():
		df = consumption_train[consumption_train['series_id'] == series_id].reset_index()
	elif series_id in cold_start_test['series_id'].unique():
		df = cold_start_test[cold_start_test['series_id'] == series_id].reset_index()
		pred_flag = True
	
	first_day = df.loc[0, 'timestamp'].weekday()

	start = first_day * seasonal_window
	end = start + resids_df[column].shape[0]
	zero_arr[start:end, i] = resids_df[column].values
	pred_ind_arr[start:end, i] = preds_ind_df[column].values

		
big_df = pd.DataFrame(zero_arr, columns=resids_df.columns)	
reorg_preds_ind_df = pd.DataFrame(pred_ind_arr, columns=preds_ind_df.columns)
big_df.to_csv('data\\big_df.csv', index=False, header=False)
reorg_preds_ind_df.to_csv('data\\preds_indicator.csv', index=False)
#np.savetxt('data\\big_arr.csv', zero_arr, delimiter=',')
#np.savetxt('data\\big_pred_ind_arr.csv', pred_ind_arr, delimiter=',')









