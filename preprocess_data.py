from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd 
import numpy as np
import pickle
import os

def string_to_datetime(string):
	y,m,d,h = [int(item) for item in string.split("-")]
	now = datetime(y,m,d,h)
	return now

def load_hourly_data():
	# derive train/test data from this source
	PATHS = [os.path.join(os.getcwd(), 'dataset','hourly_2017_{}.csv'.format(i)) for i in range(1,7)]
	dfs = [pd.read_csv(path, index_col=None, encoding='cp949') for path in PATHS]
	for df in dfs: clean_colname(df)
	resulting_df = pd.concat(dfs, axis=0)
	return resulting_df


def load_monthly_data():
	# used only for identifying valid stations : more than 1 transactions per hour
	path = os.path.join(os.getcwd(), 'dataset','monthly_2017.csv')
	df = pd.read_csv(path, index_col=None, encoding = 'cp949')
	return df

def calc_usage(monthly_df):
	# used only for identifying valid stations : more than 1 transactions per hour
	per_months = [monthly_df[monthly_df["'대여일자'"]=="'20170{}'".format(i)] for i in range(1,10)]
	per_stations = {}
	for month_df in per_months:
		_per_stations = {i:j for i,j in zip(month_df["'대여소번호'"].values, (month_df["'대여건수'"]).values)}
		# +month_df["'반납건수'"] 
		per_stations = add_dict(per_stations, _per_stations)

	# hours = datetime(2017,10,1,00)-datetime(2017,1,1,00)
	# hours = hours.days * 24 + hours.seconds//3600
	# print(hours, 720*9)
	per_stations = {i:j/720/9 for i,j in per_stations.items()}
	return per_stations # per_hour_mean_usage

def add_dict(self,other):
	ret_dict = {}
	union_keys = set(self.keys()).union(other.keys())
	intersec_keys = set(self.keys()).intersection(other.keys())
	for k in union_keys:
		if k in intersec_keys:
			ret_dict[k] = self[k] + other[k]
		elif k in other.keys():
			ret_dict[k] = other[k]
		else:
			ret_dict[k] = self[k]
	return ret_dict

def test_add_dict():
	a = {i:i for i in range(10)}
	b = {j:j for j in range(5,15)}
	print(add_dict(a,b))


def get_valid_station_list():
	df = load_monthly_data()
	per_stations = calc_usage(df)
	ret_list = [k for k in per_stations.keys() if per_stations[k] >= 1]
	return ret_list
	###### test code ######
	# valid_stations = get_valid_station_list()
	# print(valid_stations)
	# print(len(valid_stations))

def load_up_adjacency_matrix(save_path='refined_data/up_adj.dict'):
	from collections import Counter 
	import numpy as np
	# adjacency matrix도 load하기 (type : up or idw)
	# make adj dict of format : {(from_station_id, to_station_id): number of transactions during period}
	print("Loading fromto data...")
	path = os.path.join(os.getcwd(), 'dataset','fromto_2017_{}.csv')
	dfs = [pd.read_csv(path.format(i), index_col=None, encoding='cp949') for i in range(1,8)]


	for df in dfs: clean_colname(df)
	
	cat_df = pd.concat(dfs, axis=0)

	print('Getting the List Of Valid Stations...')
	valid_stations = get_valid_station_list()
	valid_stations.sort()
	valid_stations_id = {int(item.strip("'")):valid_stations.index(item) for item in valid_stations}
	print("station_mapping: ",valid_stations_id)

	valid_stations_int = [int(item.strip("'")) for item in valid_stations]



	valid_mask = cat_df["대여대여소번호"].isin(valid_stations + valid_stations_int+['{}'.format(item) for item in valid_stations_int]) & cat_df["반납대여소번호"].isin(valid_stations + valid_stations_int+['{}'.format(item) for item in valid_stations_int])
	cat_df = cat_df[valid_mask]

	# print(cat_df['대여일시'])

	print('='*30,"Stats",'='*30)
	print("columns : {}".format(cat_df.columns))
	print("num of relavant stations : {}".format(len(valid_stations)))
	print("dataframe size : {}".format(cat_df.shape))
	print('='*67)
	
	_month = [int(item.strip("'").split('-')[1]) for item in cat_df['대여일시'].values]
	print('demand per month: ',Counter(_month))
	# exit()
	_from = [int(item.strip("'")) if isinstance(item,str) else int(item) for item in cat_df['대여대여소번호'].values ]
	_to = [int(item.strip("'")) if isinstance(item,str) else int(item) for item in cat_df['반납대여소번호'].values ]

	adj_dict = {}
	for month in range(2,8): # From Fab to July
		adj = np.zeros((171,171))
		for m,x,y in zip(_month,_from,_to):
			if m != month: continue
			x,y = valid_stations_id[x], valid_stations_id[y]
			adj[x,y] += 1
		print(adj)
		adj_dict.update({month : adj})
	with open(save_path,'wb') as f:
		pickle.dump(adj_dict, f)
	print("saved result!")
	return adj_dict

def load_hourly_time_series_data(save_path='refined_data/hourly_time_series.df'):
	# load time series data : x-axis(time), y-axis(station_id)
	if os.path.exists(save_path):
		return pickle.load(open(save_path,'rb'))	
	else:
		from collections import Counter
		


		#load list of valid stations and their ids
		valid_stations = get_valid_station_list()
		valid_stations.sort()
		valid_stations_id = {item:valid_stations.index(item) for item in valid_stations}

		hour_df = load_hourly_data()
		print("original size : {}".format(hour_df.shape))
		hour_df = hour_df[hour_df["대여소번호"].isin(valid_stations)]
		print("filtered size : {}".format(hour_df.shape))

		# print(hour_df[hour_df["대여일자"]=="'2017-01-01'"])
		day_list = ["'2017-{:02d}-{:02d}'".format(i,j) for i in range(1,10) for j in range(1,32)]
		remove_invalid_day_list(day_list)
		# print(day_list)
		hour_list = ["'{:02d}'".format(i) for i in range(0,24)]
		# print(hour_list)

		day_plus_hour = ["{}-{}".format(d.strip("'"),h.strip("'")) for d in day_list for h in hour_list]
		# print(day_plus_hour[:100])
		# exit()
		time_series_df = pd.DataFrame(data=0, index = range(len(valid_stations)), columns = day_plus_hour)
		# print(time_series_df)
		# exit()
		for day in tqdm(day_list):
			for hour in hour_list:
				day_hour_name = "{}-{}".format(day.strip("'"),hour.strip("'"))
				df_now = hour_df[(hour_df["대여일자"]==day) & (hour_df["대여시간"]==hour)]
				transaction_now = [valid_stations_id[item] for item in df_now["대여소번호"].values]
				tr_cnt = Counter(transaction_now)
				# print(tr_cnt)
				# exit()
				time_series_df[day_hour_name] = [tr_cnt[k] if k in tr_cnt else 0 for k in range(len(valid_stations))]
				# print(time_series_df[day_hour_name])
				# exit()
		# print(sample_df)

		print("saving {}...".format(save_path))
		with open(save_path,'wb') as f:
			pickle.dump(time_series_df,f)

		return time_series_df

def load_daily_time_series_data(save_path='refined_data/daily_time_series.df'):
	# get time series based on daily transaction
	if os.path.exists(save_path):
		return pickle.load(open(save_path,'rb'))
	else:
		from collections import Counter
		


		#load list of valid stations and their ids
		valid_stations = get_valid_station_list()
		valid_stations.sort()
		valid_stations_id = {item:valid_stations.index(item) for item in valid_stations}

		hour_df = load_hourly_data()
		print("original size : {}".format(hour_df.shape))
		hour_df = hour_df[hour_df["대여소번호"].isin(valid_stations)]
		print("filtered size : {}".format(hour_df.shape))

		day_list = ["'2017-{:02d}-{:02d}'".format(i,j) for i in range(1,10) for j in range(1,32)]
		remove_invalid_day_list(day_list)

		time_series_df = pd.DataFrame(data=0, index = range(len(valid_stations)), columns = [item.strip("'") for item in day_list])

		for day in tqdm(day_list):
			day_name = day.strip("'")
			df_now = hour_df[(hour_df["대여일자"]==day)]
			transaction_now = [valid_stations_id[item] for item in df_now["대여소번호"].values]
			tr_cnt = Counter(transaction_now)
			time_series_df[day_name] = [tr_cnt[k] if k in tr_cnt else 0 for k in range(len(valid_stations))]

		print("saving {}...".format(save_path))
		with open(save_path,'wb') as f:
			pickle.dump(time_series_df,f)

		return time_series_df 

def load_weekly_time_series_data(save_path='refined_data/weekly_time_series.df'):
	# get weekly time series based on weekly transactions
	from collections import OrderedDict 

	if os.path.exists(save_path):
		return pickle.load(open(save_path,'rb'))
	else:
		from collections import Counter
		


		#load list of valid stations and their ids
		valid_stations = get_valid_station_list()
		valid_stations.sort()
		valid_stations_id = {item:valid_stations.index(item) for item in valid_stations}

		hour_df = load_hourly_data()
		print("original size : {}".format(hour_df.shape))
		hour_df = hour_df[hour_df["대여소번호"].isin(valid_stations)]
		print("filtered size : {}".format(hour_df.shape))

		day_list = ["'2017-{:02d}-{:02d}'".format(i,j) for i in range(1,10) for j in range(1,32)]
		remove_invalid_day_list(day_list)
		week_dict = OrderedDict([("{}-{}".format(day_list[i].strip("'"), day_list[i+7].strip("'")),day_list[i:i+7]) for i in range(0,len(day_list)-7,7)])

		time_series_df = pd.DataFrame(data=0, index = range(len(valid_stations)), columns = week_dict.keys())

		for week in tqdm(week_dict.keys(), total=len(week_dict)):
			week_name = week
			df_now = hour_df[(hour_df["대여일자"].isin(week_dict[week]))]
			transaction_now = [valid_stations_id[item] for item in df_now["대여소번호"].values]
			tr_cnt = Counter(transaction_now)
			time_series_df[week_name] = [tr_cnt[k] if k in tr_cnt else 0 for k in range(len(valid_stations))]

		print("saving {}...".format(save_path))
		with open(save_path,'wb') as f:
			pickle.dump(time_series_df,f)

		return time_series_df 


def remove_invalid_day_list(day_list):
	to_remove = ["2017-02-29","2017-02-30","2017-02-31",
				"2017-04-31", "2017-06-31","2017-09-31"]
	to_remove = ["'{}'".format(item) for item in to_remove]
	for item in to_remove:
		day_list.remove(item)

def clean_colname(df):
	df.columns = [item.strip("'") for item in df.columns]

def parse_rainfall(date_string, rainfall):
	y_m_d, h = date_string.split()
	y,m,d = [int(item) for item in y_m_d.split("-")]
	h = int(h.split(":")[0])
	if m in (1,2,3,11,12):
		# winter season --> record every 3 hours
		target_dates = [datetime(y,m,d,h) - timedelta(hours=i) for i in range(3,0,-1)]
		return_dict = {"{}-{:02d}-{:02d}-{:02d}".format(date.year,date.month,date.day,date.hour):rainfall/3 for date in target_dates}
	else:
		# non-winter season --> record every 1 hour 
		target_dates = [datetime(y,m,d,h) - timedelta(hours=1)]
		return_dict = {"{}-{:02d}-{:02d}-{:02d}".format(date.year,date.month,date.day,date.hour):rainfall for date in target_dates}

	return return_dict

def load_rainfall():
	if os.path.exists('refined_data/rainfall.dict'):
		rainfall_dict = pickle.load(open('refined_data/rainfall.dict','rb'))
	else:
		
		path = 'dataset/rainfall_201701_09.csv'
		rainfall_df = pd.read_csv(path,index_col=None, encoding='cp949')
		dates = rainfall_df['일시'].values
		rainfall = rainfall_df['강수량(mm)'].values
		rainfall_dict = {}
		for i in range(len(dates)):
			rainfall_dict.update(parse_rainfall(dates[i], rainfall[i]))

		# make day-hour list
		day_list = ["'2017-{:02d}-{:02d}'".format(i,j) for i in range(1,10) for j in range(1,32)]
		remove_invalid_day_list(day_list)
		hour_list = ["'{:02d}'".format(i) for i in range(0,24)]
		day_plus_hour = ["{}-{}".format(d.strip("'"),h.strip("'")) for d in day_list for h in hour_list]

		leftover_record_dict = {date_string:0. for date_string in day_plus_hour if date_string not in rainfall_dict.keys()}
		rainfall_dict.update(leftover_record_dict)

		with open('refined_data/rainfall.dict','wb') as f:
			pickle.dump(rainfall_dict, f)

	return rainfall_dict

def load_is_weekend():
	
	def load_tuple_for_datetime_string(datetime_string):
		if string_to_datetime(datetime_string).weekday() in (5,6):
			return (1,0)
		else:
			return (0,1)

	if os.path.exists('refined_data/is_weekend.dict'):
		is_weekend_dict = pickle.load(open('refined_data/is_weekend.dict','rb'))
	else:
		# make day-hour list
		
		day_list = ["'2017-{:02d}-{:02d}'".format(i,j) for i in range(1,10) for j in range(1,32)]
		remove_invalid_day_list(day_list)
		hour_list = ["'{:02d}'".format(i) for i in range(0,24)]
		day_plus_hour = ["{}-{}".format(d.strip("'"),h.strip("'")) for d in day_list for h in hour_list]

		is_weekend_dict = {item:load_tuple_for_datetime_string(item) for item in day_plus_hour}
		with open('refined_data/is_weekend.dict','wb') as f:
			pickle.dump(is_weekend_dict, f)

	return is_weekend_dict

def datetime_to_string(datetime_object, mode='hour'):
	if mode =='hour':
		return "{}-{:02d}-{:02d}-{:02d}".format(datetime_object.year, datetime_object.month, datetime_object.day, datetime_object.hour)
	elif mode == 'day':
		return "{}-{:02d}-{:02d}".format(datetime_object.year, datetime_object.month, datetime_object.day)
	elif mode =='week':
		weekend_date = datetime_object - timedelta(days=datetime_object.weekday()+1) if datetime_object.weekday() != 6 else datetime_object # If sunday, use it. If not, use previous sunday.
		seven_day_later = weekend_date + timedelta(days=7)
		return "{}-{:02d}-{:02d}-{}-{:02d}-{:02d}".format(weekend_date.year, weekend_date.month, weekend_date.day, seven_day_later.year, seven_day_later.month, seven_day_later.day)

def prev(date, mode='hour'):
	if mode=='hour':
		return [datetime_to_string(date-timedelta(hours=i), mode='hour') for i in range(4,0,-1)]
	elif mode =='day':
		return [datetime_to_string(date-timedelta(days=i), mode='day') for i in range(3,0,-1)]
	elif mode =='week':
		return [datetime_to_string(date-timedelta(days=i*7), mode='week') for i in range(3,0,-1)]


def build_taesan_data():
	# import pandas as pd
	x_hour = pickle.load(open('refined_data/hourly_time_series.df','rb')) # previous 4 timestep
	x_day = pickle.load(open('refined_data/daily_time_series.df','rb')) # previous 3 timestep
	x_week = pickle.load(open('refined_data/weekly_time_series.df','rb')) # previous 3 timestep
	rainfall = pickle.load(open('refined_data/rainfall.dict','rb'))
	rainfall_df = pd.DataFrame(rainfall.values(), columns=['rainfall'], index=rainfall.keys()).transpose()
	adj_dict = pickle.load(open('refined_data/up_adj.dict','rb'))
	is_weekend = pickle.load(open('refined_data/is_weekend.dict','rb'))
	is_weekend_df = pd.DataFrame(is_weekend.values(), columns=['is_weekend','is_weekday'], index=is_weekend.keys()).transpose()

	# print(list(rainfall.keys()))
	# exit()

	# make data per month
	milestone = [datetime(2017,i,1,00) for i in range(2,11)]
	data_dict = {}
	for i in range(6):
		_data_list = []
		for k in tqdm(range(3),total=3):
			hour_in = milestone[i+k+1] - milestone[i+k]
			hour_in = hour_in.days*24 + hour_in.seconds//3600
			_x_hour = [x_hour[prev(milestone[i+k]+timedelta(hours=j), mode='hour')].values for j in range(hour_in)]
			_x_day = [x_day[prev(milestone[i+k]+timedelta(hours=j), mode='day')].values for j in range(hour_in)]
			_x_week = [x_week[prev(milestone[i+k]+timedelta(hours=j), mode='week')].values for j in range(hour_in)]
			_rainfall = [rainfall_df[prev(milestone[i+k]+timedelta(hours=j), mode='hour')].values for j in range(hour_in)]
			_is_weekend = [is_weekend_df[prev(milestone[i+k]+timedelta(hours=j), mode='hour')].values for j in range(hour_in)]
			_target = [x_hour[datetime_to_string(milestone[i+k]+timedelta(hours=j), mode='hour')].values for j in range(hour_in)]

			_data_list.append([np.array(_x_hour), np.array(_x_day), np.array(_x_week),
								np.array(_rainfall),np.array(_is_weekend), adj_dict[i+2], np.array(_target)])
		

		data_dict.update({i+2 : _data_list})

		# exit()
	with open('refined_data/taesan_data.dict','wb') as f:
		pickle.dump(data_dict, f)

def prev_our(date, mode='short'):
	if mode=='short':
		return [datetime_to_string(date-timedelta(hours=i), mode='hour') for i in range(24,0,-1)]
	if mode=='long':
		return [datetime_to_string(date-timedelta(days=i), mode='hour') for i in range(21,0,-1)]

def build_our_data():
	# import pandas as pd
	x_hour = pickle.load(open('refined_data/hourly_time_series.df','rb')) # previous 4 timestep
	rainfall = pickle.load(open('refined_data/rainfall.dict','rb'))
	rainfall_df = pd.DataFrame(rainfall.values(), columns=['rainfall'], index=rainfall.keys()).transpose()
	adj_dict = pickle.load(open('refined_data/up_adj.dict','rb'))

	# make data per month
	milestone = [datetime(2017,i,1,00) for i in range(2,11)]
	data_dict = {}
	for i in range(6):
		_data_list = []
		for k in tqdm(range(3),total=3):
			hour_in = milestone[i+k+1] - milestone[i+k]
			hour_in = hour_in.days*24 + hour_in.seconds//3600
			_x_hour_short = [x_hour[prev_our(milestone[i+k]+timedelta(hours=j), mode='short')].values for j in range(hour_in)]
			_x_hour_long = [x_hour[prev_our(milestone[i+k]+timedelta(hours=j), mode='long')].values for j in range(hour_in)]
			_hour_code = [(milestone[i+k]+timedelta(hours=j)).hour for j in range(hour_in)]
			_day_code = [int((milestone[i+k]+timedelta(hours=j)).weekday() in (5,6)) for j in range(hour_in)]
			_location_code = [list(range(171)) for j in range(hour_in)]
			_rainfall = [rainfall_df[prev_our(milestone[i+k]+timedelta(hours=j), mode='short')].values for j in range(hour_in)]
			_target = [x_hour[datetime_to_string(milestone[i+k]+timedelta(hours=j), mode='hour')].values for j in range(hour_in)]

			_data_list.append([np.array(_x_hour_short), np.array(_x_hour_long), np.array(_hour_code), np.array(_day_code),
								np.array(_location_code), np.array(_rainfall), adj_dict[i+2], np.array(_target)])
		

		data_dict.update({i+2 : _data_list})

		# exit()
	with open('refined_data/our_data.dict','wb') as f:
		pickle.dump(data_dict, f)


if __name__ == '__main__':
	if not os.path.exists('refined_data'): os.mkdir('refined_data')
	print('Preprocessing : It may take ~1 hour')
	load_hourly_time_series_data()
	load_up_adjacency_matrix()
	load_rainfall()
	build_our_data()