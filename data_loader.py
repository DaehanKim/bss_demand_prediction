from datetime import datetime, timedelta 
import pandas as pd
import numpy as np
import pickle
import torch

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std+1e-7)

    def inverse_transform(self, data):
        return data * self.std + self.mean

class Model7Dataloader:
	def __init__(self, month, batch_size):
		self.batch_size = batch_size
		# key=month, value=dict format
		# data order : _x_hour, _x_day, _x_week,_rainfall,_is_weekend, adj_dict[i+2]  
		self.data_list = pickle.load(open('refined_data/our_data.dict','rb'))[month] 
		self.scalers = self.get_scalers()
		# self.normalize_features()

	def normalize_features(self):
		for j in range(3): # train/val/test
			for i in (0,1,5): # x_hour_short/x_hour_long/rainfall
				self.data_list[j][i] = self.scalers[i].transform(self.data_list[j][i])

	def get_scalers(self):
		train_features = self.data_list[0]
		return {i:StandardScaler(train_features[i].mean(),train_features[i].std()) for i in (0,1,5)}


	def get_iter(self, mode='train',shuffle=False):
		# get hourly, daily, weekly dataloader
		# can use record in 1/22~(-4 hours/-3 days/-3 weeks are used as history) 

		if mode == 'train': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[0]
		elif mode =='valid': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[1]
		elif mode =='test': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[2]
		else: assert False,"mode should be among train/valid/test"

		batch_num = x_hour_short.shape[0]//self.batch_size
		is_residual = (x_hour_short.shape[0] % self.batch_size != 0)

		for i in range(batch_num):
			yield {'x_hour_short':torch.FloatTensor(x_hour_short[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'x_hour_long':torch.FloatTensor(x_hour_long[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'hour_code': torch.LongTensor(hour_code[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'day_code': torch.LongTensor(day_code[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'location_code': torch.LongTensor(location_code[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'rainfall': torch.FloatTensor(rainfall[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'target':torch.FloatTensor(target[i*self.batch_size:(i+1)*self.batch_size]).cuda(),
					'adj':torch.FloatTensor(adj_dict).cuda()}

		if is_residual:
			yield {'x_hour_short':torch.FloatTensor(x_hour_short[batch_num*self.batch_size:]).cuda(),
					'x_hour_long':torch.FloatTensor(x_hour_long[batch_num*self.batch_size:]).cuda(),
					'hour_code': torch.LongTensor(hour_code[batch_num*self.batch_size:]).cuda(),
					'day_code': torch.LongTensor(day_code[batch_num*self.batch_size:]).cuda(),
					'location_code': torch.LongTensor(location_code[batch_num*self.batch_size:]).cuda(),
					'rainfall': torch.FloatTensor(rainfall[batch_num*self.batch_size:]).cuda(),
					'target':torch.FloatTensor(target[batch_num*self.batch_size:]).cuda(),
					'adj':torch.FloatTensor(adj_dict).cuda()}

	def get_batch_num(self, mode='train'):
		if mode == 'train': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[0]
		elif mode =='valid': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[1]
		elif mode =='test': x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall, adj_dict, target = self.data_list[2]
		else: assert False,"mode should be among train/valid/test"

		batch_num = x_hour_short.shape[0]//self.batch_size
		self.is_residual = (x_hour_short.shape[0] % self.batch_size != 0)

		return batch_num