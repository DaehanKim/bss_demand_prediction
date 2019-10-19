import torch
from tqdm import tqdm

from preprocess_data import *
from data_loader import * 
from early_stop import *
from model import *


class Trainer:
	def __init__(self, model, dataloader, early_stopper, optimizer):
		self.model = model 
		self.dataloader = dataloader
		self.early_stopper = early_stopper
		self.optimizer = optimizer
		self.scalers = dataloader.scalers
		self.model.set_adj(torch.FloatTensor(self.dataloader.data_list[0][-2]).cuda())
		self.model.normalize_adj() # compute normalized adjacency matrix


	def eval(self, mode): # compute mse over all test period
		mse, cnt = 0.0, 0
		self.model.eval()

		for batch in self.dataloader.get_iter(mode=mode):
			pred = self.model(**{k:v for k,v in batch.items() if k not in ('target','adj')}) # NEED TO match key of batch dict and model input variable names
			# pred = self.scalers[0].inverse_transform(pred)
			loss = (pred-batch['target']).pow(2).mean()*batch['target'].size(0)
			mse += loss.item()
			cnt += batch['target'].size(0)
		return mse/cnt 

	def train(self):
		# compute batch number
		batch_num = self.dataloader.get_batch_num(mode='train')
		if self.dataloader.is_residual: batch_num += 1


		for epoch in range(500):
			for current_batch_num, batch in enumerate(self.dataloader.get_iter(mode='train', shuffle=True)): 
				self.model.train()
				self.optimizer.zero_grad()
				pred = self.model(**{k:v for k,v in batch.items() if k not in ('target','adj')}) # NEED TO match key of batch dict and model input variable names
				# pred = self.scalers[0].inverse_transform(pred)
				loss = (pred-batch['target']).pow(2).mean()
				loss.backward()
				self.optimizer.step()

				valid_mse, test_mse = self.eval(mode='valid'), self.eval(mode='test')
				print("batch {}/{} | train/valid/test_mse : {:.8f}/{:.8f}/{:.8f}".format(current_batch_num+1, batch_num, loss.item(), valid_mse, test_mse))
				self.early_stopper(valid_mse, test_mse) # early stopping criterion check
				if self.early_stopper.stop_train: 
					best_result = self.early_stopper.best_valid_and_test_loss()
					print("Training ended : best_valid/test={:.6f}/{:.6f}".format(*best_result))
					return best_result

		best_result = self.early_stopper.best_valid_and_test_loss()
		print("Training ended : best_valid/test={:.6f}/{:.6f}".format(*best_result))	
		return best_result

