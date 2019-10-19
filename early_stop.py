import sys

class EarlyStopping:
	def __init__(self, patience):
		self.patience = patience 
		self.valid_losses = []
		self.test_losses = []
		self.counter = 0
		self.best_loss = None
		self.best_index = None
		self.stop_train = False

	def __call__(self, valid_loss, test_loss):
		# record scores
		self.valid_losses.append(valid_loss)
		self.test_losses.append(test_loss)
		current_index = len(self.valid_losses) - 1

		# check criterion
		if self.best_loss is None:
			self.best_loss = valid_loss
			self.best_index = current_index

		elif valid_loss >= self.best_loss:
			self.counter += 1
			print("EarlyStop : {}/{}".format(self.counter,self.patience))
			if self.counter >= self.patience:
				self.stop_train = True
		else:
			self.best_index = current_index
			self.best_loss = valid_loss 
			self.counter = 0

	def best_valid_and_test_loss(self):
		return self.valid_losses[self.best_index], self.test_losses[self.best_index]