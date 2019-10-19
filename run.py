import torch
from tqdm import tqdm

from preprocess_data import *
from trainer import Trainer
from model import *
from data_loader import * 
from early_stop import *

def print_result(wr):
	string = "train/val/test={}/{}/{} --> val/test_mse={:.6f}/{:.6f}"
	for i, tup in zip(range(2,8), wr):
		print(string.format(i,i+1,i+2, *tup))


def run():	
	# load model
	result = []
	for i in range(2,8):
		model = model_8_2(num_self_att = 4, num_heads = 4, keep_short=24,keep_long=7).cuda()
		ld = Model7Dataloader(month=i, batch_size = 100)
		adam = torch.optim.Adam(model.parameters(), lr=3e-4)
		es = EarlyStopping(patience=50)
		tr = Trainer(model=model, dataloader = ld, early_stopper = es, optimizer = adam)

		result.append(tr.train())

	print_result(result)


if __name__ == '__main__':
	run()
