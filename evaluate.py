# import keras
# from keras.models import load_model
import torch
from agent.agent import Agent
from functions import *
import sys
from utils import *
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if len(sys.argv) != 3:
	print ("Usage: python evaluate.py [stock] [model]")
	exit()
stock_name, model_name = sys.argv[1], sys.argv[2]
PATH = "models/" + model_name+'.pt'
window_size = 10
num_ensemble = 3
prior_variance=0.1
noise_variance=0.1

agent = Agent(window_size,prior_variance,noise_variance,num_ensemble,PATH)
data = getStockDataVec(stock_name)
data = data[:100]
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

print(len(data))
plt.axis([0, len(data)-1, min(data), max(data)])

firstbuy = True
firstsell = True
for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print ("Buy: " + formatPrice(data[t]))
		if firstbuy == True:
			plt.scatter(t, data[t],s=20,c='b',marker='.', linewidths=2, label="buy")
			firstbuy = False
		else:
			plt.scatter(t, data[t],s=20,c='b',marker='.', linewidths=2)
		plt.pause(0.01)

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
		if firstsell == True:
			plt.scatter(t, data[t],s=20,c='r',marker='.', linewidths=2, label="sell")
			firstsell = False
		else:
			plt.scatter(t, data[t],s=20,c='r',marker='.', linewidths=2)
		plt.pause(0.01)

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print ("--------------------------------")
		plt.plot(range(l),data[:-1], c='k', linewidth=1.0)
		plt.xlabel("data_idx")
		plt.ylabel("Reward")
		plt.title(f'{stock_name}, Total Reward = {formatPrice(total_profit)}')
		plt.legend()
		plt.savefig(f'{stock_name}_res.pdf')
