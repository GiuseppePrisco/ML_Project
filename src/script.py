
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from collections import deque

from tqdm import tqdm
import winsound
import datetime

# used for plots
import csv
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean


# hyperparameters
n_episodes = 2000
# learning_rate = 0.00025
# learning_rate = 0.001
start_epsilon = 1.0
final_epsilon = 0.01
# epsilon_decay = 0.999 first attempt - not dependent on episode number
epsilon_decay = (start_epsilon - final_epsilon) / n_episodes # linear decay
# epsilon_decay = (final_epsilon/start_epsilon)**(1/n_episodes) # exponential decay
discount_factor = 0.95

# past experience parameters
EXP_MAX_SIZE=4000
BATCH_SIZE=int(64+0.25*64) # batch size used for training
experience = deque([],EXP_MAX_SIZE) 


# class denoting the agent
class CartPoleAgent:
	def __init__(
        self,
        initial_epsilon: float,
        final_epsilon: float,
		epsilon_decay: float,
        discount_factor: float
    ):	

		self.epsilon = initial_epsilon
		self.final_epsilon = final_epsilon
		self.epsilon_decay = epsilon_decay
		self.discount_factor = discount_factor


def setup_network():
	
	# neural network architecture
	model = Sequential()
	model.add(Dense(64, input_shape=(4,), activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(2,activation='linear'))
	model.compile(optimizer = "adam", loss="mse", metrics=["accuracy"])
	return model
	

def train_episodes():

	# initialize agent
	agent = CartPoleAgent(
		initial_epsilon = start_epsilon,
		final_epsilon = final_epsilon,
		epsilon_decay = epsilon_decay,
		discount_factor = discount_factor,
	)
	
	model = setup_network()
	model.summary()

	# create environment
	env = gym.make("CartPole-v1")

	# create csv file to store relevant episode data
	with open('data.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["episode", "epsilon", "score", "time"])


	# episodes loop
	for episode in tqdm(range(n_episodes)):
		
		start_time = datetime.datetime.now()
		
		# get first observation of the environment
		observation, info = env.reset()
	
		terminated = truncated = False
	
		c_reward = 0
		
		# perform one episode
		while not (terminated or truncated):
		
			# with probability epsilon return a random action to explore the environment (exploration)
			if np.random.random() < agent.epsilon:
				act = env.action_space.sample()

			# with probability (1 - epsilon) act greedily (exploitation)
			else:
				act = np.argmax(model.predict_on_batch(observation.reshape(1,4)))
			
			# perform the action
			next_observation, reward, terminated, truncated, info = env.step(act)

			c_reward+= reward

			# record experience, used for training the neural network
			if len(experience)>=EXP_MAX_SIZE:
				experience.popleft()
			experience.append((observation, act, reward, next_observation, terminated, truncated))

			# update the current state
			observation = next_observation 
		
		# update the epsilon at the end of the episode
		agent.epsilon = max(agent.final_epsilon, agent.epsilon - epsilon_decay) # linear decay
		# agent.epsilon = max(agent.final_epsilon, agent.epsilon*epsilon_decay) # exponential decay


		# train the neural network when the experience list contains at least BATCH_SIZE records
		if len(experience) >= BATCH_SIZE:
			# sample batch
			batch = random.sample(experience, BATCH_SIZE)
			
			# prepare datasets
			currentObservationBatch = np.zeros((BATCH_SIZE, 4))
			nextObservationBatch = np.zeros((BATCH_SIZE, 4))
			
			for i, tuple in enumerate(batch):
				# take current observation
				currentObservationBatch[i,:] = tuple[0]
	
				# take next observation
				nextObservationBatch[i,:] = tuple[3]
				
			# initialize the input and output for the neural network
			X = currentObservationBatch			
			Y = np.zeros((BATCH_SIZE, env.action_space.n))
		
			# use the neural network to predict the Q-values
			target = model.predict_on_batch(currentObservationBatch)
			targetNext = model.predict_on_batch(nextObservationBatch)
		
			for i,(observation,act,reward,next_observation,terminated,truncated) in enumerate(batch):
		
				# when the the next observation is a terminal state, simply assign the reward
				if terminated:
					y = reward
				
				# when the next observation is NOT a terminal state, use the predicted next reward
				else:
					y = reward + agent.discount_factor*np.max(targetNext[i])
				
				# assign the target output 
				Y[i] = target[i]
				Y[i,act] = y
					
			#train the neural network
			model.fit(X, Y, batch_size = BATCH_SIZE, validation_split = 0.25, verbose = 0)

		# print debug information
		print("----------------------------------episode ", episode)
		print("epsilon = ", agent.epsilon)
		print("experience size = ", len(experience))
		print("cumulative reward = ", c_reward)
		
		end_time = datetime.datetime.now()
		elapsed_time = end_time - start_time
		
		# append entry in the csv file
		with open('data.csv', 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([episode, agent.epsilon, c_reward, elapsed_time])
		
		
		# if c_reward == 500:
			# count = count + 1
		# if count > 15:
			# break
		# print("count=", count)
		
		# gradually update the model every 100 episodes
		# if episode % 100 == 0:
			# model.save("cartpole.keras")


	model.save("cartpole.keras")
	env.close()


# plot the data saved in the csv file
def show_results():
	
	episode = []
	epsilon = []
	reward = []
	time = []
	w = []
	x = []
	y = []
	z = []

	path = "data.csv"
	data = pd.read_csv(path)
	
	with open(path,'r') as file:
		next(file)
		lines = csv.reader(file, delimiter=',')
		
		count = 0
		
		for row in file:
			data = row.split(",")
			if count % 20 == 0 and count!=0 or count == len(data)-1:
				episode.append(round(mean(w),0))
				epsilon.append(round(mean(x),4))
				reward.append(round(mean(y),2))
				time.append(round(mean(z),4))
				w = []
				x = []
				y = []
				z = []
				
			w.append(int(data[0]))
			x.append(float(data[1]))
			y.append(float(data[2]))
			z.append(float(data[3].split(":")[1]+data[3].split(":")[2]))
			count+=1
	
	# plot for the epsilon data
	plt.plot(episode,epsilon,color="g",marker="o",markersize=5,markeredgecolor="k",markeredgewidth=0.5,label="Epsilon Data")
	plt.xlabel("Episodes")
	plt.ylabel("Epsilon")
	plt.title("Epsilon Report", fontsize = 20)
	plt.grid()
	plt.legend()
	plt.savefig("epsilon.png")
	plt.show()
	
	# plot for the reward data
	plt.plot(episode, reward, color = "r",label = "Reward Data")
	plt.xlabel("Episodes")
	plt.ylabel("Reward")
	plt.title("Reward Report", fontsize = 20)
	plt.grid()
	plt.legend()
	plt.savefig("reward.png")
	plt.show()
	
	# plot for the time data
	plt.plot(episode, time, color = "b",linestyle = "dashed",label = "Time Data")
	plt.xlabel("Episodes")
	plt.ylabel("Time")
	plt.title("Time Report", fontsize = 20)
	plt.grid()
	plt.legend()
	plt.savefig("time.png")
	plt.show()
	

# perform testing on the model
def test():

	# create environment 
	env = gym.make("CartPole-v1", render_mode = "human")
	
	# load the tested model
	model = tf.keras.saving.load_model("cartpole.keras")
	
	sum = 0
	count = 0
	
	# test the episodes
	for e in range(int(n_episodes/2)):
	
		observation, info = env.reset()
			
		terminated = truncated = False
		
		cumulative_reward = 0
		
		while not (terminated or truncated):
		
			# graphically show the environment
			env.render()
				
			# always choose the best action
			act = np.argmax(model.predict_on_batch(observation.reshape(1,4)))
			
			# perform the action
			observation, reward, terminated, truncated, info = env.step(act)
			
			# update the cumulative reward
			cumulative_reward+= reward
		
		# print relevant info at the end of the episode
		print("episode: {}/{}, score: {}".format(e+1, int(n_episodes/2), cumulative_reward))
		
		sum+= cumulative_reward
		print("average score = ", sum/(e+1))
		
		if cumulative_reward == 500:
			count = count + 1
		print("count = ", count)

	env.close()


if __name__ == "__main__":

	#train_episodes()
	
	frequency = 2500
	duration = 3000
	#winsound.Beep(frequency, duration)
	
	#show_results()
	
	test()


	

