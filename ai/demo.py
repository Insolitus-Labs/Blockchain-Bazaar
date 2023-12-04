# This is a non flask version of the final code which
# also includes several graphs to show how the AI is running

import numpy as np
from copy import deepcopy
from contextualbandits.online import LinUCB

# batch size - algorithm will be refit after N rounds
# used during pre-training
batch_size = 50
# there are 100 outputs possible by the model
nchoices = 50

# models, action history, and rewards are stored in a map with key as modelId
# create both the map and global variables to allow for either way of calling
# for debugging and testing purposes

''' explanation of parameters:
nchoices              - sets number of arms, 100 arms used, each representing the 10 closest values for a total of 1000

beta_prior            - used to set the way arms behave
                          'auto' makes them choose from a random distribution when they don't have much data

alpha                 - controls upper confidence bound, higher values increase exploration

ucb_from_empty        - controls whether arms without data are chosen from based on policy
                          we use bet_prior for better control of this so it isn't needed

random_state          - sets the random seed for this model, used in exploration

assume_unique_reward  - causes the model to assume there can only be one label for each data point
                          this creates negative labels for other arms to also fit from
'''
seller = LinUCB(nchoices = nchoices, beta_prior = 'auto', alpha = 0.1,
                ucb_from_empty = False, random_state = 1111, assume_unique_reward = True)
buyer = LinUCB(nchoices = nchoices, beta_prior = 'auto', alpha = 0.1,
                ucb_from_empty = False, random_state = 2222, assume_unique_reward = True)

models = {
  0: seller,
  1: buyer
}

seller_actions = np.array(nchoices)
buyer_actions = np.array(nchoices)

actions = {
  0: seller_actions,
  1: buyer_actions
}

seller_rewards = list()
buyer_rewards = list()

rewards = {
  0: seller_rewards,
  1: buyer_rewards
}

# takes the stats of an item and returns the two arrays needed by the model
def parse_data(rarity, level, weight, defense, damage, a_range, speed, price):
  label = np.zeros(50)
  label[int( int( price / 20) ) ] = 1
  features = np.array([rarity, level, weight, defense, damage, a_range, speed])
  return features, label

# given a piece of data (features of an item) and a modelId (seller or buyer)
# returns the pricing from that model by combining the top 5 results
def askQuery(data, modelId):
  top5 = models[modelId].topN(data, 5) * 20

  if modelId == 0:
    top5[0][::-1].sort()
  else:
    top5[0].sort()
  
  answer = 0
  for i in range(5):
    if i < 4:
      answer += top5[0,i] / 2**(i + 1)
    else:
      answer += top5[0,i] / 2**(i)

  return(int(answer))

# given data (features of an item), a label (array with binary label data) and a modelId
# runs a single round (partial fit with new data) with batch size 1 (only this new data) on the specified model
def giveData(data, label, modelId):
    # choosing actions for this item
    action = models[modelId].predict(data).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards[modelId].append(label[action])
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions[modelId], action)
    
    # rewards obtained now
    reward = label[action]
    
    # now refitting the algorithms after observing these new rewards
    models[modelId].fit(data, action, reward)
    
    return new_actions_hist

# takes an int and returns the proper structure of the data for the model
def makeLabel(price):
  label = np.zeros(100)
  label[int( round( price / 20) ) ] = 1
  return label

################ Offline Pretraining  ################
# random data follows the pattern of label being the sum of all stats
# stats are generated with np.random
def create_data(amount):
  features = np.empty(shape=(amount, 7))
  labels = np.empty(shape=(amount, 50))
  for i in range(amount):
    type = np.random.randint(0,2)
    if type == 0: # defense i.e. armor
      rarity = np.random.randint(0, 4)
      level = np.random.randint(rarity * 5, rarity * 25 + 25)
      weight = np.random.randint(0, 100)
      defense = np.random.randint(rarity * 5, rarity * 25 + 25)
      price = rarity * 25 + level + weight * 2 + defense
      price = defense + level
      features[i], labels[i] = parse_data(rarity, level, weight, defense, 0, 0, 0, price)
    if type == 1: # offense i.e weapon
      rarity = np.random.randint(0, 4)
      level = np.random.randint(rarity * 5, rarity * 25 + 25)
      weight = np.random.randint(0, 100)
      damage = np.random.randint(rarity * 5, rarity * 25 + 25)
      a_range = np.random.randint(rarity * 5, rarity * 25 + 25)
      speed = np.random.randint(rarity * 5, rarity * 25 + 25)
      price = rarity * 25 + level + weight * 2 + damage + a_range + speed
      features[i], labels[i] = parse_data(rarity, level, weight, 0, damage, a_range, speed, price)
    if type == 2: # misc
      rarity = np.random.randint(0, 4)
      level = np.random.randint(rarity * 5, rarity * 25 + 25)
      weight = np.random.randint(0, 100)
      price = rarity * 25 + level + weight * 2
      features[i], labels[i] = parse_data(rarity, level, weight, 0, 0, 0, 0, price)
  return features, labels

def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_batch[np.arange(y_batch.shape[0]), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)
    
    return new_actions_hist

X, y = create_data(15000)

# fitting models for the first time
first_batch = X[:batch_size, :]
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

models[0].fit(X=first_batch, a=action_chosen, r=rewards_received)
actions[0] = action_chosen.copy()

action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

models[1].fit(X=first_batch, a=action_chosen, r=rewards_received)
actions[1] = action_chosen.copy()

# running all pre-training rounds
for model in range (2):
  print("Starting Training For Model: " + str(model))
  for i in range(int(np.floor(X.shape[0] / batch_size)) - 1):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    X_batch = X[batch_st:batch_end, :]
    y_batch = y[batch_st:batch_end, :]
    actions[model] = simulate_rounds_stoch(models[model],
                                                   rewards[model],
                                                   actions[model],
                                                   X_batch, y_batch,
                                                   rnd_seed = batch_st)
print("Finished Training")

################ Graph Creation  ################
# this graph shows the two models training and the average
# rewards they receive where 1 is perfect and 0.01 is random chance
import matplotlib.pyplot as plt
from pylab import rcParams

def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

lwd = 5
cmap = plt.get_cmap('tab20')
colors=plt.cm.tab20(np.linspace(0, 1, 20))
rcParams['figure.figsize'] = 15, 7

plt.plot(get_mean_reward(rewards[0]), label="Seller", linewidth=5,color=colors[0])
plt.plot(get_mean_reward(rewards[1]), label="Buyer", linewidth=5,color=colors[5])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':20})

plt.xlabel('Rounds (models were updated every 50 rounds)', size=30)
plt.ylabel('Cumulative Mean Reward', size=30)
plt.grid()
plt.show()
