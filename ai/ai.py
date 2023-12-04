from flask import Flask, request

# Lines of code to enter to start up flask app for debugging or testing
''' 
export FLASK_APP=online
export FLASK_ENV=development
flask run
'''

ai = Flask(__name__)

# this routine handles the seller model which the ai uses to price
# its own items to be sold to players
@ai.route('/sell')
def sell():
  data = request.args
  if len(data) == 7:
    features, _ = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), 0)
    return(str(askQuery(features, 0)))
  else:
    if len(data) == 8:
      features, label = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), int(data['price']))
      giveData(features, label, 0)
      return("Data received")
  return("Bad call")

# this handles the buyer model which handles how much the ai pays
# for items it buys from players
@ai.route('/buy')
def buy():
  data = request.args
  if len(data) == 7:
    features, _ = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), 0)
    return(str(askQuery(features, 1)))
  else:
    if len(data) == 8:
      features, label = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), int(data['price']))
      giveData(features, label, 1)
      return("Data received")
  return("Bad call")


import numpy as np
from contextualbandits.online import LinUCB

# batch size - algorithm will be refit after N rounds
# used during pre-training
batch_size = 50
# there are 100 outputs possible by the model
nchoices = 100

# models are stored in a map with key as modelId
# create both the map and global variables to allow for either way of calling
# for debugging and testing purposes

''' explanation of parameters:
nchoices              - sets number of arms, 100 arms used, each representing the 10 closest values for a total of 1000

beta_prior            - used to set the way arms behave
                          'auto' makes them choose from a random distribution when they don't have much data

alpha                 - controls upper confidence bound, higher values increase exploration, a lower value is recommended

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

# takes the stats of an item and returns the two arrays needed by the model
def parse_data(rarity, level, weight, defense, damage, a_range, speed, price):
  label = np.zeros(100)
  label[int( round( price / 10) ) ] = 1
  features = np.array([rarity, level, weight, defense, damage, a_range, speed])
  return features, label

# given a piece of data (features of an item) and a modelId (seller or buyer)
# currently prints but should return a weighted answer of the top 5 arms chosen
# top arm accoutns for 50% with each subsequent arm counting for less
def askQuery(data, modelId):
  top5 = models[modelId].topN(data, 5) * 10
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
  return(round(answer))

# given data (features of an item), a label (array with binary label data) and a modelId
# runs a single round (partial fit with new data) with batch size 1 (only this new data) on the specified model
def giveData(data, label, modelId):
    # choosing actions for this batch
    action = models[modelId].predict(data).astype('uint8')
    
    # rewards obtained now
    reward = label[action]
    
    # now refitting the algorithms after observing these new rewards
    models[modelId].partial_fit(data, action, reward)


################ Offline Pretraining  ################
# random data follows the pattern of label being the addition of all stats
# stats are generated with np.random
def create_data(amount):
  features = np.empty(shape=(amount, 7))
  labels = np.empty(shape=(amount, 100))
  for i in range(amount):
    type = np.random.randint(0,2)
    if type == 0: # defense i.e. armor
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      defense = np.random.randint(0, 99)
      price = rarity + level + weight + defense
      features[i], labels[i] = parse_data(rarity, level, weight, defense, 0, 0, 0, price)
    if type == 1: # offense i.e weapon
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      damage = np.random.randint(0, 99)
      a_range = np.random.randint(0, 99)
      speed = np.random.randint(0, 99)
      price = rarity + level + weight + damage + speed + a_range
      features[i], labels[i] = parse_data(rarity, level, weight, 0, damage, a_range, speed, price)
    if type == 2: # misc
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      price = rarity + level + weight
      features[i], labels[i] = parse_data(rarity, level, weight, 0, 0, 0, 0, price)
  return features, labels

def simulate_rounds_stoch(model, X_batch, y_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)

X, y = create_data(15000)

# fitting models for the first time
first_batch = X[:batch_size, :]
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

models[0].fit(X=first_batch, a=action_chosen, r=rewards_received)

action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

models[1].fit(X=first_batch, a=action_chosen, r=rewards_received)

# running all pre-training rounds
for model in range (2):
  for i in range(int(np.floor(X.shape[0] / batch_size)) - 1):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    X_batch = X[batch_st:batch_end, :]
    y_batch = y[batch_st:batch_end, :]
    simulate_rounds_stoch(models[model], X_batch, y_batch, rnd_seed = batch_st)
