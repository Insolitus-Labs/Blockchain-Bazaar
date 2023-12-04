from flask import Flask, abort, request, send_from_directory
from flask_cors import CORS
import ssl
import requests
import json

app = Flask(__name__)

CORS(app)

host = "127.0.0.1"
port = 5000
debug = False

items = None
with open("items.json", "r") as f_in:
    items = json.loads(f_in.read())

@app.route("/", methods=['GET','POST'])
def index():
    return "Blockchain Bazzar"

@app.route("/get_all_items", methods=["GET"])
def api_get_items():
    global items
    for i in items:
        features, _ = parse_data(int(i['rarity']), int(i['level']), 
                                 int(i['weight']), int(i['defense']), 
                                 int(i['damage']), int(i['range']),
                                 int(i['speed']), 0)

        # request AI for prices
        buy_price = askQuery(features, 1) # Buy
        sell_price = askQuery(features, 0) # Sell
        
        i["buy_price"] = buy_price / 10000   # Eth
        i["sell_price"] = sell_price / 10000 # Eth
        
        i["uri"] = f"item={i['item']}&rarity={i['rarity']}&level={i['level']}&weight={i['weight']}&defense={i['defense']}&damage={i['damage']}&range={i['range']}&speed={i['speed']}"

    return items


@app.route("/price", methods=['GET'])
def api_price():

    is_buy = int(request.args.get('order_type'))

    data = request.args

    features, _ = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), 0)

    # request AI for prices
    price = askQuery(features, is_buy) * (10**18) / 10000 # WEI

    return {"price" : price}


@app.route("/smart_contract_price", methods=['GET'])
def api_sc_price():

    print("request:",request.args)
    eth_sent = int(request.args.get('eth'))
    is_buy = int(request.args.get('order_type'))
    
    data = request.args
    
    features, _ = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), 0)
    
    # request AI for prices
    price = askQuery(features, is_buy) * (10Z**18) / 10000 # WEI
    
    if(price <= eth_sent):
        features, label = parse_data(int(data['rarity']), int(data['level']), 
                             int(data['weight']), int(data['defense']), 
                             int(data['damage']), int(data['range']),
                             int(data['speed']), price)
        giveData(features, label, is_buy)
        # AI.update_price(buy_or_sell,damage,weight)
    
    return {"price" : price,"refund" : eth_sent - price}


import re, numpy as np
from contextualbandits.online import LinUCB

# batch size - algorithm will be refit after N rounds
# used during pre-training
batch_size = 50
# there are 100 outputs possible by the model
nchoices = 100

# models, action history, and rewards are stored in a map with key as modelId
# create both the map and global variables to allow for either way of calling
# for debugging and testing purposes

''' explanation of parameters:
nchoices              - sets number of arms, 100 arms used, each representing the 10 closest values for a total of 1000b

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

# function to run through a single round given one piece of data and its label
# modelId is 0 for seller and 1 for buyer
def single_round(data, label, modelId):
    
    # choosing actions for this batch
    action = models[modelId].predict(data).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards[modelId].append(label[action])
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions[modelId], action)
    
    # rewards obtained now
    reward = label[action]
    
    # now refitting the algorithms after observing these new rewards
    models[modelId].partial_fit(data, action, reward)
    
    return new_actions_hist

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
    top5[::-1].sort()
  else:
    top5.sort()
  print(top5)
  answer = 0
  for i in range(5):
    if i < 3:
      answer += top5[0,i] / 2**(i + 1)
    else:
      answer += top5[0,i] / 2**(i)
  return(round(answer))

# given data (features of an item), a label (array with binary label data) and a modelId
# runs a single round (partial fit with new data) with batch size 1 (only this new data) on the specified model
def giveData(data, label, modelId):
  single_round(data, label, modelId)


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
  print(model)
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


if __name__ == "__main__":
    app.run(host=host, port=port, debug=debug, ssl_context=None)


