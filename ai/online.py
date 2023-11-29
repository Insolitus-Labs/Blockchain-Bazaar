import re, numpy as np
from sklearn.linear_model import SGDClassifier
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import LinUCB

# batch size - algorithm will be refit after N rounds
# used during pre-training
batch_size = 50
# there are 100 outputs possible by the model
nchoices = 100
base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

# models, action history, and rewards are stored in a map with key as modelId
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

# takes a string and outputs the proper form of input for the ai model
# as well as two booleans to decide which model to give and if this is
# a query or training data
def parse_data(order):
  # order is a string in the form 
  # (for getting price) 'buy'       price something to be bought (available for user to 'buy')
  #                     'sell'      price something to be sold (available for user to 'sell')
  # (for giving data)   'bought'    give data to selling model (user has 'bought' something)
  #                     'sold'      give data to buying model (user has 'sold' something)
  #
  #                              ,rarity,level,weight,defense,damage,range,speed,price(closest),price(second)
  # i.e. "buy,35,0,40,32,19,0" or "sold,35,0,40,32,19,145"
  query = (order.startswith("buy") or order.startswith("sell"))
  buy = (order.startswith("b"))
  stats = [int(x) for x in re.sub(r'[A-Za-z]+,', '', order).split(",")]
  label = np.zeros(100)
  label[int( round( stats[7] / 10) ) ] = 1
  features = np.zeros(10)
  features[stats[0] + 1] = 1
  for i in range(1, 7):
    features[i + 3] = stats[i] / 100
  return features, label, query, buy

# a decision maker which calls the proper model and action
# does not return anything itself, instead having the functions it calls return the proper data
def decision(is_query, is_buy, data, label):
  if is_buy: # seller modelId = 0
    if is_query:
      askQuery(data, 0)
    else:
      giveData(data, 0)
  else: # buyer modelId = 1
    if is_query: 
      askQuery(data, 1)
    else:
      giveData(data, 1)

# given a piece of data (features of an item) and a modelId (seller or buyer)
# currently prints but should return a weighted answer of the top 5 arms chosen
# top arm accoutns for 50% with each subsequent arm counting for less
def askQuery(data, modelId):
  model = map(key = modelId, value = model) # map needs creation ---------------------------------------------------------------------------
  weights = model.topN(data, 5)
  answer = 0
  for i in range(5):
    if i < 4:
      answer = weights[i] / 2**(i + 1)
    else:
      answer = weights[i] / 2**(i)
  print(answer) # integration needed -------------------------------------------------------------------------------------------------------

# given data (features of an item), a label (array with binary label data) and a modelId
# runs a single round (partial fit with new data) with batch size 1 (only this new data) on the specified model
def giveData(data, label, modelId):
  single_round(data, label, modelId)


################ Offline Pretraining  ################
# random data follows the pattern of label being the addition of all stats
# stats are generated with np.random
def create_data(amount):
  features = np.empty(shape=(amount, 10))
  labels = np.empty(shape=(amount, 100))
  for i in range(amount):
    type = np.random.randint(0,2)
    if type == 0: # defense i.e. armor
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      defense = np.random.randint(0, 99)
      price = rarity + level + weight + defense
      features[i], labels[i], _, _ = parse_data(f'sell,{rarity},{level},{weight},{defense},0,0,0,{price}')
    if type == 1: # offense i.e weapon
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      damage = np.random.randint(0, 99)
      a_range = np.random.randint(0, 99)
      speed = np.random.randint(0, 99)
      price = rarity + level + weight + damage + speed + a_range
      features[i], labels[i], _, _ = parse_data(f'sell,{rarity},{level},{weight},0,{damage},{a_range},{speed},{price}')
    if type == 2: # misc
      rarity = np.random.randint(0, 3)
      level = np.random.randint(0, 99)
      weight = np.random.randint(0, 99)
      price = rarity + level + weight
      features[i], labels[i], _, _ = parse_data(f'sell,{rarity},{level},{weight},0,0,0,0,{price}')
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

# not needed for final product but used for pre-integration testing --------------------------------------------
    
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

ax = plt.subplot(111)
plt.plot(get_mean_reward(rewards[0]), label="Seller", linewidth=lwd,color=colors[0])
plt.plot(get_mean_reward(rewards[1]), label="Buyer", linewidth=lwd,color=colors[1])

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 1.25])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':20})


plt.tick_params(axis='both', which='major', labelsize=25)
plt.xticks([i*40 for i in range(8)], [i*2000 for i in range(8)])


plt.xlabel('Rounds (models were updated every 50 rounds)', size=30)
plt.ylabel('Cumulative Mean Reward', size=30)
plt.grid()
plt.show()