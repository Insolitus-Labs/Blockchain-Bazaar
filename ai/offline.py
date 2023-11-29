import re, numpy as np

# takes a string and outputs the proper form of input for the ai model
# as well as two booleans to decide which model to give and if this is
# a query or training data
def parse_data(order):
  # order is a string in the form 
  # (for getting price) 'buy'
  #                     'sell'
  # (for giving data)   'bought'
  #                     'sold'
  #                              ,rarity,level,weight,defense,damage,range,speed,price(closest),price(second)
  # i.e. "buy,35,0,40,32,19,0" or "sold,35,0,40,32,19,145"
  query = (order.startswith("buy") or order.startswith("sell"))
  buy = (order.startswith("b"))
  stats = [int(x) for x in re.sub(r'[A-Za-z]+,', '', order).split(",")]
  label = np.zeros(100)
  label[int(stats[7] / 10)] = 1
  features = np.zeros(10)
  features[stats[0] + 1] = 1
  for i in range(1, 7):
    features[i + 3] = stats[i] / 100
  return features, label, query, buy

# features, label, query, buy = parse_data("buy,1,0,40,32,19,0,43,25")
# print(features, label, query, buy)

# creates a large set of random data
# defaults price to be addition of all features
def create_data():
  features = np.empty(shape=(15000, 10))
  labels = np.empty(shape=(15000, 100))
  for i in range(15000):
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

X, y = create_data()

from sklearn.linear_model import SGDClassifier
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import LinUCB

nchoices = y.shape[1]
base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

## Metaheuristic
linucb = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.1,
                ucb_from_empty = False, random_state = 1111, assume_unique_reward = True)

# This list will keep track of the rewards obtained
rewards = list()

# batch size - algorithm will be refit after N rounds
batch_size = 50

# initial seed - all policy starts with a small random selection of actions/rewards
first_batch = X[:batch_size, :]
np.random.seed(1)
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

# fitting model for the first time
linucb.fit(X=first_batch, a=action_chosen, r=rewards_received)
    
# this list will keep track of which actions does each policy choose
actions = action_chosen.copy()

# rounds are simulated from the full dataset
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

# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size)) - 1):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    X_batch = X[batch_st:batch_end, :]
    y_batch = y[batch_st:batch_end, :]
    actions = simulate_rounds_stoch(linucb,
                                                   rewards,
                                                   actions,
                                                   X_batch, y_batch,
                                                   rnd_seed = batch_st)
    print(format(i/(X.shape[0]/batch_size), ".0%")) 
print("100%")
    
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
plt.plot(get_mean_reward(rewards), label="LinUCB (OLS)", linewidth=lwd,color=colors[0])

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