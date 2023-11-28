import re, numpy as np
from sklearn.linear_model import SGDClassifier
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import LinUCB

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
def decision(is_query, is_buy, message):
  if is_buy:
    if is_query:
      querySeller(message)
    else:
      dataSeller(message)
  else:
    if is_query:
      queryBuyer(message)
    else:
      dataBuyer(message)

nchoices = 100
base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

## Metaheuristic
seller = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.1,
                ucb_from_empty = False, random_state = 1111, assume_unique_reward = True)

buyer = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.1,
                ucb_from_empty = False, random_state = 2222, assume_unique_reward = True)

# These lists will keep track of the rewards obtained
seller_rewards = list()
buyer_rewards = list()

# batch size - algorithm will be refit after N rounds
batch_size = 1



# add offline pretraining here ---------------------------------------------------------------------------------------

# fitting model for the first time
seller.fit(X=first_batch, a=action_chosen, r=rewards_received)
seller_actions = action_chosen.copy()

buyer.fit(X=first_batch, a=action_chosen, r=rewards_received)
buyer_actions = action_chosen.copy()

# function to run through a single round given one piece of data and its label
# modelId is 0 for seller and 1 for buyer
def single_round(model, rewards, actions_hist, data, label, modelId):
    
    # choosing actions for this batch
    action = model.predict(data).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(label[modelId, action])
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, action)
    
    # rewards obtained now
    reward = label[modelId, action]
    
    # now refitting the algorithms after observing these new rewards
    model.partial_fit(data, action, reward)
    
    return new_actions_hist


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