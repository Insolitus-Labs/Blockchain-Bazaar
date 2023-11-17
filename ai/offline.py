import re, numpy as np

# takes a string and outputs the proper form of input for the ai model
# as well as two booleans to decide which model to give and if this is
# a query or training data
def create_data_file(order):
  # order is a string in the form 
  # (for getting price) 'buy'
  #                     'sell'
  # (for giving data)   'bought'
  #                     'sold'
  #                              ,weight,defense,damage,range,speed,price
  # i.e. "buy,35,0,40,32,19,0" or "sold,35,0,40,32,19,145"
  query = (order.startswith("buy") or order.startswith("sell"))
  buy = (order.startswith("b"))
  stats = [int(x) for x in re.sub(r'[A-Za-z]+,', '', order).split(",")]
  label = stats[5]
  features = np.full((1, 500), 0)
  for i in range(0, 6):
    if stats[i] != 0:
      features[0, (i * 100 - 1) + stats[i]] = 1
  print(features[0])
  return features, label, query, buy

features, label, query, buy = create_data_file("buy,1,0,40,32,19,0")
print(features, label, query, buy)