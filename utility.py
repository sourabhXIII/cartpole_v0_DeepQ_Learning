import numpy as np



def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
      if v > max_val:
        max_val = v
        max_key = k
    return max_key, max_val