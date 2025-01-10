import pickle
import pandas as pd
from recommender import recommend_system


pickle_out = open("recommend.pkl", "wb")
pickle.dump(recommend_system, pickle_out)
pickle_out.close()
