import pickle
import json
import pandas as pd
from keras.models import load_model

model=load_model('saved_model_3')

lstm_from_pickle_3 = pickle.loads(model)

jdata = json.loads('/Users/aman/coding stuff/sem 6/tarp/temp_login/public/jackson.json')
df = pd.DataFrame(jdata)

f = open("Downloads/dynamic.txt", "r")
str=f.read()

pred=lstm_from_pickle_3.predict(str)

df.insert(0,"text",str)
df.insert(0,"class",pred)


df.to_json(r'/Users/aman/coding stuff/sem 6/tarp/temp_login/public/jackson.json')
