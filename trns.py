import pandas as pd
from textblob import TextBlob
import nltk
from textblob import Word
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df=pd.read_csv("/Users/aman/coding stuff/sem 6/tarp/dataset/Suicide_preprocessed1.csv")

df_text=df["text"].tolist()
df_class=df["class"].tolist()


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis",model=model, tokenizer=tokenizer)
l1=[]
for z in range(0,1000,10):
    l1.append(sentiment_pipeline(df_text[z:z+10]))


print(l1[0]["label"])

l3=[]
for i in range(0,1000):
    if l1[0]["label"]=="NEGATIVE":
        l3.append("suicide")
    else:
        l3.append("non-suicide")

l2=df_class[0:1000]
print(accuracy_score(l2,l3))

