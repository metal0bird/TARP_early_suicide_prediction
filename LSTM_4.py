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
import pickle

max_words=5000
max_len=38

df=pd.read_csv("/Users/aman/coding stuff/sem 6/tarp/dataset/Suicide_preprocessed4.csv")
df=df.dropna()


shortRows = df[df["text"].str.split().str.len()>40].index 
df =  df.drop(shortRows, axis = 0)


def cleaning(df):
    df['text'] = df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Lemmatization
    df['text'] = df['text'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    return df

#Generating Embeddings using tokenizer
df=df.iloc[0:10000,:]
data_cleaned=cleaning(df)
tokenizer = Tokenizer(num_words=700, split=' ') 
tokenizer.fit_on_texts(data_cleaned['text'].values)
X = tokenizer.texts_to_sequences(data_cleaned['text'].values)
X = pad_sequences(X)

#Model Building
model = Sequential()
model.add(Embedding(700, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(100, activation='LeakyReLU'))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
#print(model.summary())
df["class"]=preprocessing.LabelEncoder().fit_transform(df["class"])
X_train,X_test,y_train,y_test=train_test_split(X,df["class"],test_size=0.2)

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
#Model Training
print(X_train,X_test,y_train,y_test)
model.fit(X_train, y_train, epochs =5, batch_size=32, verbose =1)
#Model Testing
model.evaluate(X_test,y_test)

df_test=pd.read_csv("/Users/aman/coding stuff/sem 6/tarp/TARP_early_suicide_prediction/extracted_comments_preprocessed.csv")
df_test=df.dropna()

# Save the trained model as a pickle string.
saved_model_3 = pickle.dumps(model)


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_test['text'].values)
sequences = tokenizer.texts_to_sequences(df_test['text'].values)
comments = pad_sequences(sequences, maxlen=max_len)

# Load the pickled model
lstm_from_pickle_3 = pickle.loads(saved_model_3)
 
# Use the loaded pickled model to make predictions

print(lstm_from_pickle_3.predict(comments))

