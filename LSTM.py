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
df=pd.read_csv("/Users/aman/coding stuff/sem 6/tarp/dataset/Suicide_preprocessed1.csv")

def cleaning(df):
    df['text'] = df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Lemmatization
    df['text'] = df['text'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    return df


#Generating Embeddings using tokenizer
df=df.iloc[0:100,:]
data_cleaned=cleaning(df)
tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(data_cleaned['text'].values)
X = tokenizer.texts_to_sequences(data_cleaned['text'].values)
X = pad_sequences(X)
#Model Building
model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(704, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='LeakyReLU'))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())
df["class"]=preprocessing.LabelEncoder().fit_transform(df["class"])
X_train,X_test,y_train,y_test=train_test_split(X,df["class"],test_size=0.2)

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
#Model Training
print(X_train,X_test,y_train,y_test)
model.fit(X_train, y_train, epochs =30, batch_size=32, verbose =1)
#Model Testing
model.evaluate(X_test,y_test)


  
# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)

# Load the pickled model
#lstm_from_pickle = pickle.loads(saved_model)
  
# Use the loaded pickled model to make predictions
#lstm_from_pickle.predict(X_test)