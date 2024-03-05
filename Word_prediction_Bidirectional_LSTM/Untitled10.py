#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.optimizers import Adam
import numpy as np


# In[3]:


tokenizer = Tokenizer()
data = open(r'E:\sahilproject\NLPTutorial\Word_prediction_Bidirectional_LSTM\irish-lyrics-eof.txt').read()
print(data)


# In[4]:


corpus = data.lower().split("\n")


# In[5]:


print(corpus)


# In[6]:


tokenizer.fit_on_texts(corpus)


# In[7]:


len(tokenizer.word_index)


# In[8]:


tokenizer.word_counts


# In[9]:


total_words = len(tokenizer.word_index)+1


# In[10]:


print(tokenizer.word_index)
print(total_words)


# In[11]:


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# In[12]:


type(input_sequences)
print(input_sequences)


# In[13]:


if isinstance(input_sequences, list):
    # Check if all elements of input_sequences are lists
    if all(isinstance(seq, list) for seq in input_sequences):
        print("input_sequences is a list of lists")
    else:
        print("input_sequences is a list, but not all elements are lists")
else:
    print("input_sequences is not a list")


# In[14]:


for seq in input_sequences:
    if not isinstance(seq, list):
        print("Found non-list element in input_sequences:", seq)
for seq in input_sequences:
    print("Sequence:", seq)
    for item in seq:
        print("  Item:", item)



# In[15]:


max_len_sequence= max([len(x) for x in input_sequences])


# In[16]:


print(max_len_sequence)


# In[17]:


# Assuming input_sequences is a list of lists
#X = [seq[:-1] for seq in input_sequences]
#labels = [seq[-1] for seq in input_sequences]
padded_sequences = pad_sequences(input_sequences, maxlen=max_len_sequence, padding='pre')

# Convert padded_sequences to a numpy array
input_sequences_array = np.array(padded_sequences)


# In[18]:


X = input_sequences_array[:,:-1]
labels = input_sequences_array[:,-1]


# In[19]:


type(X)
print(X)


# In[20]:


Y = tf.keras.utils.to_categorical(labels, num_classes=total_words)
type(Y)


# In[21]:


print(Y)


# In[22]:


model= Sequential()


# In[23]:


model.add(Embedding(total_words, 100, input_length=max_len_sequence-1))


# In[24]:


model.add(Bidirectional(LSTM(150)))


# In[25]:


model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.01)


# In[26]:


model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[27]:


tf.cm onfig.run_functions_eagerly(True)
history = model.fit(X, Y, epochs=10, verbose=1)


# In[39]:


print(model)


# In[40]:


import pandas as pd
model_loss=pd.DataFrame(history.history["loss"])
model_loss.plot()


# In[62]:


text = "Laurence went to Dublin"
token_list = tokenizer.texts_to_sequences([text])[0]
#we need to do pre padding to make each sequences same length by longest sentence in the corpus
token_list = pad_sequences([token_list], maxlen=max_len_sequence-1, padding="pre")

# and by passing our token list into the prediction function we can do prediction.
# This will give us the token of the word most likely to be the next one in the sequence. 
#predicted = model.predict_class(token_list)
predictions = model.predict(token_list)
print(predictions)

# Extracting the index of the word with the highest probability as the predicted class
predicted_index = np.argmax(predictions)


print(tokenizer.word_index["home"])
# Converting the index back to the actual word using the tokenizer
predicted_word = tokenizer.index_word[predicted_index]

print("Predicted next word:", predicted_word)


# In[84]:


input_text="Laurence went to dublin "
prediction_word_range=10
for _ in range(prediction_word_range):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    #we need to do pre padding to make each sequences same length by longest sentence in the corpus
    token_list = pad_sequences([token_list], maxlen=max_len_sequence-1, padding="pre")
    predictions = model.predict(token_list)
    predicted_word = np.argmax(predictions)
    output_word=""
    for word,index in tokenizer.word_index.items():
        if index == predicted_word:
            output_word= word
            break
    input_text += " " + output_word
print(input_text)
    


# In[ ]:




