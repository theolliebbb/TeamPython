import pandas as pd
import string
import numpy as np
import json
import random
import keras_nlp
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np 
from keras.layers import TextVectorization
import os

tf.random.set_seed(2)
from numpy.random import seed
seed(1)

tf.config.optimizer.set_jit(True)


#load all the datasets 
df1 = pd.read_csv(r"Analysis\descriptions.csv", encoding='utf-8'
)
def category_extractor(data):
    i_d = [data['items'][i]['id'] for i in range(len(data['items']))]
    title = [data['items'][i]['snippet']["title"] for i in range(len(data['items']))]
    i_d = list(map(int, i_d))
    category = zip(i_d, title)
    category = dict(category)
    return category

#join the dataframes
df = pd.concat([df1], ignore_index=True)

#remove punctuations and convert text to lowercase
def clean_text(text):
    try:
        text = ''.join(str(e) for e in text if e not in string.punctuation).lower()
    except:
        x = "x"
    try:
        text = text.encode('utf8').decode('ascii', 'ignore')
    except:
        x="x"
    return text
columnlist = []
for e in df['description']:
    try:
        columnlist.append(str(e))
    except:
        continue
corpus = [clean_text(e) for e in columnlist]

tokenizer = Tokenizer(num_words=100)

def get_sequence_of_tokens(corpus):
  #get tokens
  tokenizer.fit_on_texts(corpus)

  total_words = len(tokenizer.word_index) + 1
 
  #convert to sequence of tokens
  input_sequences = []
  for line in corpus:
    token_list = tokenizer.texts_to_sequences([str(line)])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)
 
  return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)
def generate_padded_sequences(input_sequences):
    max_sequence_len=0
    max_sequence_len = max([len(x) for x in input_sequences])
    
    input_sequences = np.array(pad_sequences(input_sequences,  maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1], input_sequences[:, -1]
    
    label = ku.to_categorical(label, num_classes = total_words)
    return predictors, label, max_sequence_len
    

  


def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence

maxlen = 50
vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens
        
    # Helper method to choose a word from the top K probable words with respect to their probabilities
    # in a sequence
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt
        
        for i in range(self.max_tokens-1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            predictions = self.model.predict([tokenized_prompt], verbose=0)
            # To find the index of the next word in the prediction array.
            # The tokenized prompt is already shorter than the original decoded sample
            # by one, len(decoded_sample.split()) is two words ahead - so we remove 1 to get
            # the next word in the sequence
            sample_index = len(decoded_sample.strip().split())-1
            
            sampled_token = self.sample_token(predictions[0][sample_index])
            sampled_token = index_lookup[sampled_token]
            decoded_sample += " " + sampled_token
            
        print(f"\nSample text:\n{decoded_sample}...\n")
def clean_text(text):
    try:
        text = ''.join(str(e) for e in text if e not in string.punctuation).lower()
    except:
        x = "x"
    try:
        text = text.encode('utf8').decode('ascii', 'ignore')
    except:
        x="x"
    return text
columnlist = []
for e in df1['description']:
    try:
        columnlist.append(str(e))
    except:
        continue
corpus = [clean_text(e) for e in columnlist]
texts = ""
for e in columnlist:
    texts += e
texts = texts.lower()
text_list = texts.split('.')
len(text_list) 
len(texts.replace('\n', ' ').split(' '))
text_list = list(filter(None, text_list))
import random
random.shuffle(text_list)
length = len(text_list)
text_train = text_list[:int(0.7*length)]
text_test = text_list[int(0.7*length):int(0.85*length)]
text_valid = text_list[int(0.85*length):]

def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence
def preprocess_text(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y
maxlen = 50
vectorize_layer.adapt(text_list)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)
index_lookup = dict(zip(range(len(vocab)), vocab))    
index_lookup[5]

batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices(text_train)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(text_test)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size)

valid_dataset = tf.data.Dataset.from_tensor_slices(text_valid)
valid_dataset = valid_dataset.shuffle(buffer_size=256)
valid_dataset = valid_dataset.batch(batch_size)
train_dataset = train_dataset.map(preprocess_text)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = valid_dataset.map(preprocess_text)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
random_sentence = ' '.join(random.choice(text_valid).replace('\n', ' ').split(' ')[:4])
sampler = TextSampler(random_sentence, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')
embed_dim = 128
num_heads = 4
def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim, 
                                                            num_heads=num_heads, 
                                                            dropout=0.5)(embedding_layer)
    
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="adam", 
        loss='sparse_categorical_crossentropy',
        metrics=[keras_nlp.metrics.Perplexity(), 'accuracy'],
        run_eagerly=True
    )
    checkpoint_path = r"Analysis\machine\description\descweights"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.listdir(checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    print("Restored model")
    return model
    
model = create_model()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=r"Analysis\machine\description\descweights",
                                                 save_weights_only=True,
                                                 save_freq=1010,
                                                 verbose=1)
# model.fit(train_dataset, 
#                      validation_data=valid_dataset,
#                      epochs=5, 
#                      callbacks=[cp_callback])
def sample_token(logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
def generate_text(prompt, response_length=48):
    decoded_sample = prompt
    if decoded_sample == "":
        decoded_sample = random_sentence
    for i in range(response_length-1):
        tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
        predictions = model.predict([tokenized_prompt], verbose=0)
        sample_index = len(decoded_sample.strip().split())-1
        random.shuffle(predictions[0:1])
        sampled_token = sample_token(predictions[0][sample_index])
        sampled_token = index_lookup[sampled_token]
        available = r"abcdefghijklmnopqrstuvwxyz1234567890!\"#$%&'()-^\=~|@[;:]./\,<>?_+*}`{=~|"
        for part in sampled_token:
            if part not in available:
                sampled_token =""
                break
        if 'http' in sampled_token:
            sampled_token=""
        decoded_sample += " " + sampled_token
    return decoded_sample




def generate_textOllie(seed_text, next_words, model, max_sequence_len):
    wordlist = []
    wordlist.append(seed_text)
    for _ in range(next_words):
        
        token_list = tokenizer.fit_on_texts(wordlist[_])
        token_list = tokenizer.texts_to_sequences(wordlist[_])
        y = seed_text[0]
        token_list = pad_sequences(token_list, maxlen=max_sequence_len-1,  padding='post')
    
        predicted = (model.predict(token_list)>0.09).astype("int32")
        random.shuffle(predicted[0][0:7872])
        output_word = ""
        for word,index in tokenizer.word_index.items():
            for l in predicted:
                h = l[0:7872][index]
                if 1 == l[0:7872][index-1]:
                    output_word = word
                    seed_text += " "+output_word
                    predicted = ""
                    wordlist.append(output_word)
                    break
                    
                break
        continue  
            
    return seed_text.title()
    

    

