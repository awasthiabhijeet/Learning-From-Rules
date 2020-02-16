import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os,sys
import pickle
from tqdm import tqdm

ELMO_EMBED_SIZE=1024

def get_embeddings_list(message_embeddings, message_lengths, embedding_size):
  result = []
  for i in range(len(message_lengths)):
    message_vectors = []
    for j in range(message_lengths[i]):
      message_vectors.append(message_embeddings[i][j])
    result.append(message_vectors)
  return result

def sentences_to_contx_embs(messages,batch_size=64):
  message_lengths = [len(m.split()) for m in messages]
  module_url = "https://tfhub.dev/google/elmo/2"
  elmo = hub.Module(module_url,trainable=False)
  tf.logging.set_verbosity(tf.logging.ERROR)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = []
    for i in tqdm(range(0,len(messages),batch_size)):
      #print("Embedding sentences from {} to {}".format(i,min(i+batch_size,len(messages))-1))
      message_batch = messages[i:i+batch_size]
      length_batch = message_lengths[i:i+batch_size]
      embeddings_batch = session.run(elmo(message_batch,signature="default",as_dict=True))["elmo"]
      embeddings_batch = get_embeddings_list(embeddings_batch, length_batch, ELMO_EMBED_SIZE)
      message_embeddings.extend(embeddings_batch)
  return message_embeddings
