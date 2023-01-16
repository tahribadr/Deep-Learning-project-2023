import numpy as np
import gensim
import torch
import torch.nn.functional as F


#embed tokenized data using word2vec
def word2vec(data, vector_size=200, min_count=5):
    w2v_model = gensim.models.Word2Vec(data,
                                   vector_size=vector_size,
                                   window=5,
                                   min_count=min_count)
    words = set(w2v_model.wv.index_to_key )
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])for ls in data], dtype=object)
            
    w2v_weights = w2v_model.wv.vectors
    vocab_size, embedding_size = w2v_weights.shape
            
    return X_train_vect, vocab_size

#pad all embeddings to max len
def pad_to_max(embeddings, max_len):
    padded_embeddings = torch.empty(embeddings.shape[0], max_len, 200)
    for index, embedding in enumerate(embeddings):
        pad = (0, 0, 0, max_len-embedding.shape[0]) #pad the first dimension with zeros until it reaches max len
        padded_embeddings[index] = F.pad(torch.from_numpy(embedding), pad, "constant", 0)
    return padded_embeddings
  