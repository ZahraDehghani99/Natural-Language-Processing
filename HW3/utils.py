import pandas as pd
import numpy as np
import operator
import pickle

def read_train_file(train_file_path, output_path):

    with open(train_file_path,'r') as f:
        lines = f.read().splitlines()
    word, tag = [], []
    for i in range(len(lines)):
        try : 
            x, y = lines[i].split(" ")
            word.append(x)
            tag.append(y)
        except ValueError: # if we have empty line
            word.append(" ")
            tag.append("<S>")

    data = {'word': word, 'tag': tag}
    df = pd.DataFrame(data)
    new_row = pd.DataFrame({'word':" ", 'tag': "<S>"}, index =[0]) #add start charachter at the first of the file
    df = pd.concat([new_row, df]).reset_index(drop = True)
    df.to_csv(output_path, index=False)
    return df


def read_test_file(train_file_path):

    with open(train_file_path,'r') as f:
        lines = f.read().splitlines()
    words, tags = [], []
    for i in range(len(lines)):
        try : 
            x, y = lines[i].split(" ")
            words.append(x)
            tags.append(y)
        except ValueError: # if we have empty line
            pass
    return words, tags 


def create_transition_matrix(df, alpha): # df : tag_word_dataframe
    # transiont_count dictionary
    # alpha : smoothing parameter
    tag_count = df.tag.value_counts().to_dict()
    all_tags = sorted(tag_count.keys())
    tag_seq = df.tag.to_list()
    transition_count = {} 
    for i in all_tags:
        for j in all_tags:
            key = (i, j)
            transition_count[key] = 0
    for i in range(len(tag_seq)-1):
        transition_count[(tag_seq[i], tag_seq[i+1])] += 1

    # create transition matrix : bigram
    transition_matrix = np.zeros((len(all_tags), len(all_tags)))

    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            count = 0
            key = (all_tags[i], all_tags[j])
            if key in transition_count:
                count = transition_count[key]
            count_prev_tag = tag_count[all_tags[i]]
            transition_matrix[i][j] = (count + alpha) / (count_prev_tag + alpha* len(all_tags))
    transition_matrix = np.delete(transition_matrix, 0, axis=1) # eliminate column 1 correspondes to <S> tag       
    return transition_matrix     


def extract_vocab(df):
    word_count = df.word.value_counts().to_dict()
    sorted_word_count = dict(sorted(word_count.items(), key=operator.itemgetter(1),reverse=True))
    del sorted_word_count[" "] # delete blanck lines
    
    single_occurrence_words = [] # threshold
    for word in sorted_word_count:
        if sorted_word_count[word] == 1:
            single_occurrence_words.append(word)
    new_word_counts = {k:v for k,v in sorted_word_count.items() if v != 1}
    new_word_counts["UNK"] = len(single_occurrence_words)
    new_word_counts = dict(sorted(new_word_counts.items(), key=operator.itemgetter(1),reverse=True)) # keys are our vocabulary
    vocabs = sorted(new_word_counts.keys())
    return vocabs 


def create_observation_matrix(df, alpha):
    # alpha : smoothing parameter
    vocabs = extract_vocab(df)
    tag_count = df.tag.value_counts().to_dict()
    all_tags = sorted(tag_count.keys())
    word_given_tag_count = {}
    for i in vocabs: # change name to vocab
        for j in all_tags:
                key = (i, j)
                word_given_tag_count[key] = 0

    for i in range(df.shape[0]):
        word, tag = df.word[i], df.tag[i]
        key = (word, tag)
        if word == " ": # for blank llines
            pass
        elif word not in vocabs:
            key = ("UNK", tag)
            word_given_tag_count[key] += 1
        else:
            word_given_tag_count[key] += 1  

    tag_count = df.tag.value_counts().to_dict()
    all_tags = sorted(tag_count.keys())
    observation_matrix = np.zeros((len(vocabs), len(all_tags))) 

    for i in range(observation_matrix.shape[0]):
        for j in range(observation_matrix.shape[1]):
            count = 0
            key = (vocabs[i], all_tags[j])
            if key in word_given_tag_count:
                count = word_given_tag_count[key]
            count_tag = tag_count[all_tags[j]]
            observation_matrix[i][j] = (count + alpha) / (count_tag + alpha*len(vocabs))   
    observation_matrix = np.delete(observation_matrix, 0, axis=1) 
    observation_matrix = observation_matrix.T
    return observation_matrix


def initial_probabilities(transition_matrix):
    return transition_matrix[0, :]


def vocab_index_observ(o, t, vocab):
    if o[t] in vocab:
        return vocab.index(o[t])
    else:
        unk_index = vocab.index("UNK")
        return unk_index  


def viterbi(observ, observation_matrix, transition_matrix, vocab):
    pi = initial_probabilities(transition_matrix)
    transition_matrix = transition_matrix[1:, :]
    N, T = observation_matrix.shape[0], len(observ)
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    viterbi[:, 0] = pi* observation_matrix[:, vocab_index_observ(observ, 0, vocab)]
    backpointer[:, 0] = 0
    for t in range(1, T):
        for s in range(N):
            value = viterbi[:, t-1] * transition_matrix[:, s] * observation_matrix[s, vocab_index_observ(observ, t, vocab)]
            viterbi[s, t] = np.max(value)
            backpointer[s, t] = np.argmax(value)
    bestpath = np.zeros(T)
    prob = np.max(viterbi[:, T-1])            
    last_state = np.argmax(viterbi[:, T-1])  
    bestpath[0] = last_state
    backtrack_index = 1
    for t in range(T-2, -1, -1):
        bestpath[backtrack_index] = backpointer[int(last_state), t]
        last_state = backpointer[int(last_state), t] 
        backtrack_index += 1

    # Flip the path array since we were backtracking    
    bestpath = np.flip(bestpath, axis=0)    
    return bestpath, prob                       


def viterbi_log(observ, observation_matrix, transition_matrix, vocab):
    pi = initial_probabilities(transition_matrix)
    transition_matrix = transition_matrix[1:, :]
    N, T = observation_matrix.shape[0], len(observ)
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    viterbi[:, 0] = np.log(pi* observation_matrix[:, vocab_index_observ(observ, 0, vocab)])
    backpointer[:, 0] = 0
    for t in range(1, T):
        for s in range(N):
            value = viterbi[:, t-1] + np.log(transition_matrix[:, s]) + np.log(observation_matrix[s, vocab_index_observ(observ, t, vocab)])
            viterbi[s, t] = np.max(value)
            backpointer[s, t] = np.argmax(value)
    bestpath = np.zeros(T)
    prob = np.max(viterbi[:, T-1])            
    last_state = np.argmax(viterbi[:, T-1])  
    bestpath[0] = last_state
    backtrack_index = 1
    for t in range(T-2, -1, -1):
        bestpath[backtrack_index] = backpointer[int(last_state), t]
        last_state = backpointer[int(last_state), t] 
        backtrack_index += 1

    # Flip the path array since we were backtracking    
    bestpath = np.flip(bestpath, axis=0)    
    return bestpath, prob    


def calculate_accuracy(df, pred_path, gold_tag):

    tag_count = df.tag.value_counts().to_dict()
    all_tags = sorted(tag_count.keys())
    all_tags.remove("<S>")
    pred_path = pred_path.astype(np.int16)
    pred_tag = [all_tags[i] for i in pred_path] 

    true = 0
    for i in range(len(gold_tag)):
        if pred_tag[i] == gold_tag[i]:
            true += 1
    acc = (true/ len(gold_tag)) * 100
    return acc, pred_tag


def save_list(vocab, output_path):
    with open(output_path, "wb") as f: 
        pickle.dump(vocab, f)


def load_list(file_path):
    with open(file_path, "rb") as f:  
        vocab = pickle.load(f)
    return vocab

def save_matrix(matrix_name, file_path):
    np.save(file_path, matrix_name)