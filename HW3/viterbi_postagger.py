from utils import *
import time

start_time = time.time()

TRAIN_PATH = '/mnt/DAE855F7E855D1FD/github_msc/NLP/HW3/Train.txt'
TEST_PATH = '/mnt/DAE855F7E855D1FD/github_msc/NLP/HW3/Test.txt'
BASE_PATH = '/mnt/DAE855F7E855D1FD/github_msc/NLP/HW3/'

df = read_train_file(TRAIN_PATH, BASE_PATH + "train_tag_word.csv")
transition_matrix = create_transition_matrix(df, alpha=0.01)
vocab = extract_vocab(df)
observation_matrix = create_observation_matrix(df, alpha=0.01)
observ, gold_tag = read_test_file(TEST_PATH)

save_matrix(transition_matrix, BASE_PATH + "transition_matrix.npy")
save_list(vocab, BASE_PATH + "vocab")
save_matrix(observation_matrix, BASE_PATH + "observation_matrix.npy")

pred_tag , prob = viterbi(observ, observation_matrix, transition_matrix, vocab)
accuracy, pred_tag = calculate_accuracy(df, pred_tag, gold_tag)
save_list(pred_tag, BASE_PATH + "predicted_tag")

with open(BASE_PATH + "result.txt", "a") as f:
  f.write(f'\nPOSTagging probability : {prob} | accuracy : {accuracy} %\n')
  f.write(f'Wall time : {(time.time() - start_time)} seconds')
