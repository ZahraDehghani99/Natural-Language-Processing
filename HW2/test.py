import sys
from hmm import *

model_list = sys.argv[1]
test_data = sys.argv[2]
result_addres = sys.argv[3]

models = readFile(model_list)
test_data_ = readFile(test_data)
# print(f'sf: {test_data_}')
best_result = []
for i in range(len(test_data_)):
    prob = [0, 0, 0, 0, 0]
    for j in range(len(models)):
        pi, A, B = read_init_file(models[j])
        observ = mapping_observation_single_seq(test_data_[i])
        prob[j] = likelihood(observ, A, B, pi)
    max_value = max(prob)
    max_model_index = prob.index(max_value)
    best_result.append(models[max_model_index]+ " " + str(max_value))

with open(result_addres, 'w') as f:
    f.write('\n'.join(best_result))
