import pandas as pd
# make functions
with open('/mnt/DAE855F7E855D1FD/github_msc/NLP/HW3/Train.txt','r') as f:
  lines=f.read().splitlines()

word, tag = [], []
for i in range(len(lines)):
    try : 
        x, y = lines[i].split(" ")
        word.append(x)
        tag.append(y)
    except ValueError: # if we have empty line
        word.append(" ")
        tag.append("<S>")

# print(f'empty lines : {len(empty_lines)}')
data = {'word': word, 'tag': tag}
df = pd.DataFrame(data)
df.to_csv('/mnt/DAE855F7E855D1FD/github_msc/NLP/HW3/train_tag.csv', index=False)
print(df.head())