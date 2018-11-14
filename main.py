import csv
from collections import defaultdict
import numpy as np

to_train = set()

us_dic = defaultdict(set)
uk_dic = defaultdict(set)

labels = np.array([])
data = np.array([])

with open('Data\\forvo_raw\\forvo_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['country'] == 'United States':
            word = row['word']
            file_id = row['id']
            us_dic[word].add(file_id)
        elif row['country'] == 'United Kingdom':
            word = row['word']
            file_id = row['id']
            uk_dic[word].add(file_id)

    for word in us_dic.keys():
        if word in uk_dic.keys():
            us_ids = us_dic[word]
            uk_ids = uk_dic[word]
            
            for us_id in us_ids:
                for uk_id in uk_ids:
                    to_train.add((us_id, uk_id))

while len(to_train) > 0:
    if len(to_train) % 500 == 0:
        print(len(to_train))
        
    sample, target = to_train.pop()
    
    try:
        m, y, s = gen_mfcc('Data\\forvo_raw\\United States\\' + sample + '.mp3')
    except:
        print(sample, 'us')
        continue
    
    try:
        m2, y2, s2 = gen_mfcc('Data\\forvo_raw\\United Kingdom\\' + target + '.mp3')
    except:
        print(target, 'uk')
        continue
        
    m, m2 = align_sample_target(m, m2)
    
    if labels.size == 0:
        data = m.T
        labels = m2.T
    else:
        data = np.vstack((data, m.T))
        labels = np.vstack((labels, m2.T))


np.save('kaggle_data.npy', data)
np.save('kaggle_labels.npy', labels)