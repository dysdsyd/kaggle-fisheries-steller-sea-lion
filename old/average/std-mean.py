import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#train = pd.read_csv('train.csv')
#submission = pd.read_csv('sample_submission.csv')

train = pd.read_csv('train.csv')
submission = pd.read_csv('sample_submission.csv')

print(train.mean(axis=0))
print(train.std(axis=0))

mean_std = 0.94*train.mean(axis=0) - 0.12*train.std(axis=0)
print(mean_std)

mean_std['adult_males'] = mean_std['adult_males'].mean()+1
mean_std['subadult_males'] = mean_std['subadult_males'].mean()+1
mean_std['adult_females'] = mean_std['adult_females'].mean()+3
mean_std['juveniles'] = mean_std['juveniles'].mean()+3
mean_std['pups'] = mean_std['pups'].mean()+3

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean_std[c])
submission.to_csv('submission.csv', index=False)