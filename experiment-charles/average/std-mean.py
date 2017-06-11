

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#train = pd.read_csv('train.csv')
#submission = pd.read_csv('sample_submission.csv')
version = "v4"

train = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/counts/trainCount_'+ version +'.csv')
submission = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/submissions/sample_submission.csv')

print(train.mean(axis=0))
print(train.std(axis=0))

mean_std = 0.94*train.mean(axis=0) - 0.12*train.std(axis=0)
print(mean_std)

print(mean_std['adult_males'].mean())
print(mean_std['subadult_males'].mean())
print(mean_std['adult_females'].mean())
print(mean_std['juveniles'].mean())
print(mean_std['pups'].mean())

mean_std['adult_males'] = mean_std['adult_males'].mean()
mean_std['subadult_males'] = mean_std['subadult_males'].mean()
mean_std['adult_females'] = mean_std['adult_females'].mean()
mean_std['juveniles'] = mean_std['juveniles'].mean()
mean_std['pups'] = mean_std['pups'].mean()

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean_std[c])
submission.to_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/submissions/avg_coord_'+ version +'.csv', index=False)