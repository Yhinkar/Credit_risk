#Import the necessary modules
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# Read the data
df = pd.read_csv('CreditScoring.csv')
df.head(5)
df.columns = df.columns.str.lower()



status_values = {
    0:'unk', 
    1:'ok',
    2: 'default'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}
df.home = df.home.map(home_values)

marital_values = {
    0: 'unk',
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'seperated',
    5: 'divorced'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
}
df.records = df.records.map(records_values)

job_values = {
    0: 'unk',
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others'
}
df.job = df.job.map(job_values)
df.describe().round()


#turn inconsistent values to nan
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace = 99999999, value = np.nan)


df.status.value_counts()
df = df[df.status != 'unk'].reset_index(drop = True)
# # Exploratory Analysis
plt.figure(figsize = (10, 5))
sns.set_style('darkgrid')
sns.barplot(x = 'job', y= 'amount', data = df)
plt.figure(figsize = (10, 5))
sns.set_style('darkgrid')
sns.countplot(x = 'status', hue = 'marital', data = df)


# # Validation_set
df_full_train, df_test = train_test_split(df, test_size = 0.20, random_state = 101)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 101)

df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)

y_train = (df_train.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values

del df_train['status']
del df_test['status']
del df_val['status']


# # Encoding
train_dict = df_train.fillna(0).to_dict(orient = 'records')
val_dict = df_val.fillna(0).to_dict(orient = 'records')

dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

# Train the model
model = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 30)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[: , 1]
roc_auc_score(y_val, y_pred)

print(export_text(model, feature_names = dv.get_feature_names()))

# train with random_forest_classifier
rf = RandomForestClassifier(n_estimators = 50, max_depth = 10, min_samples_leaf = 3,
                            n_jobs = -1, random_state = 1)
rf.fit(X_train, y_train)



y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc


#train with gradient boosting
feature =dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature)
d_val= xgb.DMatrix(X_val, label = y_val, feature_names = feature)


xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round = 140)
y_pred = model.predict(d_val)
roc_auc_score(y_val, y_pred)
watchlist = [(dtrain, 'train'), (d_val, 'val')]

#Output

def parse_xgb_output(output):
    results = []
   
    
    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')
        
        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])
        
        results.append((it, train, val))
    columns = ['num_iter', 'train_auc', 'val_auc']   
    df_results = pd.DataFrame(results, columns = columns)
    return df_results

df_score = parse_xgb_output(output)

plt.plot(df_score.num_iter, df_score.train_auc, label = 'train' )
plt.plot(df_score.num_iter, df_score.val_auc, label = 'val')
plt.legend()

plt.plot(df_score.num_iter, df_score.val_auc, label = 'val')


# # Selecting the best model
dt = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 30)
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)
rf = RandomForestClassifier(n_estimators = 50, max_depth = 10, min_samples_leaf = 3,
                            n_jobs = -1, random_state = 1)
rf.fit(X_train, y_train)


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain,num_boost_round = 140)


y_pred = model.predict(d_val)
roc_auc_score(y_val, y_pred)

#training the final model with xgboost
df_full_train = df_full_train.reset_index(drop = True)
y_full_train = (df_full_train.status == 'default').astype('int').values
del df_full_train['status']



dicts_full_train = df_full_train.to_dict(orient = 'records')
dv= DictVectorizer(sparse = False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient = 'records')
X_test = dv.transform(dicts_test)



dfulltrain = xgb.DMatrix(X_full_train, label = y_full_train,
                   feature_names = dz.get_feature_names())

dtest = xgb.DMatrix(X_test, 
                   feature_names = dz.get_feature_names())

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain,num_boost_round = 140)
y_pred = model.predict(dtest)
roc_auc_score(y_test, y_pred)





