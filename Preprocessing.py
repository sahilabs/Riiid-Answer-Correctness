
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

#Memory Reduction 
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns
        col_type = df[col].dtypes
        if col_type in numerics: #numerics
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
  
 #Data Loading
lecture=pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')
lecture=reduce_mem_usage(lecture, verbose=True)
question=pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
question=reduce_mem_usage(question, verbose=True)
train=pd.read_csv('../input/riiid-test-answer-prediction/train.csv')
train=reduce_mem_usage(train, verbose=True)
 
#memory deduction
print(train.memory_usage().sum() / 1024**2 )
#train['user_id']=train.user_id-train.user_id.min()
train['prior_question_had_explanation'].replace({np.NaN:-2,True:1,False:0},inplace=True)
train['prior_question_elapsed_time'].fillna(-1,inplace=True)
train.drop(columns='row_id',inplace=True)
train=reduce_mem_usage(train, verbose=True)
train.prior_question_elapsed_time=train.prior_question_elapsed_time.astype(int)

#Replacing the  first value of Every Student as -2 for the  columnsPrior_Elasped_Time
ind=np.nonzero(train.prior_question_had_explanation.values==-2)
train['prior_question_elapsed_time'].loc[ind]=-2
del ind

# There are Some Value of Prior Elasped time and Prior Question Had Explaination that had been taken from Future which impossible which meaning it's type of mistake.  
ind=train[(train.timestamp==0)&(train.prior_question_elapsed_time>=0)]['prior_question_elapsed_time'].index
train['prior_question_elapsed_time'].loc[ind]=-2
train['prior_question_had_explanation'].loc[ind]=-2





#------------------------------------------------------ Feature generation -------------------------------------------------------------------------------------------

#Timestamp Diff
#Timestamp_diff can Able to explain time taken by student to answer the question and to move to another task
timestamp=np.diff(timestamp)
for i,t in enumerate(timestamp):
    if((t==0)&(i>0)):
        timestamp[i]=timestamp[i-1]
print(timestamp.shape)

timestamp[timestamp<0]=0
timestamp=np.append([0],timestamp)

#Shifting Back the prior_question_had_explanation for Analysis Purpose
def func():
    global prior_question_had_explanation
    ind=train[train.content_type_id==0].index
    prior_question_had_explanation=prior_question_had_explanation[ind]
    prior_question_had_explanation=np.append(prior_question_had_explanation[1:],[-2])
    array=np.ones(train.shape[0])
    array[ind]=prior_question_had_explanation
    return array
prior_question_had_explanation=func()

#Shifting Back the prior_question_elapsed_time for Analysis Purpose
data=train[['user_id','task_container_id','prior_question_elapsed_time']].groupby(['user_id','task_container_id'],sort=False)['prior_question_elapsed_time'].max()
data=data.reset_index()
prior_question_elapsed_time=data[data.prior_question_elapsed_time!=-1].prior_question_elapsed_time.values
prior_question_elapsed_time=np.append(prior_question_elapsed_time[1:],[-2])
index=data.loc[data.prior_question_elapsed_time!=-1].index
array=np.zeros(data.shape[0])
array[index]=prior_question_elapsed_time
del index
index=data.loc[data.prior_question_elapsed_time==-1].index
array[index]=-1
del index,prior_question_elapsed_time
data['prior_question_elapsed_time']=array
del array
data['prior_question_elapsed_time']=data.prior_question_elapsed_time.astype(int)
train.drop(columns=['prior_question_elapsed_time'],inplace=True)
train['c']=1
df=train.groupby(['user_id','task_container_id'],sort=False).count()
df=df.reset_index()
count=df.c.values
del df,train
prior_question_elapsed_time=data.prior_question_elapsed_time.values
del data

#Dividing the Work in 5 steps as it taken lot memory 
steps=5
shape=prior_question_elapsed_time.shape[0]
persteps=shape//steps
for s in range(steps):
    if(s!=steps-1):
        #save all
        prior_question_elapsed_time_final=prior_count(prior_question_elapsed_time[s*persteps:(s+1)*persteps],count[s*persteps:(s+1)*persteps])
    else:
        prior_question_elapsed_time_final=prior_count(prior_question_elapsed_time[s*persteps:],count[s*persteps:])
    print('Prior_question_elapsed_time'+' '+str(s+1)+' '+'saving.......')
    time.sleep(2)
    np.save('Prior_question_elapsed_time'+' '+' ' + str(s+1),prior_question_elapsed_time_final)
    del prior_question_elapsed_time_final
    gc.collect()

