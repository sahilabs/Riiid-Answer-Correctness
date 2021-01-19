
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

def fun():
    global train
    temp=train[train.timestamp==0]['user_id'].copy()
    temp=temp.reset_index()
    temp=temp.groupby('user_id').min()
    temp=temp.reset_index()
    return temp
first_index_user_id=fun()
first_index_user_id.index=first_index_user_id['index'].values
del fun
gc.collect()

def func():
    global first_index_user_id
    global train
    no_of_question_answered=[]
    no_of_question_correct_answered=[]
    ind=first_index_user_id.index
    correct_answer=train.answered_correctly.values
    for i,value in enumerate(correct_answer):
        if(value!=-1):
            if(i in ind):
                count_1=0
                count_2=0 #reset
            no_of_question_answered.append(count_1)
            count_1+=1
            if(value==1):
                no_of_question_correct_answered.append(count_2)
                count_2+=1
            else:
                no_of_question_correct_answered.append(count_2)
        else:
            no_of_question_answered.append(-1)
            no_of_question_correct_answered.append(-1)
        
    gc.collect()
    return no_of_question_answered,no_of_question_correct_answered

no_of_question_answered,no_of_question_correct_answered=func()

def fun():
    global train
    temp=train.groupby(['user_id','task_container_id'],sort=False)['content_type_id'].count()
    temp=temp.reset_index()
    return temp.content_type_id.values
count_task_cnt_id=fun()
del fun
gc.collect()

def fun():
    global count_task_cnt_id
    array=np.zeros(train.shape[0])
    ind=0
    for i in tqdm(count_task_cnt_id):
        if(i==1):
            array[ind]=i
            ind=ind+i
        else:
            temp=[i]*i
            array[ind:ind+i]=temp
            ind=ind+i
        if(i==len(count_task_cnt_id)//2 | i==len(count_task_cnt_id)-1):
            gc.collect()
            print(gc.collect())
    gc.collect()
    #array=np.hstack(array)
    return array

timestamp_diff=np.load('../input/riid-prepocessing/timestamp.npy',allow_pickle=False)
array=fun()
del fun
gc.collect()
timestamp_diff=timestamp_diff/array
timestamp_diff[timestamp_diff>10*60*1000]=10*60*1000
############
question.question_id=question.question_id.astype(int)
question.bundle_id=question.bundle_id.astype(int)
question.correct_answer=question.correct_answer.astype(int)
question.part=question.part.astype(int)

temp=train[train.content_type_id==0].groupby('content_id')['answered_correctly']
temp=temp.sum()/temp.count()*100
temp=temp.reset_index()
temp.columns=['question_id','Percent_of_student_Answered_Correctly']
question=pd.merge(question,temp,on='question_id',how='left')
del temp

#######
#Average time to Answer the Question Correctly
def func():
    average_answer_question=train[(train.answered_correctly==1)&(train.prior_question_elapsed_time>-1)].groupby('content_id')['prior_question_elapsed_time'].mean()
    gc.collect()
    average_answer_question=average_answer_question.reset_index()
    average_answer_question.columns=['content_id', 'Avg_prior_question_elapsed_time']
    return average_answer_question
average_answer_question=func()

mean=average_answer_question.Avg_prior_question_elapsed_time.mean()
df2=pd.DataFrame({average_answer_question.columns[0]:[1484,1485,1486,10007],average_answer_question.columns[1]:[mean]*4})
average_answer_question=average_answer_question.append(df2)
average_answer_question=average_answer_question.sort_values('content_id')
#Merging the Average time to Answer the Question to Train Data
def func():
    global average_answer_question
    global train
    temp=train[train.content_type_id==0]['content_id'].copy()
    temp=pd.DataFrame(temp)
    gc.collect()
    Avg_prior_question_elapsed_time=pd.merge(temp,average_answer_question,how='left',on='content_id').Avg_prior_question_elapsed_time
    gc.collect()
    array=np.array([-1]*train.shape[0])
    array[train[train.content_type_id==0].index]=Avg_prior_question_elapsed_time.values
    train['Avg_prior_question_elapsed_time']=array
    del array,Avg_prior_question_elapsed_time
func()

############################
#Average_Question_Index: Dividing the Prior_Elasped_time to Average time to Answer the Question
# This Gives How far the user is away from average

def func():
    t1=train.prior_question_elapsed_time.values
    t2=train.Avg_prior_question_elapsed_time.values
    gc.collect()
    t=t1/t2
    t[t1<0]=t1[t1<0]
    return t
prior_question_Avg_prior_index=func()#Dividing the Prior Elasped with Average time to Answer without shifting back
train['prior_question_Avg_prior_index']=prior_question_Avg_prior_index #creating Features 
train=train[['user_id','task_container_id','prior_question_Avg_prior_index']]
#create seprate data for the calculation 
data=train[['user_id','task_container_id','prior_question_Avg_prior_index']].groupby(['user_id','task_container_id'],sort=False)['prior_question_Avg_prior_index'].mean()
data=data.reset_index()

prior_question_Avg_prior_index=data[data.prior_question_Avg_prior_index!=-1].prior_question_Avg_prior_index.values
prior_question_Avg_prior_index=np.append([-2.0],prior_question_Avg_prior_index[0:-1])#putting -2 value for every first bundle of the User
index=data.loc[data.prior_question_Avg_prior_index!=-1].index
array=np.zeros(data.shape[0])
array[index]=prior_question_Avg_prior_index
del index
index=data.loc[data.prior_question_Avg_prior_index==-1].index
array[index]=-1
del index,prior_question_Avg_prior_index
data['prior_question_Avg_prior_index']=array
del array
data['prior_question_Avg_prior_index']=data.prior_question_Avg_prior_index
data=reduce_mem_usage(data, verbose=True)

#Now the in data the Prior_question_avg_Index is shift back to it's Place
#Next merge this data to training
train.drop(columns=['prior_question_Avg_prior_index'],inplace=True)
train['c']=1
gc.collect()
df=train.groupby(['user_id','task_container_id'],sort=False).count()
df=df.reset_index()
count=df.c.values
del df,train

prior_question_Avg_prior_index=data.prior_question_Avg_prior_index.values
del data
gc.collect()
import gc
def prior_count(prior_question_Avg_prior_index,count):
    prior_question_Avg_prior_index_final=[]
    #prior_question_elapsed_time_final=np.array(prior_question_elapsed_time_final)
    #cnt=0
    for i,j in tqdm(zip(prior_question_Avg_prior_index,count)):
        if(j>1):
            temp=[i]*j #for _ in range(j)]
        else:
            temp=[i]
        #prior_question_elapsed_time_final=np.append(prior_question_elapsed_time_final,temp)
        prior_question_Avg_prior_index_final.append(temp)
        del temp
    return np.hstack(prior_question_Avg_prior_index_final)
#Saving the Prior_Question_Avg_Index
steps=5
shape=prior_question_Avg_prior_index.shape[0]
persteps=shape//steps
for s in range(steps):
    if(s!=steps-1):
        #save all
        prior_question_Avg_prior_index_final=prior_count(prior_question_Avg_prior_index[s*persteps:(s+1)*persteps],count[s*persteps:(s+1)*persteps])
    else:
        prior_question_Avg_prior_index_final=prior_count(prior_question_Avg_prior_index[s*persteps:],count[s*persteps:])
    print('prior_question_Avg_prior_index'+' '+str(s+1)+' '+'saving.......')
    time.sleep(2)
    np.save('prior_question_Avg_prior_index'+' '+' ' + str(s+1),prior_question_Avg_prior_index_final)
    del prior_question_Avg_prior_index_final

################
