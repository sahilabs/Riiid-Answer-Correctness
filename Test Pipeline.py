import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import psutil
import random
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as Models
import pandas as pd
def seed_everything(seed): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(42)
from threading import Thread

#basic pipeline contains filling NaN values 
#the creating feature No_of_Question_attempted and correctly attemped using the column prior group responses
def basic_pipeline(test):
    test=test.reset_index()
    test['task_container_id'].fillna(0,inplace=True)
    test['timestamp'].fillna(0,inplace=True)
    test['content_id'].fillna(0,inplace=True)
    test['user_id'].fillna(0,inplace=True)
    test['prior_question_had_explanation'].replace({np.NaN:0,True:1,False:0},inplace=True)
    test['prior_question_elapsed_time'].fillna(-1,inplace=True)
    test['prior_question_elapsed_time']=test['prior_question_elapsed_time'].astype(int) 
    test.loc.__setitem__((test.timestamp==0, ('prior_question_elapsed_time')),-2)
    test.drop(columns=['content_type_id','prior_group_responses'],inplace=True)
    test['prior_group_answers_correct']=test.prior_group_answers_correct.fillna(-1).values.astype(str)
    ###########
    array=[eval(row) for row in test.prior_group_answers_correct.values]
    value=np.array([[[len(i),np.sum(i)] if(len(i)>0) else [0,0]][0] if(type(i)==list) else [[0,0]][0]  for i in array ])
    #value
    test['no_of_question_answered']=value[:,0]
    test['no_of_question_correct_answered']=value[:,1]
    test=test.merge(average_answer_question,on='content_id',how='left')
    test=test.merge(question,on='content_id',how='left')
    test.drop(columns=['group_num','row_id','content_id','prior_group_answers_correct'],inplace=True)
    return test
 

#this pipeline used to connect the train data which contains features of Known User_id and match it to test data
#add values to the columns No_of_question_attempted and correctly answered
#Moving average of Prior_question_had_explaination
  def feature_1_pipeline(test):
    global train_1
    global train_user_id
    test_user_id=test.user_id.unique()
    array_1=np.zeros(test.shape[0])
    array_2=array_1.copy()
    array_3=array_1.copy()
    for ID in test_user_id:
        temp=test[test.user_id==ID]
        index=np.array(temp.index)
        temp=temp.values
        cnt=temp.shape[0]
        temp=np.cumsum(temp[:,4])
        if(ID in train_user_id):
            data_point=train_1[train_1[:,0]==ID][-1]
            no_of_question_answered,no_of_question_correct_answered,moving_exp=data_point[1],data_point[2],data_point[3]
            train_1[train_1[:,0]==ID,1]=no_of_question_answered+cnt
            #train_1[train_1[:,0]==ID,-1]=moving_exp+temp[-1]
        else:
            no_of_question_answered=0
            no_of_question_correct_answered=0
            moving_exp=0
            #train_1=np.insert(train_1,0,[ID,cnt-1,0,temp[-1]],axis=0)
            
        no_of_question_answered=np.arange(0,cnt)+no_of_question_answered
        moving_prior_exp=temp+moving_exp
        #update
        array_1[index]=no_of_question_answered
        array_2[index]=no_of_question_correct_answered
        array_3[index]=moving_prior_exp
    test['no_of_question_answered']=test['no_of_question_answered'].values+array_1.astype(int)
    test['no_of_question_correct_answered']=array_2+test['no_of_question_correct_answered'].values
    test['moving_explaination']=array_3
    
    return test
  
  #In this pipeline Moving Average Index for the prior_question_elasped_time
  # Timestamp_diff is generated with dividing the no question in bundle
  #Updating the recent values to train_data and if the there is new user_id then train_data add with new user_id
  def feature_2_pipeline(test):
    global data,train,train_user_id
    user_id=test.user_id.values
    test_user_id=np.unique(user_id)
    count=test.groupby(['user_id','task_container_id'],sort=False)['Avg_prior_question_elapsed_time'].count().values
    temp_=test.groupby(['user_id','task_container_id'],sort=False).mean().reset_index()
    temp_['count']=count
    temp_=temp_[['user_id','timestamp','prior_question_elapsed_time','Avg_prior_question_elapsed_time','count']].values
    ########################
    array_1=np.zeros(test.shape[0])
    array_2=array_1.copy()
    for ID in test_user_id:
        temp=temp_[temp_[:,0]==ID].copy()
        cnt=temp.shape[0]
        Moving_average_index=[]
        Timestamp_diff=[]  
        #average of train
        if(ID in train_user_id):
            mvr,timestamp,avg=train[train[:,0]==ID,[3,2,1]].copy()
            distance=data[data[:,0]==ID,1].copy()
            timestamp_diff=(temp[0,1]-timestamp)/temp[0,-1]
            if(temp[0,2]<0):
                temp[0,2]=timestamp_diff
            moving_average_index=((temp[0,2]/avg)*np.arange(1,temp[0,-1]+1)+mvr*distance)/(distance+np.arange(1,temp[0,-1]+1))
            distance+=int(temp[0,-1])
            Moving_average_index.append(moving_average_index)
            Timestamp_diff.append([timestamp_diff]*int(temp[0,-1]))
            moving_average_index_=np.mean(moving_average_index)
        else:
            distance=0
            Moving_average_index.append([-2]*int(temp[0,-1]))
            Timestamp_diff.append([0]*int(temp[0,-1]))
            moving_average_index_=-2
            train=np.insert(train,0,[ID,temp[0,3],0,-2],axis=0)
            data=np.insert(data,0,[ID,0],axis=0)
        for i  in range(1,cnt):
            if(temp[i,2]<0):
                temp[i,2]=timestamp_diff
            moving_average_index=((temp[i,2]/temp[i-1,3])*np.arange(1,temp[i,-1]+1)+moving_average_index_*distance)/(distance+np.arange(1,temp[i,-1]+1))
            timestamp_diff=(temp[i,1]-temp[i-1,1])/temp[i,-1]
            Moving_average_index.append(moving_average_index)
            Timestamp_diff.append([timestamp_diff]*int(temp[i,-1]))
            moving_average_index_=np.mean(moving_average_index)
            distance+=int(temp[i,-1])
        ########
        train[train[:,0]==ID,3]=moving_average_index_
        train[train[:,0]==ID,2]=temp[-1,1]
        data[data[:,0]==ID,1]=distance
        ######
        array_1[user_id==ID]=np.hstack(Moving_average_index)
        array_2[user_id==ID]=np.hstack(Timestamp_diff)
    test['timestamp_diff']=array_2.astype(int)
    test['moving_average_index']=array_1
    return test
  ######
  train=train.groupby(['user_id']).mean().reset_index()
columns=train.columns#train.values
Dict={col:i for i,col in enumerate(columns)}
train=train.values
print(Dict)

#Model
import os
Dir='../input/neural-network'
file=[]
for f in os.listdir(Dir) :
    if('.h5' in str(f)):
        file.append(Dir+'/'+f)
        
import os
Dir='../input/riiid-model-2-6-1'
file_1=[]
for f in os.listdir(Dir) :
    if('.h5' in str(f)):
        file_1.append(Dir+'/'+f)

import tensorflow_addons as tfa
def model():
    Input=layers.Input((8,))
    x=layers.BatchNormalization()(Input)
    x=tfa.layers.WeightNormalization(layers.Dense(128,activation='relu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.2)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(64,activation='relu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.1)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(32,activation='relu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.1)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(16,activation='relu'))(x)
    x=layers.BatchNormalization()(x)
    #x=layers.Dropout(0.1)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(1,activation='sigmoid'))(x)
    model_neural=Models.Model(Input,x)
    ###################
    bce = tf.keras.losses.BinaryCrossentropy()
    optimizer=tf.keras.optimizers.Adam(lr=0.1)# if optimizer=='Adam' else tf.keras.optimizers.SGD()
    model_neural.compile(loss=bce,optimizer=optimizer,metrics=[tf.keras.metrics.AUC()])
    return model_neural

  
  features=['Day_passed', 'timestamp_diff', 'Avg_prior_question_elapsed_time','Percent_of_student_Answered_Correctly','no_of_question_correct_answered',
            'moving_explaination', 'no_of_question_answered', 'moving_average_index']
#Creating Model sepratly for every folds
#that save time
model_neural_1=model()
model_neural_2=model()
model_neural_3=model()
model_neural_4=model()
model_neural_5=model()
model_neural_6=model()

model_neural_1.load_weights(file[0])
model_neural_2.load_weights(file[1])
model_neural_3.load_weights(file[2])
model_neural_4.load_weights(file_1[0])
model_neural_5.load_weights(file_1[1])
model_neural_6.load_weights(file_1[2])

def model_prediction(test_df):
    global mean_std,features,file,model_neural_1,model_neural_2,model_neural_3,model_neural_4
    prediction=[]
    temp_test=test_df[features].copy().values
    t=temp_test[:,6].copy()
    t[t==0]=1
    temp_test[:,4]=temp_test[:,4]/t
    temp_test[:,5]=temp_test[:,5]/t
    #print(mean_std)
    for i,(_,mean,std) in enumerate(mean_std):
        temp_test[:,i]=(temp_test[:,i]-float(mean))/float(std)
    prediction=[]
    pred1=model_neural_1(temp_test)
    pred2=model_neural_2(temp_test)
    pred3=model_neural_3(temp_test)
    pred4=model_neural_4(temp_test)
    pred5=model_neural_5(temp_test)
    pred6=model_neural_6(temp_test)
    prediction=np.array(pred1+pred2+pred3+pred4+pred5+pred6)/6
    prediction=prediction.reshape(-1,)
    prediction=np.round(prediction,2)
    return prediction
 


#taking test_data from Environemnt
import riiideducation
import time
env = riiideducation.make_env()
iter_test = env.iter_test()
start_=time.process_time()
for test_df, sample_prediction_df in tqdm(iter_test):
    if(len(sample_prediction_df)<=1): 
        env.predict(sample_prediction_df)
        continue
    test_df=test_df.loc[test_df['content_type_id'] == 0]
    #####
    start=time.process_time()
    test_df=basic_pipeline(test_df.copy())
    ######################
    #fist pipeline
    test_df=feature_1_pipeline(test_df.copy())
    test_df=feature_2_pipeline(test_df.copy())
    test_df['Day_passed']=(test_df['timestamp'].values/(24*60*60*1000)).astype(int)
    test_df.loc.__setitem__((test_df['timestamp_diff'].values>=10*60*1000, ('timestamp_diff')),10*60*1000)
    #################################################################
    prediction=model_prediction(test_df)
    sample_prediction_df['answered_correctly']=prediction
    #update for first pipeline
    test_df['no_of_question_correct_answered']=test_df['no_of_question_correct_answered'].values+prediction
    temp=test_df.groupby('user_id',sort=False)[['no_of_question_answered','no_of_question_correct_answered','moving_explaination']].max().reset_index()
    train_1=pd.concat([pd.DataFrame(train_1,columns=['user_id', 'no_of_question_answered','no_of_question_correct_answered','moving_explaination']),temp]).groupby('user_id',sort=False).max().reset_index().values
    env.predict(sample_prediction_df)
    train_user_id=train_1[:,0].copy()
    del temp,sample_prediction_df,test_df
    #AUC obtain 0.746
