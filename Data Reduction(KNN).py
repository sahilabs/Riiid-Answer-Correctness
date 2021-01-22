#total train_size 9.8x10^7
#In which class 1 is of size 6.5x10^7
#In which class 0is of size 3.38x10^7

def Kmeans_Sample_Clustering(train,target,content_id)
  train_1=train[target==1]
  train_0=train[target==0]

  #3000000/64930989*100->4.62%
  #2000000/33872016*100->6%

  class_weight=[0.06,0.0462]

  #for every sample get the number of neighbour by multiplying class weight

  def clustering(temp,index):
      value=class_weight[index]
      if(len(temp)<1000):
          return np.mean(temp,axis=0).reshape(1,-1)
      n_clusters=int(temp.shape[0]*value)
      batch_size=n_clusters//10
      t2,t3=temp[:,[2,3]][0]
      temp=temp[:,[0,1,4,5,6,7]]
      kmeans=MiniBatchKMeans(n_clusters=n_clusters,batch_size=batch_size,random_state=0).partial_fit(temp)
      center=kmeans.cluster_centers_
      center=np.insert(center,2,t2,axis=1)
      center=np.insert(center,3,t3,axis=1)
      return center

  for Id in tqdm(content_id):
      temp=train_1[train_1[:,-1]==Id].copy()
      temp=temp[:,0:-1]
      center=clustering(temp,1)
      #print(center.shape)
      try:
          data_point=np.append(data_point,center,axis=0)
      except:
          data_point=center

  for Id in tqdm(content_id):
      temp=train_0[train_0[:,-1]==Id].copy()
      temp=temp[:,0:-1]
      center=clustering(temp,0)
      try:
          data_point=np.append(data_point,center,axis=0)
      except:
          data_point=center

  del data_point
  
  return data_1,data_0
