
import numpy as np


train_ratings_path = "./../Data/netflix/TrainingRatings.txt"
test_ratings_path = "./../Data/netflix/TestingRatings.txt"

map_users={}
map_titles={}
data_matrix = np.empty((28978,1821),dtype=np.float32)
data_matrix[:] = np.nan

with open(train_ratings_path,'r') as reader:
    counter_titles=0
    counter_users = 0

    for line in reader:
        title,user_id,rating = line.split(',')
        if not title in map_titles:
            map_titles[title] = counter_titles
            counter_titles +=1
        if not user_id in map_users:
            map_users[user_id]=counter_users
            counter_users +=1

        data_matrix[map_users[user_id]][map_titles[title]] = rating

del reader
mean_rating = np.nanmean(data_matrix,axis=1)
data_matrix[np.isnan(data_matrix)]=0

deviation = data_matrix - mean_rating[:,np.newaxis]


weights = {}
ratings={}
predicted = {}

squared_dev = (deviation**2).sum(axis=1)
act_ratings=[]
pred_ratings=[]
error_rating=[]

with open(test_ratings_path,'r') as reader:
    c=0
    for line in reader:
        title,user_id,rating = line.split(',')
        mapped_user = map_users[user_id]
        mapped_title = map_titles[title]


        if user_id not in weights:
            n_correlation = np.abs((deviation[mapped_user] * deviation).sum(axis=1))
            d_correlation = np.sqrt(squared_dev[mapped_user] * squared_dev)
            weights[user_id]=n_correlation/d_correlation

        normalising_constant = weights[user_id].sum()
        weighted_sum = (weights[user_id]*(data_matrix[:,mapped_title] - mean_rating)).sum()
        predicted[(mapped_title,user_id)] = mean_rating[mapped_user] + weighted_sum/normalising_constant
        act_ratings.append(float(rating.replace("\n", "")))
        error_rating.append(float(rating.replace("\n", ""))-predicted[(mapped_title,user_id)])

        print(c," Acct : ",float(rating.replace("\n", "")), "Pred : ",predicted[(mapped_title,user_id)])
        c+=1


temp = np.arange(6)
temp = temp/normalising_constant

temp = np.cov(deviation)