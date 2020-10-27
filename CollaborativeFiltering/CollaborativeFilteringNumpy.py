import numpy as np


train_ratings_path = "./../Data/netflix/TrainingRatings.txt"
test_ratings_path = "./../Data/netflix/TestingRatings.txt"

map_users={}
map_titles={}
data_matrix = np.empty((28978,1821))
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


mean_rating = np.nanmean(data_matrix,axis=1)

deviation = data_matrix - mean_rating[:,np.newaxis]
data_matrix[np.isnan(data_matrix)]=0
deviation[np.isnan(deviation)]=0

numerator_correlation = deviation.dot(deviation.T)
d_correlation = (deviation**2).sum(axis=1)[:,np.newaxis]

denominator_correlation = d_correlation.dot(d_correlation.T)

weights = numerator_correlation/np.sqrt(denominator_correlation)

weights[np.isnan(weights)]=0
weights[np.isinf(weights)]=0


act_ratings=[]
pred_ratings=[]
error_rating=[]
predicted = {}
with open(test_ratings_path,'r') as reader:
    c=0
    for line in reader:
        title,user_id,rating = line.split(',')
        mapped_user = map_users[user_id]
        mapped_title = map_titles[title]

        normalising_constant = weights[user_id].sum()
        predicted[(mapped_title,user_id)] = mean_rating[mapped_user] + (weights[mapped_user].dot(data_matrix[:,mapped_title] - mean_rating))/normalising_constant
        act_ratings.append(float(rating.replace("\n", "")))
        error_rating.append(float(rating.replace("\n", ""))-predicted[(mapped_title,user_id)])

        print(c," Acct : ",float(rating.replace("\n", "")), "Pred : ",predicted[(mapped_title,user_id)])
        c+=1
        # break

a = np.array([np.nan,2,3])
b = np.array([np.nan,0,3])
c=a/b
c[np.isnan(c)]=0
c[np.isinf(c)]=0

z = data_matrix[:,1]