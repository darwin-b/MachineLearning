
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

# del reader
mean_rating = np.nanmean(data_matrix,axis=1)
# data_matrix[np.isnan(data_matrix)]=0

deviation = data_matrix - mean_rating[:,np.newaxis]
deviation[np.isnan(deviation)]=0

# numerator_correlation = deviation.dot(deviation.T)
# d_correlation = (deviation**2).sum(axis=1)[:,np.newaxis]
#
# denominator_correlation = d_correlation.dot(d_correlation.T)

weights = {}
ratings={}
predicted = {}
# temp =(deviation[1]*deviation).sum(axis=1)
# temp2 =(deviation[1]**2).sum()*((deviation**2).sum(axis=1))

with open(test_ratings_path,'r') as reader:

    for line in reader:
        title,user_id,rating = line.split(',')
        mapped_user = map_users[user_id]
        mapped_title = map_titles[title]

        n_correlation = (deviation[mapped_user]*deviation).sum(axis=1)
        d_correlation = ((deviation[mapped_user]**2).sum())*((deviation**2).sum(axis=1))

        if user_id not in weights:
            weights[user_id]=n_correlation/d_correlation

        normalising_constant = 1/np.nansum(weights[user_id])
        weighted_sum = np.nansum(weights[user_id]*(data_matrix[:,mapped_title] - mean_rating))
        predicted[user_id] = mean_rating[mapped_user] + weighted_sum*normalising_constant

        # break


# error=[]
# with open(test_ratings_path,'r') as reader:
#
#     for line in reader:
#         title,user_id,rating = line.split(',')
#         if(float(rating)-predicted[user_id])
