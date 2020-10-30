import numpy as np

print("-----------------Reading Train Ratings -------------------")

train_ratings_path = "./../Data/netflix/TrainingRatings.txt"
test_ratings_path = "./../Data/netflix/TestingRatings.txt"

map_users={}
map_titles={}
data_matrix = np.empty((28978,1821))
data_matrix[:] = np.nan

'''
Reading Train Ratings
'''
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


print("-----------------Computing Weights-------------------")

'''
Calculating Mean of voted ratings for each user
'''
mean_rating = np.nanmean(data_matrix,axis=1)

deviation = data_matrix - mean_rating[:,np.newaxis]

'''
Replacing Nan values with Zeroes
'''
data_matrix[np.isnan(data_matrix)]=0
deviation[np.isnan(deviation)]=0

numerator_correlation = deviation.dot(deviation.T)
d_correlation = (deviation**2).sum(axis=1)[:,np.newaxis]
denominator_correlation = d_correlation.dot(d_correlation.T)

weights = numerator_correlation/np.sqrt(denominator_correlation)

print("-----------------Weights Computed-------------------")

'''
Replacing any Nan or invalid values due to divison of 0 with Zeros
'''
weights[np.isnan(weights)]=0
weights[np.isinf(weights)]=0


act_ratings=[]
pred_ratings=[]
error_rating=[]
predicted = {}
MAE =0
abs_error=0
squared_error=0
'''
Predicting Train Ratings & computing MSE & MAE
'''
with open(test_ratings_path,'r') as reader:
    c=0
    for line in reader:
        title,user_id,rating = line.split(',')
        mapped_user = map_users[user_id]
        mapped_title = map_titles[title]

        normalising_constant = weights[mapped_user].sum()
        predicted[(mapped_title,user_id)] = mean_rating[mapped_user] + (weights[mapped_user].dot(data_matrix[:,mapped_title] - mean_rating))/normalising_constant
        act_ratings.append(float(rating.replace("\n", "")))
        error =  (float(rating.replace("\n", "")))-predicted[(mapped_title,user_id)]
        error_rating.append(error)
        abs_error += abs(error)
        squared_error += error**2
        print(c," Acct : ",float(rating.replace("\n", "")), "Pred : ",predicted[(mapped_title,user_id)], "Error : ", error," len: ",len(act_ratings))
        c+=1
        # break


print("Mean absolute Error : ", abs_error / len(act_ratings))
print("Mean Squared Error : ",squared_error/len(act_ratings))

print("Errors (MAE, MSE) : ",abs_error,squared_error)

file_name="resultsMatrix_NetflixPredictions"+".txt"
with open(file_name,'w') as file:
    text = "MAE : "+str(abs_error/len(act_ratings))+"\n"+"MSE : "+str(squared_error/len(act_ratings))+"\n\n"
    text = text + "-----------------------------------------------------------------------------------------\n"
    text = text + " Predicted Ratings : \n" + str(predicted)

