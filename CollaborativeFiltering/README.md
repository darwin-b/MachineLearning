# Collaborative filtering


## Instructions to Execute 
> python CollaborativeFilteringNumpy.py <path-to-Train-data-file> <path-to-Test-data-file>  
>> Eg: python CollaborativeFilteringNumpy.py /home/Data/Netflix/TrainingRatings.txt /home/Data/Netflix/TestingRatings.txt 


## Note: 
> If a user has a voted same rating for all the titles in train set
	it will lead to zero value on computing [v(a,j) - v(mean)]**2 in denominator and 
     	hence error on divide. Replacing such inf or NaN values with 0 in weights matrix