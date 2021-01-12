# MachineLearning 

## 1. Decision Trees
	- Find Decision Trees Folder 
	
	Ignore the following steps if the following packages are already installed
	----------------------------  Installation--------------------------------
	- Ensure that the packages in Pipfile are installed 
	(Version numbers need not be matched just ensure that following packages are available for code to compile)
		-pip install numpy
		-pip install pandas
		-pip install sklearn
		-pip install scikit-learn

	-------------------------------Execution------------------------------------

	- Run any follwing python scripts from the terminal from respective folder (i.e Decision Trees)

	- python InfoGainDTree.py
		- Will read 15 sets of Datasets in Data folder and 
		start printing accuracies with "ENTROPY" as impurity heuristic for each set of data
		- Followed by accuracies of depth based pruning [depths : 20,15,10,5]
		
		- Assumptions : Root node in tree is at depth --> 0 "Zero" 
			      : Depth 5 pruning ==> that the max depth of tree is 5, 
				which is that maximum possible depth of any leaf node is 5 "five"

	- python VarianceImpurityDTree.py
		- Will read 15 sets of Datasets in Data folder and 
		start printing accuracies with "VARIANCE" as impurity heuristic for each set of data
		- Followed by accuracies of depth based pruning [depths : 20,15,10,5]
		
		- Assumptions : Root node in tree is at depth --> 0 "Zero" 
			      : Depth 5 pruning ==> that the max depth of tree is 5, 
			      which is that maximum possible depth of any leaf node is 5 "five"
	

	Note : Already executed results can be found in Results folder. [Run times could be huge for large datasets]

	- python RandomForests.py
		- will read 15 sets of Datasets in Data folder and start printing accuracies


	Note : Running time take hours to complete execution.
	[Please run it in background mode and result matrices shall be written to various output files in the same folder level]

	Refer sample InfoGain & variance Dtree results screenshots saved in the folder.

	Refer the following index in result ouput files

	1 - c300_d100
	2 - c300_d1000
	3 - c300_d5000
	4 - 
	.
	.
	.
	.
	13 - c1800_d1000
	14 - c1800_d5000


	testDataFiles  = ["test_c300_d100","test_c300_d1000","test_c300_d5000","test_c500_d100","test_c500_d1000","test_c500_d5000","test_c1000_d100","test_c1000_d1000","test_c1000_d5000","test_c1500_d100","test_c1500_d1000","test_c1500_d5000","test_c1800_d100","test_c1800_d1000","test_c1800_d5000"]
	validDataFiles = ["valid_c300_d100","valid_c300_d1000","valid_c300_d5000","valid_c500_d100","valid_c500_d1000","valid_c500_d5000","valid_c1000_d100","valid_c1000_d1000","valid_c1000_d5000","valid_c1500_d100","valid_c1500_d1000","valid_c1500_d5000","valid_c1800_d100","valid_c1800_d1000","valid_c1800_d5000"]
	trainDataFiles = ["train_c300_d100","train_c300_d1000","train_c300_d5000","train_c500_d100","train_c500_d1000","train_c500_d5000","train_c1000_d100","train_c1000_d1000","train_c1000_d5000","train_c1500_d100","train_c1500_d1000","train_c1500_d5000","train_c1800_d100","train_c1800_d1000","train_c1800_d5000"]

	Also please Feel free to edit above lines of code to run files those of interest in any source code.
			
