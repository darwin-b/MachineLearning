# K-Means Clustering

## Requirements
JDK version 15

## Run Instructions
> Javac .\KMeans.java 
>> Generates respective class file (KMeans.class) 
> Java KMeans <path-to-image-file> k <path-to-output-imagefile-with-jpg-extension>

>> Eg: 
>>> Java KMeans .\Penguins.jpg 10 .\out_penguins.jpg

## Note 
> Find pre-computed results in Results folder 
>> Each execution results start with randomized k initail points.
	This is also evident from multiple executions of K values for penguins image where if the initial point has 
	started within the region of yellow colour will segment earlier for as low as k=5 & on worst for until k=9 the yellow colour doesn't 
	show up
>> With the image of Koala hard limit for multiple iterations are set ragning from 10,20,50,100,100  are performed to optimize/strike balance 
	between runtime & image quality.
	 

