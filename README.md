# sentiment-analysis
Sentiment analysis-classification project for the Big Data course I'm taking in NKUA Department of Informatics and Telecommunications postgraduate program. Python v3.6.7 is used, and requires the following libraries: tensorflow 2.0(preferably with GPU support for faster training), keras, matplotlib, numpy, sklearn.

The following publicly available datasets are used: Sentiment 140: http://help.sentiment140.com/for-students

Amazon review data (Cell Phones and accessories category): http://jmcauley.ucsd.edu/data/amazon/ plus the Movies & TV data from the link.

Three types of neural networks are tested: Classical MLP, CNN and LSTM. There are 9 python scripts in total, 3 for each dataset. To run the scripts, you should have a data folder in the same directory as the scripts (check the code for exact filenames!). The data for each dataset can de downloaded from the links given above. An additional script that analyzes and generates wordclouds for each dataset is provided.

Sentiment140, the twitter corpus, is a binary classification problem as each sentiment can be either positive or negative. The data from the Amazon review repository are multiclass, with 5 classes in total that correspond to the respective rating (from 1 up to 5 stars). More information about each dataset is contained in the links. 

Some sample statistics are collected, which reside in their respective folders for each dataset. These contain the test accuracy in respect to the classifier's parameters. Accuracy could get even higher with careful tuning.
