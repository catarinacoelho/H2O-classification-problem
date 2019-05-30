import h2o
import time
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

#start the h2o server
h2o.init()

#import files with the datasets
iris_csv = "irisdata.csv" #file that contains iris data set
sonar_csv = "sonar.csv" #file that contains sonar data set
votes_csv = "votes.csv" #file that contains votes data set


#import_file imports the datasets to be used
iris_data = h2o.import_file(iris_csv)
sonar_data = h2o.import_file(sonar_csv)
votes_data = h2o.import_file(votes_csv)


#split_frame splits a frame into distinct subsets of size determined by the given ratios
#in this case the subsets will be 75% of the original one
#we will have a test/train split of 0.75/0.25 respectivly
iris_train = iris_data.split_frame(ratios=[0.75])
sonar_train = sonar_data.split_frame(ratios=[0.75])
votes_train = votes_data.split_frame(ratios=[0.75])


#########################################################################################
######              Variables to use to build model and train the model             #####
#########################################################################################

#nfolds_values : specifies the number of folds for cross validation which will be no cross-validation (0), 5-fold and 10-fold
nfolds_values = [0, 5, 10]
#activation_values: represents the activation function we will use in the neural network, in this case RectifierWithDropout and Tahn
activation_values = ["RectifierWithDropout", "Tanh"]
#response_columns: specifies the vectors containing the names of the predictor variables to use when building the model
response_columns = [iris_train[0].names[0:4], sonar_train[0].names[0:60], votes_train[0].names[1:]]
#predictor_columns: specifies the columns to use as the dependent variable
predictor_columns = ["Column5", "Column61", "C1"]
#frames: contains the datasets used to build each model
frames = [iris_train[0], sonar_train[0], votes_train[0]]

predictvar = [iris_train[1], sonar_train[1], votes_train[1]]


for activation_function in activation_values:
    for nfold in nfolds_values:
    	#hidden = [5,20,100] corresponds to the parameter that defines the number of layers ans number of neurons for each layer
        dlmodel = H2ODeepLearningEstimator(nfolds= nfold, activation = activation_function, hidden = [5,20,100])
        for i in range(3):
            t1=time.time()
            dlmodel.train(x=response_columns[i], y=predictor_columns[i], training_frame = frames[i])
            t2=time.time()
            training_time = t2-t1
            #display perfomance metrics
            performancetest = dlmodel.model_performance(predictvar[i]) #score and compute new metrics on the test data
            print("\ndataset:" + str(i) + " activation:" + str(dlmodel.activation) + " nfolds:" + str(dlmodel.nfolds) )
            print("\nPerformance metrics: " + str(performancetest))
            print("\ntraining time: " + str(training_time))

            



