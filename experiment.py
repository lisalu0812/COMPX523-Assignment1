from skmultiflow.data import FileStream
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
from knn_classifier import MyKNNClassifier


#------------------------------------------------Experiment 1---------------------------------------------------------------
r'''
# Read in stream
stream = FileStream(r"C:\Users\luyj0\OneDrive\Desktop\COMPX523-Data Stream Mining\data_n30000.csv")
# Set up different classifiers
knn = MyKNNClassifier()
ht = HoeffdingTreeClassifier()
nb = NaiveBayes()
wv_knn = MyKNNClassifier(weighted_vote=True)
s_knn = MyKNNClassifier(standardize=True)
metrics = ['accuracy', 'kappa', 'kappa_m','kappa_t', 'running_time', 'model_size']
# use a test-then-train evaluation approach
evaluator = EvaluatePrequential(max_samples=30000,
                                n_wait=100,
                                show_plot=False,
                                metrics=metrics)

model_list = [knn,ht,nb,wv_knn,s_knn]
name_list = ['KNN','HoeffdingTree','NaiveBayes','KNN+WeightedVote','KNN+Standardize']
# Execute each evaluation in the list until it reaches the end
for index in range(len(model_list)):
    evaluator.evaluate(stream=stream,model=[model_list[index]],model_names=[name_list[index]])
    cm = evaluator.get_mean_measurements(0).confusion_matrix
    print("Recall per class")
    # Recall = True Positive / (True Positive + False Negative)
    for i in range(cm.n_classes):
        recall = cm.data[(i,i)]/cm.sum_col[i] \
        if cm.sum_col[i] != 0 else 'Ill-defined'
        print("Class {}: {}".format(i, recall))
'''

        
#------------------------------------------------Experiment 2---------------------------------------------------------------
import sys
r'''
# usage: python experiment.py <MyKNNClassifier>
# e.g. python experiment.py "MyKNNClassifier(n_neighbors=3)" 
# Read in stream
stream = FileStream(r"C:\Users\luyj0\OneDrive\Desktop\COMPX523-Data Stream Mining\data_n30000.csv")
# Set up knn classifier from the user input by using eval()
knn = eval(sys.argv[1])
metrics = ['accuracy', 'kappa', 'kappa_m','kappa_t', 'running_time', 'model_size']
# use a test-then-train evaluation approach
evaluator = EvaluatePrequential(max_samples=30000,
                                n_wait=100,
                                show_plot=False,
                                metrics=metrics)
evaluator.evaluate(stream=stream,model=[knn],model_names=['KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix
print("Recall per class")
# Recall = True Positive / (True Positive + False Negative)
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))
'''
#------------------------------------------------Experiment 3--------------------------------------------------------------- 
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
# Read in stream
stream = FileStream(r"C:\Users\luyj0\OneDrive\Desktop\COMPX523-Data Stream Mining\covtype_numeric.csv")
# Set up different classifiers
knn = MyKNNClassifier()
ht = HoeffdingTreeClassifier()
nb = NaiveBayes()
wv_knn = MyKNNClassifier(weighted_vote=True)
s_knn = MyKNNClassifier(standardize=True)
arf = AdaptiveRandomForestClassifier()
lb = LeveragingBaggingClassifier()
# Set up two ensemble algorithms
metrics = ['accuracy', 'kappa', 'kappa_m','kappa_t', 'running_time', 'model_size']
# use a test-then-train evaluation approach
evaluator = EvaluatePrequential(max_samples=30000,
                                n_wait=100,
                                show_plot=False,
                                metrics=metrics)

model_list = [knn,ht,nb,wv_knn,s_knn,arf,lb]
name_list = ['KNN','HoeffdingTree','NaiveBayes','KNN+WeightedVote','KNN+Standardize','AdaptiveRandomForest','Leverage Bagging']
# Execute each evaluation in the list until it reaches the end
for index in range(len(model_list)):
    evaluator.evaluate(stream=stream,model=[model_list[index]],model_names=[name_list[index]])
    cm = evaluator.get_mean_measurements(0).confusion_matrix
    print("Recall per class")
    # Recall = True Positive / (True Positive + False Negative)
    for i in range(cm.n_classes):
        recall = cm.data[(i,i)]/cm.sum_col[i] \
        if cm.sum_col[i] != 0 else 'Ill-defined'
        print("Class {}: {}".format(i, recall))
    