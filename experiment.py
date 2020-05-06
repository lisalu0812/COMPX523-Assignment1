from skmultiflow.data import FileStream
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
from knn_classifier import MyKNNClassifier

stream = FileStream(r"C:\Users\luyj0\OneDrive\Desktop\COMPX523-Data Stream Mining\data_n30000.csv")
knn = KNNClassifier()
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
for index in range(len(model_list)):
    evaluator.evaluate(stream=stream,model=[model_list[index]],model_names=[name_list[index]])
    cm = evaluator.get_mean_measurements(0).confusion_matrix
    print("Recall per class")
    # Recall = True Positive / (True Positive + False Negative)
    for i in range(cm.n_classes):
        recall = cm.data[(i,i)]/cm.sum_col[i] \
        if cm.sum_col[i] != 0 else 'Ill-defined'
        print("Class {}: {}".format(i, recall))