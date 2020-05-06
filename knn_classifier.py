from skmultiflow.lazy import KNNClassifier
from skmultiflow.data import SEAGenerator
from skmultiflow.utils.utils import *
from sklearn import preprocessing   
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential

class MyKNNClassifier(KNNClassifier):
    
    def __init__(self, 
                 n_neighbors=5, 
                 max_window_size=1000, 
                 leaf_size=30, 
                 metric='euclidean', 
                 weighted_vote=False,
                 standardize=False):
        super().__init__(n_neighbors=n_neighbors, 
                         max_window_size=max_window_size, 
                         leaf_size=leaf_size, 
                         metric=metric)
        self.weighted_vote = weighted_vote
        self.standardize = standardize
        
    def predict_proba(self,X):
        if self.standardize:
            X = self.CalculateStandardization(X)
        r, c = get_dimensions(X)
        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # The model is empty, defaulting to zero
            return np.zeros(shape=(r, 1))
        proba = []

        self.classes = list(set().union(self.classes,
                                        np.unique(self.data_window.targets_buffer.astype(np.int))))

        new_dist, new_ind = self._get_neighbors(X)
        dist_list = new_dist.tolist()
        count = 0
    
        for i in range(r):
            votes = [0.0 for _ in range(int(max(self.classes) + 1))]
            for index in new_ind[i]:
                # Calculate votes by adding 1/distance
                if self.weighted_vote:
                    votes[int(self.data_window.targets_buffer[index])] += 1. / float(dist_list[i][count])
                else: 
                    votes[int(self.data_window.targets_buffer[index])] += 1. / len(new_ind[i])
                count = count + 1
            proba.append(votes)
        return np.asarray(proba)
    
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.standardize:
            X = self.CalculateStandardization(X)
        r, c = get_dimensions(X)
        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        for i in range(r):
            self.data_window.add_sample(X[i], y[i])
        return self
    
    def CalculateStandardization(self,X):
        X = X.astype(float)
        X_mean = X.mean(axis = 0)
        X_std = X.std()
        standardization = (X-X_mean) / X_std
        return standardization



r'''
stream = FileStream(r"C:\Users\luyj0\OneDrive\Desktop\COMPX523-Data Stream Mining\data_n30000.csv")
#knn = MyKNNClassifier(weighted_vote=True)
s_knn = MyKNNClassifier(standardize=True)
metrics = ['accuracy', 'kappa', 'kappa_m','kappa_t', 'running_time', 'model_size']
# use a test-then-train evaluation approach
evaluator = EvaluatePrequential(max_samples=30000,
                                n_wait=100,
                                show_plot=False,
                                metrics=metrics)
evaluator.evaluate(stream=stream,model=[s_knn],model_names=['KNN + standardize'])
#evaluator.evaluate(stream=stream,model=[knn],model_names=['KNN'])
# Setting up the stream
stream = SEAGenerator(random_state=1, noise_percentage=.1)
knn = MyKNNClassifier(weighted_vote=True)

# Keep track of sample count and correct prediction count
n_samples = 0
corrects = 0
while n_samples < 5000:
    X, y = stream.next_sample()
    if (knn.standardize):
        X = knn.CalculateStandardization(X)
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X,y)
    n_samples += 1
# Displaying results
print('{} samples analyzed.'.format(n_samples))
print("KNNClassifier's performance: {}".format(corrects/n_samples))
'''
