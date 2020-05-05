from skmultiflow.lazy import KNNClassifier
from skmultiflow.data import SEAGenerator
from skmultiflow.utils.utils import *


class MyKNNClassifier(KNNClassifier):
    def __init__(self, 
                 n_neighbors=5, 
                 max_window_size=1000, 
                 leaf_size=30, 
                 metric='euclidean', 
                 weighted_vote=False):
        super().__init__(n_neighbors=n_neighbors, 
                         max_window_size=max_window_size, 
                         leaf_size=leaf_size, 
                         metric=metric)
        self.weighted_vote = weighted_vote
        
    def predict_proba(self,X):
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
                votes[int(self.data_window.targets_buffer[index])] += 1. / float(dist_list[i][count])
                #votes[int(self.data_window.targets_buffer[index])] += 1. / len(new_ind[i])
                count = count + 1
            proba.append(votes)
        return np.asarray(proba)

# Setting up the stream
stream = SEAGenerator(random_state=1, noise_percentage=.1)
my_knn = MyKNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40, weighted_vote = True)
#knn = KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40)
# Keep track of sample count and correct prediction count
n_samples = 0
corrects = 0
while n_samples < 5000:
    X, y = stream.next_sample()
    my_pred = my_knn.predict(X)
    #my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    #knn = knn.partial_fit(X, y)
    my_knn = my_knn.partial_fit(X,y)
    n_samples += 1
# Displaying results
#print('KNNClassifier usage example')
print('{} samples analyzed.'.format(n_samples))
#5000 samples analyzed.
print("KNNClassifier's performance: {}".format(corrects/n_samples))
#KNN's performance: 0.8776