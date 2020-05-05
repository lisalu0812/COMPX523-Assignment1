from skmultiflow.lazy import KNNClassifier
from skmultiflow.data import SEAGenerator


# Setting up the stream
stream = SEAGenerator(random_state=1, noise_percentage=.1)
knn = KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40)
# Keep track of sample count and correct prediction count
n_samples = 0
corrects = 0
while n_samples < 5000:
    X, y = stream.next_sample()
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X, y)
    n_samples += 1
# Displaying results
#print('KNNClassifier usage example')
print('{} samples analyzed.'.format(n_samples))
#5000 samples analyzed.
print("KNNClassifier's performance: {}".format(corrects/n_samples))
#KNN's performance: 0.8776