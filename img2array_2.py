import numpy, os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from PIL import Image
import numpy as np
from sklearn.externals import joblib
X=[]
Y=[]

path="image/"

n_sample = 0 #Total number of Images
h = 100 #Height of image in float
w = 100 #Width of image in float 
n_features = 10000 #Length of feature vector
target_names = [] #Array to store the names of the persons
label_count = 0
n_classes = 0

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=Image.open(path+directory+"/"+file).convert('L') 
        featurevector=numpy.array(img).flatten()
        X.append(featurevector)
        Y.append(int(directory))
        n_sample = n_sample + 1
    target_names.append(directory)
    label_count=label_count+1

n_classes = len(target_names)

X = np.asarray(X)
Y = np.asarray(Y)
target_names = np.asarray(target_names)

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and teststing set

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

n_components = 10

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, len(X_test)))

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

#eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

###############################################################################
# Train a SVM classification models

print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
              
clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

acc = metrics.accuracy_score(y_test, y_pred)
print('########################')
print("Accuracy:%.3f" % acc)

filename = 'finalized_model.sav'
joblib.dump(clf, filename) 
