import cv2
import numpy as np
from sklearn import neighbors
    img = cv2.imread("digits.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    data_set = [np.vsplit(row,50)] for row in np.hsplit(img,100)
    [
     [
      [0 0 1 0 0]
      [0 1 1 0 0]
      [0 0 1 0 0]
      [0 0 1 0 0]
      [0 0 1 0 0]
      ] #Number 1
     [
      [1 1 1 1 1]
      [1 0 0 1 0]
      [0 0 1 0 0]
      [0 1 0 0 0]
      [1 1 1 1 1]
      ] #Number 2
     .
     .
     .
     ]
    data_set = np.array(data_set).reshape(-1,400)
    [
     [0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0] #Number 1
     [1 1 1 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1 1 1 1] #Number 2
     .
     .
     .
    ]
    X_train = data_set[:2500]
    X_test = data_set[2500:]
    Y_train = np.tile(np.repeat(np.arange(10),5),50)
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train,Y_train)
    predicted = classifier.predict(X_test[20])
    print(predicted) #[4]
    cv2.imshow(str(predicted),X_test[20].reshape(20,20))
    cv2.waitKey()

