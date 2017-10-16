from PIL import Image
import numpy as np

def img2arr(original):
    img = original.convert('L')
    ndarr = np.array(img.getdata())
    return ndarr

original = Image.open("face0.jpg")
arr = img2arr(original)
X = np.array[(arr)]
imgarr = np.array(X)


for num in range(0,11):
            file_name = 'face'+str(num)+'.jpg'
            original = Image.open(file_name)
            arr = img2arr(original)
            X = np.array[(arr)]
            imgarr = np.concatenate((imgarr,X))
            
    
   