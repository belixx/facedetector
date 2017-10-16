from PIL import Image
import numpy as np

def img2array(original):
	img=original.convert('L')  
	ndarr = np.array(img.getdata())
	return ndarr




original=Image.open("face1.bmp")
arr = img2array(original)
X = np.array([arr])
imgarr = np.array(X)

        
for num in range(1,100):
    file_name = 'face'+str(num)+'.bmp'
    original=Image.open(file_name)
    arr = img2array(original)
    X = np.array([arr])     
    imgarr = np.concatenate((imgarr,X))