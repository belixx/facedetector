from PIL import Image
import numpy as np

def img2array(original):

	img=original.convert('L')  

	ndarr = np.array(img.getdata())

	return ndarr


        

for num1 in range(00,100):
            file_name ='face'+str(num1)+'.jpg'
            original=Image.open(file_name)
            arr = img2array(original)
            X = np.array([arr])
            if num1 ==0 :
                imgarr = np.array(X)
            else:        
                imgarr = np.concatenate((imgarr,X))
          
          

  #          file_name ='face'+str(num1)+'.jpg'
   #         original=Image.open(file_name)
   #         arr = img2array(original)
   #         X = np.array([arr])
   #         imgarr = np.array(X)
   #         imgarr = np.concatenate((imgarr,X))