import cv2

def num(i):
	i+=1
	return i
cam = cv2.VideoCapture(0)

name = "Face Detecter"

hc = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

while True:
	i = 0
	s, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = hc.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		save = img[y:y+h,x:x+w]
		filename = 'face'+str(x+y)+'.bmp'
		img_resize = cv2.resize(save,(100,100))
		cv2.imwrite(filename,img_resize)
	
	cv2.imshow(name, img)
	k = cv2.waitKey(10)
	if k == 27:
		cv2.destroyWindow("Detect")
		break
