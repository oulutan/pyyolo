import pyyolo
import numpy as np
import sys
import cv2
import ipdb
import time

datacfg = 'cfg/coco.data'
#cfgfile = 'cfg/tiny-yolo.cfg'
#cfgfile = 'cfg/yolo.cfg'
cfgfile = 'cfg/yolo_oytun.cfg'
#weightfile = '../tiny-yolo.weights'
weightfile = './yolo.weights'
filename = 'data/oytun.jpg'
#filename = 'data/VIRAT_S_0401.jpg'
thresh = 0.20
hier_thresh = 0.5
# cam = cv2.VideoCapture(-1)
cam = cv2.VideoCapture('/home/oytun/programs/darknet/darknet/data/mydata/VIRAT_S_000002.mp4')
#ret_val, img = cam.read()
#print(ret_val)
#ret_val = cv2.imwrite(filename,img)
#print(ret_val)

pyyolo.init(datacfg, cfgfile, weightfile)

# from file
print('----- test original C using a file')
outputs = pyyolo.test(filename, thresh, hier_thresh)
for output in outputs:
	print(output)

# camera 
print('----- test python API using a file')
i = 1
while i < 50:
	start = time.time()
	ret_val, img_in = cam.read()
	#img_in = cv2.imread(filename)
	img = img_in.transpose(2,0,1)
	ch, hh, ww = img.shape[0], img.shape[1], img.shape[2]
	# print w, h, c 
	data = img.ravel()/255.0
	#data = np.ascontiguousarray(data, dtype=np.float32)
	data = np.asarray(data, dtype=np.float32, order="c")
	#ipdb.set_trace()
	outputs = pyyolo.detect(ww, hh, ch, data, thresh, hier_thresh)	
	#cv2.rectangle(img_in,(0, 0),(1920, 1080), 2, 10)
	for output in outputs:
		print(output)
		#ipdb.set_trace()
		#cv2.circle(img_in, (output['left'], output['top']), 10, (255,0,0), -1)
		cv2.rectangle(img_in,(output['left'], output['top']),(output['right'], output['bottom']), (255,0,0), 3)
	i = i + 1
	small = cv2.resize(img_in, (0,0), fx=0.5, fy=0.5) 
	cv2.imshow("video", small)
	cv2.waitKey(1)
	dur = time.time() - start
	print("FPS:%.1f" % (1.0/dur))


# free model
pyyolo.cleanup()
