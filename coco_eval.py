import tensorflow as tf
import numpy as np
import cv2

def iou(r1, r2):
	
	intersect_w = np.maximum(np.minimum(r1[0]+r1[2], r2[0]+r2[2])-np.maximum(r1[0], r2[0]),0)
	intersect_h = np.maximum(np.minimum(r1[1]+r1[3], r2[1]+r2[3])-np.maximum(r1[1], r2[1]),0)
	area_r1 = r1[2]*r1[3]
	area_r2 = r2[2]*r2[3]
	intersect = intersect_w*intersect_h	
	union = area_r1 + area_r2 - intersect
	
	return intersect/union

def softmax(x):

    e_x = np.exp(x)
    return e_x/np.sum(e_x)

def sigmoid(x):

	return 1.0/(1.0 + np.exp(-x))

def preprocess_yolo_output(data):
	
	locations = []
	classes = []
	for i in range(grid_h):
		for j in range(grid_w):
			for k in range(n_anchors):
				
				class_vec = softmax(data[0, i, j, k*n_single_out+5:k*n_single_out+n_single_out])
				objectness = sigmoid(data[0, i, j, k*n_single_out+4])
				class_prob = objectness*class_vec		
				
				w = np.exp(data[0, i, j, k*n_single_out+2])*anchors[k][0]*net_scale
				h = np.exp(data[0, i, j, k*n_single_out+3])*anchors[k][1]*net_scale
				dx = sigmoid(data[0, i, j, k*n_single_out])
				dy = sigmoid(data[0, i, j, k*n_single_out+1])
				x = (j+dx)*net_scale-w/2.0
				y = (i+dy)*net_scale-h/2.0
				
				classes.append(class_prob[important_classes])
				locations.append([x, y, w, h])
								
	classes = np.array(classes)
	locations = np.array(locations)
	
	return classes, locations

def non_max_supression(classes,locations):

	classes = np.transpose(classes)
	indxs = np.argsort(-classes, axis=1)

	for i in range(classes.shape[0]):
		classes[i] = classes[i][indxs[i]]

	for class_idx, class_vec in enumerate(classes):
		for roi_idx, roi_prob in enumerate(class_vec):
			if roi_prob < prob_th:
				classes[class_idx][roi_idx] = 0
						
	for class_idx, class_vec in enumerate(classes):
		for roi_idx, roi_prob in enumerate(class_vec):
			
			if roi_prob == 0:
				continue
				
			roi = locations[indxs[class_idx][roi_idx]]
			
			for roi_ref_idx, roi_ref_prob in enumerate(class_vec):

				if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
					continue

				roi_ref = locations[indxs[class_idx][roi_ref_idx]]
					
				if iou(roi, roi_ref) > iou_th:
					classes[class_idx][roi_ref_idx] = 0
				
	return classes, indxs

def draw_objects(classes,rois,indxs,img):

	scale_w = img.shape[1]/float(input_w)
	scale_h = img.shape[0]/float(input_h)

	for class_idx, class_ in enumerate(classes):
		for loc_idx, class_prob in enumerate(class_):
			if class_prob>0:

				x = int(rois[indxs[class_idx][loc_idx]][0]*scale_w)
				y = int(rois[indxs[class_idx][loc_idx]][1]*scale_h)
				w = int(rois[indxs[class_idx][loc_idx]][2]*scale_w)
				h = int(rois[indxs[class_idx][loc_idx]][3]*scale_h)
				
				cv2.rectangle(img, (x,y), (x+w, y+h), colors[class_idx], 4)					
				text = names[class_idx] + ' %.2f'%(class_prob)
				cv2.putText(img, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[class_idx], 2, cv2.LINE_AA)
	return img

def read_anchors_file(file_path):

	anchors = []
	with open(file_path) as file:
		for line in file.read().splitlines():
			anchors.append(map(float,line.split()))

	return np.array(anchors)

def read_labels_file(file_path):
	
	classes, names, colors = [], [], []
	with open(file_path) as file:
		lines = file.read().splitlines()
		for line in lines:
			class_, name, color = line.split()
			classes.append(int(class_))
			names.append(name)
			colors.append(eval(color))

	return classes, names, colors

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

net_scale = 32.0
grid_w, grid_h = 39, 22
input_w,input_h= 1248, 704
prob_th, iou_th = 0.7, 0.5

important_classes, names, colors = read_labels_file('./coco_animals.labels')
anchors = read_anchors_file('./coco_anchors.txt')
n_anchors = np.shape(anchors)[0]
n_classes = 80
n_single_out = n_anchors + n_classes

sess = tf.Session() 
saver = tf.train.import_meta_graph('./tfmodel/yolo.meta')
saver.restore(sess, './tfmodel/yolo')
graph = tf.get_default_graph()
image_placeholder = graph.get_tensor_by_name("input:0")
output = graph.get_tensor_by_name("52-convolutional_2:0")

placeholders = [op.inputs[0] for op in graph.get_operations() if 'Placeholder' in op.type and 'is_training' in op.name]
d = dict((key, False) for key in placeholders)

cap = cv2.VideoCapture('./video.mp4')

while(cap.isOpened()):

	ret, frame = cap.read()
	
	if ret == True:
		
		img_for_yolo = cv2.resize(frame, (input_w, input_h))/255.0
		d.update({image_placeholder: [img_for_yolo]})
		data=sess.run(output, feed_dict=d)

		classes, rois = preprocess_yolo_output(data)
		classes, indxs = non_max_supression(classes, rois)
		img = draw_objects(classes, rois, indxs, frame)

		cv2.imshow('image', img)
		cv2.moveWindow('image', 0, 0)
		key = cv2.waitKey(30)
		if key == 27: break
