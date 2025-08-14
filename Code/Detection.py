# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from PIL import Image
import cv2
# class that defines and loads the kangaroo dataset
class Detection(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "history")
        # define data locations
		images_dir = dataset_dir + '/image/'
		annotations_dir = dataset_dir + '/annotation/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]

			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
 
	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('history'))
		return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']



# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "history_per_250_200_15_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	BACKBONE = "resnet50"
	VALIDATION_STEPS = 100
	#RPN_ANCHOR_SCALES = (32, 64)
	#RPN_ANCHOR_RATIOS = [ 1, 2,2.5]
# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, i):
    # load the image and mask
    image_id = dataset.image_ids[i]
    image = dataset.load_image(image_id)
    mask, _ = dataset.load_mask(image_id)
    # conver pixel values (e.g. center)
    scaled_image = mold_image(image,cfg)
    # conver image into one sample
    sample = expand_dims(scaled_image,0)
    #make prediction
    yhat = model.detect(sample,verbose=0)[0]
    #define subplot
    pyplot.subplot(1,2,1)
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.title('Actual')
    # plot masks
    for j in range(mask.shape[2]):
        pyplot.imshow(mask[:,:,j], cmap = 'gray', alpha = 0.3)
    # get the context for drawing boxes
    pyplot.subplot(1,2,2)
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.title('detected')
    ax = pyplot.gca()
    # plot each box
    for box in yhat['rois']:
        # get coordinates
        y1,x1,y2,x2 = box
        # calculate width and height of the box
        width,height = x2-x1, y2-y1
        # create the shape
        rect = Rectangle((x1,y1),width,height, fill = False, color = 'red')
        # draw the box
        ax.add_patch(rect)
    
    pyplot.savefig(r'F:\test\500 dataset\default-without rotation\Original test\epoch 50 (98.1)\img' + str(image_id) + '.png', dpi = 400, quality = 90 , optimize = True)
    pyplot.show()
        
     
# train set
dataset = Detection()
dataset.load_dataset(r'F:\project\Dataset\test Dataset seen by model', is_train=True)
dataset.prepare()
print('dataset: %d' % len(dataset.image_ids))
 

#%%
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=r'F:\MaskR-CNN\Mask_RCNN-master\logs\rgb', config=cfg)
# load model weights
model_path = r'F:\MaskR-CNN\Mask_RCNN-master\logs\rgb\model-rgb-x50epochs-augmentation-without-rotate-default20230304T2201\mask_rcnn_model-rgb-x50epochs-augmentation-without-rotate-default_0050.h5'
model.load_weights(model_path, by_name=True)
for i in range(0,50):
   y = plot_actual_vs_predicted(dataset, model, cfg, i)

# %%


