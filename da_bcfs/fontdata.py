from __future__ import division
from data import LabeledDataProvider
import numpy as np
import math
#import tp_utils
import bcfstore
import util2
# from PIL import Image
import Image
from StringIO import StringIO
affine2d = __import__('affine2d')

class FontnetDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self,data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_classes = dp_params['num_classes']
        self.num_colors = 1
        self.image_width = dp_params['crop_size']
        self.image_height = dp_params['crop_size']
        self.squeeze_ratio = dp_params['squeeze_ratio']	
        self.num_views = 5*2
        self.multiview = dp_params['multiview_test'] and test
        self.feature_type = 0 # Default to no feature

    def tp_init(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False, random_shuffle=True):
        imagesInBatch = 128
        numBatches = len(batch_range)
        tp_dataStore, tp_labelStore = self.initialize(data_dir,test)

        tp_image_list = np.array(range(tp_dataStore.size()))
        print tp_dataStore.size(),'flag'
        if not test:
            if random_shuffle:
                print 'Randomizing batches...'
                np.random.shuffle(tp_image_list)

        tp_batches = util2.batches_from_list(tp_image_list,numBatches,imagesInBatch)

        tp_batch_dic = {}
        count = 0
        for iBatch in batch_range:
            tp_batch_dic[iBatch] = count
            count = count + 1

        self.tp_test = test
        self.random_shuffle = random_shuffle
        self.tp_imagesInBatch = imagesInBatch
        self.tp_numBatches = numBatches
        self.tp_dataStore = tp_dataStore
        self.tp_labelStore = tp_labelStore
        self.tp_image_list = tp_image_list
        self.tp_batches = tp_batches
        self.tp_batch_dic = tp_batch_dic

    def initialize(self, data_dir, test):
        dataStore, labelStore = util2.read_bcf_file(data_dir,test)
        return dataStore, labelStore

    def make_batch(self,batch_ind):
        data,labels = make_batch_fontnet_random_crop_intensity(self.tp_test, self.tp_dataStore, self.tp_labelStore, self.tp_batches[batch_ind], 128, self.squeeze_ratio, self.image_width, self.image_height)
        #data,labels = make_batch_fontnet_random_crop_intensity(self.tp_test, self.tp_dataStore, self.tp_labelStore, self.tp_batches[batch_ind], 128, self.image_width, self.image_height)
        return data, labels

    def get_next_batch(self):
	
        self.advance_batch()

        epoch = self.curr_epoch        
        batch_num = self.curr_batchnum

        tp_batch_ind = self.tp_batch_dic[batch_num] 

        if self.feature_type!=0:
            data, labels = make_batch_fontnet_feature(self.tp_dataStore, self.tp_labelStore, self.tp_batches[tp_batch_ind], 128, self.squeeze_ratio, self.image_width, self.image_height, self.feature_type)
        else:
            if self.multiview:
                data, labels = tp_utils.make_multiview_batch_jstore(self.tp_dataStore, self.tp_labelStore, self.tp_batches[tp_batch_ind], self.tp_average)
            else:
                data,labels = self.make_batch(tp_batch_ind)

        data = np.require(data,requirements='C')
        labels = np.require(labels,requirements='C')
        return epoch, batch_num, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        #print 'Data_dims: %s' % idx
        return self.image_width*self.image_height*self.num_colors if idx == 0 else 1

    def get_num_classes(self):
        #print len(self.tp_class_dict)
        return self.num_classes

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=np.single)

    def advance_batch(self):
        #print 'Im advancing the batch.'
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx]

        if self.batch_idx == 0: # we wrapped
            if not self.tp_test:
                if self.random_shuffle:
                    np.random.shuffle(self.tp_image_list)
            #self.tp_batches = tp_utils.batches_from_list(self.tp_image_list,self.tp_numBatches,self.tp_imagesInBatch)

            self.curr_epoch += 1

class FontnetMemoryDataProvider(FontnetDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        FontnetDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

    def initialize(self, data_dir, test):
        dataStore, labelStore = util2.read_bcf_memory(data_dir,test)
        return dataStore, labelStore

def make_batch_fontnet_feature(dataStore, labelStore, batch, average, width, height, feature_type):
    data = np.zeros((len(batch), width*height), dtype=np.single)
    labels = np.zeros((1, len(batch)), dtype=np.single)

    for count, i in enumerate(batch):
        im = np.array(Image.open(StringIO(dataStore.get(i))))
        y = np.round((im.shape[1]-width)/10.0*(feature_type-1))
        #if   feature_type==1:
        #    y = 0
        #elif feature_type==2:
        #    y = (im.shape[1]-width)/2
        #elif feature_type==3:
        #    y = im.shape[1]-width
        data[count,:] = np.reshape(im[:, y:y+width], (1,width*height))
        labels[0,count] = labelStore[i]

    data -= average
    return np.transpose(data), labels

def make_batch_fontnet_random_crop(dataStore, labelStore, batch, average, width, height):
    imagesInBatch = len(batch)
    data = np.zeros((imagesInBatch, width*height), dtype=np.single)
    labels = np.zeros((1,imagesInBatch), dtype=np.single)

    for count, i in enumerate(batch):
        im = np.array(Image.open(StringIO(dataStore.get(i))))
        im_width = im.shape[1]
        im_height = im.shape[0]
        A, b = generate_random_crop(im_width, im_height, width, height)
        img_dict = {'src':im, 'dst':np.reshape(data[count,:],[height,width]), 'A':A.astype(np.float32), 'b':b.astype(np.float32)}
        affine2d.apply([img_dict])
        data[count,:] += np.random.normal(scale=3.0, size=width*height).astype(np.float32)
        labels[0,count] = labelStore[i]

    data -= average
    return np.transpose(data), labels
        


def squeeze(img_origin, test, normal, squeeze_ratio):

    if test == False:
       	ratio = squeeze_ratio + np.random.uniform(-1,1)
    else:
	ratio =  squeeze_ratio

    im = np.array(img_origin)
    im_width = im.shape[1]
    im_height = im.shape[0]
    scale=normal/im_height
    im_height = normal
    im_width = int(scale*im_width/ratio)
    img_resized = img_origin.resize((im_width, im_height), Image.ANTIALIAS)
    new_width = im_width
    im_final = np.array(img_resized)
    #mean= np.mean(im_final)
    #print mean
    #im_final = im_final - average
    j=2
    while (new_width < im_height + 20):
        im_final = np.tile(np.array(img_resized),j) 
	new_width = im_final.shape[1]
	j=j+1
	        
    return im_final, new_width
            

def image_5view(im_final, height, width, im_height, new_width):

    img_corner=[]
    img_corner1 = im_final[0:height, 0:width]
    img_corner2 = im_final[0:height, 10:width+10]
    img_corner3 = im_final[0:height, new_width - width:new_width]
    img_corner4 = im_final[0:height, new_width - width- 10:new_width - 10]
    img_corner5 = im_final[(im_height - height)/2:(im_height+height)/2+1, (new_width-width)/2:(new_width+width)/2+1]
    img_corner5 = img_corner5[0:height, 0:width]
    img_corner = [img_corner1,img_corner2,img_corner3,img_corner4,img_corner5]

    return img_corner
        

def make_batch_fontnet_random_crop_intensity(test, dataStore, labelStore, batch, average, squeeze_ratio, width, height):

    imagesInBatch = len(batch)
    data = np.zeros((imagesInBatch, width*height), dtype=np.single)
    labels = np.zeros((1,imagesInBatch), dtype=np.single)
    normal = 105
    im_height = normal

    data1 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-top corner
    data2 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-bottom corner
    data3 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-top corner
    data4 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-bottom corner
    data5 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # center

    data01 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-top corner
    data02 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-bottom corner
    data03 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-top corner
    data04 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-bottom corner
    data05 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # center

    data10 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-top corner
    data20 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # left-bottom corner
    data30 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-top corner
    data40 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # right-bottom corner
    data50 = np.zeros((imagesInBatch, width*height), dtype=np.single)   # center

    for count, i in enumerate(batch):
        img_origin = Image.open(StringIO(dataStore.get(i)))
        if test == False:
		im_final, new_width = squeeze(img_origin, test, normal, squeeze_ratio)
                A, b = generate_random_crop2(new_width, im_height, width, height)
        	fg = np.random.uniform(low=140.0, high=220.0)
        	bg = np.random.uniform(low= 20.0, high=100.0)
        	theta = np.random.uniform(low=0.0, high=np.pi*2)
        	a = np.random.uniform(low=0.4, high=0.6)
        	img_dict = {'src':im_final, 'dst':np.reshape(data[count,:],[height,width]),'param':np.array([A[0,0],A[0,1],A[1,0],A[1,1],b[0],b[1],fg,bg,theta,a]).astype(np.float32)}
        	affine2d.apply2([img_dict])
        	data[count,:] += np.random.normal(scale=3.0, size=width*height).astype(np.float32)
                labels[0,count] = labelStore[i]

        if test == True:
		im_final1, new_width = squeeze(img_origin, test, normal, squeeze_ratio)
		img_corner1 = image_5view(im_final1, height, width, im_height, new_width)

		data1[count, :] = np.reshape(img_corner1[0],(1, height*width))
        	data2[count, :] = np.reshape(img_corner1[1],(1, height*width))
        	data3[count, :] = np.reshape(img_corner1[2],(1, height*width))
        	data4[count, :] = np.reshape(img_corner1[3],(1, height*width))
        	data5[count, :] = np.reshape(img_corner1[4],(1, height*width))

		im_final2, new_width = squeeze(img_origin, test, normal, squeeze_ratio-1)
		img_corner2 = image_5view(im_final2, height, width, im_height, new_width)

		data10[count, :] = np.reshape(img_corner2[0],(1, height*width))
        	data20[count, :] = np.reshape(img_corner2[1],(1, height*width))
        	data30[count, :] = np.reshape(img_corner2[2],(1, height*width))
        	data40[count, :] = np.reshape(img_corner2[3],(1, height*width))
        	data50[count, :] = np.reshape(img_corner2[4],(1, height*width))

		im_final3, new_width = squeeze(img_origin, test, normal, squeeze_ratio+1)
		img_corner3 = image_5view(im_final3, height, width, im_height, new_width)

		data01[count, :] = np.reshape(img_corner3[0],(1, height*width))
        	data02[count, :] = np.reshape(img_corner3[1],(1, height*width))
        	data03[count, :] = np.reshape(img_corner3[2],(1, height*width))
        	data04[count, :] = np.reshape(img_corner3[3],(1, height*width))
        	data05[count, :] = np.reshape(img_corner3[4],(1, height*width))

                labels[0,count] = labelStore[i]


    if test == False:
    	data -= average
    	data=np.transpose(data)

     
    if test == True:
   	data1 = data1 - average
    	data2 = data2 - average
    	data3 = data3 - average
   	data4 = data4 - average
    	data5 = data5 - average
   	data01 = data01 - average
    	data02 = data02 - average
    	data03 = data03 - average
   	data04 = data04 - average
    	data05 = data05 - average
   	data10 = data10 - average
    	data20 = data20 - average
    	data30 = data30 - average
   	data40 = data40 - average
    	data50 = data50 - average
        data=[np.transpose(data1), np.transpose(data2), np.transpose(data3), np.transpose(data4), np.transpose(data5), np.transpose(data01), np.transpose(data02), np.transpose(data03), np.transpose(data04), np.transpose(data05), np.transpose(data10), np.transpose(data20), np.transpose(data30), np.transpose(data40), np.transpose(data50)]

    return data, labels

def generate_random_crop(width1, height1, width2, height2):
    box2 = np.array([[0,0,width2,width2],[0,height2,height2,0]])
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    offset = np.zeros((2,1))

    success = 0
    while not success:
        scale = np.random.uniform(low=10.0/15.0, high=12.0/10.0)
        for i in range(1,10):
            offset[0] = np.random.uniform(low=-(width1 -width2 )/2.0, high=(width1 -width2 )/2.0)
            offset[1] = np.random.uniform(low=-(height1-height2)/2.0, high=(height1-height2)/2.0)
            theta = np.random.uniform(low=-3.0, high=3.0)*np.pi/180.0
            A[0,0] =  np.cos(theta)*scale
            A[0,1] = -np.sin(theta)*scale
            A[1,0] =  np.sin(theta)*scale
            A[1,1] =  np.cos(theta)*scale
            b = np.array([[width1],[height1]])/2.0+offset-np.dot(A,np.array([[width2],[height2]]))/2.0
            xy = np.dot(A,box2)+b

            if np.all(xy[0,:]>1) and np.all(xy[0,:]<(width1-2)) and np.all(xy[1,:]>1) and np.all(xy[1,:]<(height1-2)):
                success = 1
                break

    return A, b
    
def generate_random_crop2(width1, height1, width2, height2):
    box2 = np.array([[0,0,width2,width2],[0,height2,height2,0]])
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    offset = np.zeros((2,1))

    scale1 = np.random.uniform(low=10.0/11.0, high=13.0/10.0)
    scale1 = scale1*height1/float(height2)
    scale2 = scale1*np.random.uniform(low=10.0/11.0, high=11.0/10.0)
    offset[0] = np.random.uniform(low=-(width1 -width2*height1/float(height2))/2.0, high=(width1 -width2*height1/float(height2))/2.0)
    offset[1] = np.random.uniform(low=-height1/10.0, high=height1/10.0)
    theta = np.random.uniform(low=-3.0, high=3.0)*np.pi/180.0
    A[0,0] =  np.cos(theta)*scale1
    A[0,1] = -np.sin(theta)*scale2
    A[1,0] =  np.sin(theta)*scale1
    A[1,1] =  np.cos(theta)*scale2
    b = np.array([[width1],[height1]])/2.0+offset-np.dot(A,np.array([[width2],[height2]]))/2.0
    return A, b
