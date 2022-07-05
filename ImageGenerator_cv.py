import os
import numpy as np
# from skimage.io import imread
import cv2
import copy
import random
from skimage.transform import resize
def DataGenerator(x_size,y_size,data_path,label_path,train_normal,train_glaucoma):
#     x_size = 331
#     y_size = 331
    tmp = np.loadtxt(label_path, dtype=np.str, delimiter=",")
    # delete one image because we don't have the jpg image, 8252 is the position of this item and 1 is related to the title
    tmp = np.delete(tmp,8252+1, axis = 0)
    ran = tmp[:,0]
    lr = tmp[:,1]
    tracking = tmp[:,2]
    tmp1=tmp[:,3]
    ran = ran[1:len(ran)]
    lr = lr[1:len(lr)]
    tracking = tracking[1:len(tracking)]
    tmp1=tmp1[1:len(tmp1)]
    
    glaucoma_images = np.ndarray((len(train_glaucoma)*20, x_size, y_size,3))
    glaucoma_labels = []
    le = 0
    for i in range(len(train_glaucoma)):        
        ind = np.argwhere(ran==train_glaucoma[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == train_glaucoma[i][1]:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                glaucoma_images[le] = cv2.resize(IM, (x_size, y_size))
               # glaucoma_images[le] = resize(IM, (x_size, y_size, 3))
               # glaucoma_images[le] = IM
                le += 1
                glaucoma_labels = np.append(glaucoma_labels,tmp1[int(ind[j])])
         #   continue
    glaucoma_images = glaucoma_images[0:le,:,:,:]
   # print('the length of glaucoma', len(glaucoma_images))
    
    normal_labels = []
    #sampling
#    non_ind_train = random.sample(range(0,len(train_normal)),len(train_glaucoma))
#    #non_ind_train = np.arange(len(train_glaucoma))
#   # print('the non_ind_train is', len(non_ind_train))
#    normal_images = np.ndarray((len(train_glaucoma)*20, 224, 224,3))
    
#    le = 0
#    for i in range(len(non_ind_train)):        
#        ind = np.argwhere(ran==train_normal[int(non_ind_train[i])][0])
#        for j in range(len(ind)):
#            if lr[int(ind[j])] == train_normal[int(non_ind_train[i])][1]:
#                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
#                IM = cv2.imread(data_paths)        
#                normal_images[i] = IM
#                le += 1
#                normal_labels = np.append(normal_labels,tmp1[int(ind[j])])
#           # continue
    
    normal_images = np.ndarray((len(train_normal)*20, x_size, y_size,3))
    le = 0
    for i in range(len(train_normal)):        
        ind = np.argwhere(ran==train_normal[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == train_normal[i][1]:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                normal_images[le] = cv2.resize(IM, (x_size, y_size))
              #  normal_images[le] = resize(IM, (x_size, y_size, 3))
              #  normal_images[i] = IM
                le += 1
                normal_labels = np.append(normal_labels,tmp1[int(ind[j])])
           # continue
    normal_images = normal_images[0:le,:,:,:]    
    train_images = np.concatenate((normal_images,glaucoma_images),axis=0)
    train_labels = np.concatenate((normal_labels,glaucoma_labels),axis=0)
    
    return train_images, train_labels
    #return train_images
    