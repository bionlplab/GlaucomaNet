import os
import numpy as np
# from skimage.io import imread
import cv2
import copy
from skimage.transform import resize
def load_data(x_size,y_size,data_path,label_path,image_s_path,uncentain_path,validation_name,test_name):

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
    
    #generate ran and tracking numer for image with ending -s 
    tmp_s = np.loadtxt(image_s_path, dtype=np.str, delimiter=",")
    ran_s = tmp_s[:,1]
    tracking_s = tmp_s[:,2]
    ran_s = ran_s[1:len(ran_s)]
    tracking_s = tracking_s[1:len(tracking_s)]
    
    #generate ran and tracking numer for image with uncentain label 
    tmp_un = np.loadtxt(uncentain_path, dtype=np.str, delimiter=",")
    ran_un = tmp_un[:,0]
    tracking_un = tmp_un[:,1]
    ran_un = ran_un[1:len(ran_un)]
    tracking_un = tracking_un[1:len(tracking_un)]
    
#     x_size = 331
#     y_size = 331
    val_images = np.ndarray((len(validation_name)*20, x_size, y_size,3))
   # val_images = []
    val_labels = []
    le = 0
    for i in range(len(validation_name)):        
        ind = np.argwhere(ran==validation_name[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == validation_name[i][1]:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                val_images[le] = cv2.resize(IM, (x_size, y_size))
                #val_images[le] = resize(IM, (x_size, y_size, 3))
                #val_images[le] = IM
                le += 1
                #val_images = np.append(val_images,IM)
                val_labels = np.append(val_labels,tmp1[int(ind[j])])
          #  continue 
    val_images = val_images[0:le,:,:,:]
    test_images = np.ndarray((len(test_name)*20, x_size, y_size,3))
    #test_images = []
    test_labels = []
    le = 0
    for i in range(len(test_name)):        
        ind = np.argwhere(ran==test_name[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == test_name[i][1]:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                test_images[le] = cv2.resize(IM, (x_size, y_size))
                #test_images[le] = resize(IM, (x_size, y_size, 3))
                #test_images[le] = IM
                #test_images = np.append(test_images,IM)
                le += 1
                test_labels = np.append(test_labels,tmp1[int(ind[j])])
        #    continue
    test_images = test_images[0:le,:,:,:]
    
    test_images_s = np.ndarray((len(test_name)*10, x_size, y_size,3))
    #test_images = []
    test_labels_s = []
    le = 0
    for i in range(len(test_name)):        
        ind = np.argwhere(ran==test_name[i][0])
        ind_s = np.argwhere(ran_s==test_name[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == test_name[i][1] and len(np.argwhere(tracking_s[ind_s]==tracking[int(ind[j])])) != 0:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                test_images_s[le] = cv2.resize(IM, (x_size, y_size))
              #  test_images_s[le] = resize(IM, (x_size, y_size, 3))
              #  test_images_s[le] = IM
                #test_images = np.append(test_images,IM)
                le += 1
                test_labels_s = np.append(test_labels_s,tmp1[int(ind[j])])
        #    continue
    test_images_s = test_images_s[0:le,:,:,:]
    
    test_images_un = np.ndarray((len(test_name)*10, x_size, y_size,3))
    #test_images = []
    test_labels_un = []
    le = 0
    for i in range(len(test_name)):        
        ind = np.argwhere(ran==test_name[i][0])
        ind_un = np.argwhere(ran_un==test_name[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == test_name[i][1] and len(np.argwhere(tracking_un[ind_un]==tracking[int(ind[j])])) != 0:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                test_images_un[le] = cv2.resize(IM, (x_size, y_size))
              #  test_images_un[le] = resize(IM, (x_size, y_size, 3))
              #  test_images_un[le] = IM
                #test_images = np.append(test_images,IM)
                le += 1
                test_labels_un = np.append(test_labels_un,tmp1[int(ind[j])])
        #    continue
    test_images_un = test_images_un[0:le,:,:,:]
                
   # return val_labels, test_labels
    return val_images,val_labels, test_images,test_labels, test_images_s, test_labels_s, test_images_un, test_labels_un