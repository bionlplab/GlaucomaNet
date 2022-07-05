import os
import numpy as np
# from skimage.io import imread
import cv2
import copy
from skimage.transform import resize
def load_data_vf(x_size,y_size,data_path,label_path,vf_path,validation_name,test_name):

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
    
    #generate ran and tracking numer for image with vf 
    tmp_vf = np.loadtxt(vf_path, dtype=np.str, delimiter=",")
    ran_vf = tmp_vf[:,0]
    tracking_vf = tmp_vf[:,1]
    ran_vf = ran_vf[1:len(ran_vf)]
    tracking_vf = tracking_vf[1:len(tracking_vf)]
    

    
#    val_images = np.ndarray((len(validation_name)*20, 224, 224,3))
#   # val_images = []
#    val_labels = []
#    le = 0
#    for i in range(len(validation_name)):        
#        ind = np.argwhere(ran==validation_name[i][0])
#        for j in range(len(ind)):
#            if lr[int(ind[j])] == validation_name[i][1]:
#                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
#                IM = cv2.imread(data_paths)
#                val_images[le] = IM
#                le += 1
#                #val_images = np.append(val_images,IM)
#                val_labels = np.append(val_labels,tmp1[int(ind[j])])
#          #  continue 
#    val_images = val_images[0:le,:,:,:]
#    test_images = np.ndarray((len(test_name)*20, 224, 224,3))
#    #test_images = []
#    test_labels = []
#    le = 0
#    for i in range(len(test_name)):        
#        ind = np.argwhere(ran==test_name[i][0])
#        for j in range(len(ind)):
#            if lr[int(ind[j])] == test_name[i][1]:
#                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
#                IM = cv2.imread(data_paths)        
#                test_images[le] = IM
#                #test_images = np.append(test_images,IM)
#                le += 1
#                test_labels = np.append(test_labels,tmp1[int(ind[j])])
#        #    continue
#    test_images = test_images[0:le,:,:,:]
    
#     x_size = 331
#     y_size = 331
    test_images_vf = np.ndarray((len(test_name)*10, x_size, y_size,3))
    #test_images = []
    test_labels_vf = []
    le = 0
    for i in range(len(test_name)):        
        ind = np.argwhere(ran==test_name[i][0])
        ind_tf = np.argwhere(ran_vf==test_name[i][0])
        for j in range(len(ind)):
            if lr[int(ind[j])] == test_name[i][1] and len(np.argwhere(tracking_vf[ind_tf]==tracking[int(ind[j])])) != 0:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                test_images_vf[le] = cv2.resize(IM, (x_size, y_size))
                #test_images_vf[le] = resize(IM, (x_size, y_size, 3))
               # test_images_vf[le] = IM
                #test_images = np.append(test_images,IM)
                le += 1
                test_labels_vf = np.append(test_labels_vf,tmp1[int(ind[j])])
        #    continue
    test_images_vf = test_images_vf[0:le,:,:,:]
                
    return test_images_vf, test_labels_vf
    #return val_images,val_labels, test_images,test_labels, test_images_s, test_labels_s, test_images_un, test_labels_un