import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from models import vgg16, ensemble_vgg
from modelnew import Res, ensemble_res, Den, ensemble_model, ensemble_resden,ensemble_resden1, multiscale_Net, Multiscale_multimodel,triplescale_Net, Res1,Den1,mobv2, vgg_16, nas,naslarge, xception
from data_load_cv import load_data
from data_load_cv_vf import load_data_vf
import numpy as np
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import copy
from skimage.transform import resize
from ImageGenerator_cv import DataGenerator
path = '/prj0129/mil4012/glaucoma' 


def weighted_binary_crossentropy(y_true, y_pred) :
    weight = 1 - K.sum(y_true) /(K.sum(y_true) + K.sum(1 - y_true))
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight +  (1 - y_true) * K.log(1 - y_pred) * (1-weight))
    return K.mean(logloss, axis=-1)


def get_train_test_p_id(glaucoma_list,normal_list, fold, total_num_fold):
    num_glaucoma = len(glaucoma_list) // 2
    test_num_glaucoma = num_glaucoma // total_num_fold * 2
    
    num_normal = len(normal_list) // 2
    test_num_normal = num_normal // total_num_fold * 2

    if fold == total_num_fold:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):,:]
        test_normal = normal_list[((fold-1) * test_num_normal):,:]
        train_glaucoma = glaucoma_list[0:((fold-1) * test_num_glaucoma),:]
        train_normal = normal_list[0:((fold-1) * test_num_normal),:]
    else:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):fold * test_num_glaucoma,:]
        test_normal = normal_list[((fold-1) * test_num_normal):fold * test_num_normal,:]
        train_glaucoma = np.concatenate((glaucoma_list[0:((fold-1) * test_num_glaucoma),:], glaucoma_list[(fold * test_num_glaucoma):,:]), axis=0)
        train_normal = np.concatenate((normal_list[0:((fold-1) * test_num_normal),:], normal_list[(fold * test_num_normal):,:]), axis=0)
    
    valiation_glaucoma = train_glaucoma[int(0.8*len(train_glaucoma) // 2) * 2:,:]
    validation_normal = train_normal[(len(train_normal) - len(valiation_glaucoma)):,:] 
    train_glaucoma = train_glaucoma[0:(len(train_glaucoma)-len(valiation_glaucoma)) :]
    train_normal = train_normal[0:(len(train_normal) - len(validation_normal)),:]
    le_train_glaucoma = len(train_glaucoma)
    le_train_normal = len(train_normal)
    le_validation_glaucoma = len(valiation_glaucoma)
    le_validation_normal = len(validation_normal)
    
    le_test_glaucoma = len(test_glaucoma)
    le_test_normal = len(test_normal)

    train_name = np.concatenate((train_normal, train_glaucoma), axis=0)
    validation_name = np.concatenate((validation_normal, valiation_glaucoma), axis=0)
    test_name = np.concatenate((test_normal, test_glaucoma), axis=0)
    return train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal



def train(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    datagen.fit(x_train)
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    print('the program start to fit')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size= 64), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 64, epochs=epochs
                        , shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')





def test(x_test, y_test, x_test_s, y_test_s, x_test_vf, y_test_vf, model, weights):
#def test(x_test, y_test, model, weights):
    model.load_weights(weights)
    p_test = model.predict(x_test)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]

    print('the shape of test is', p_test.shape)
    accuracy = accuracy_score(y_test, p_classes)
    print('classification accuracy: ', accuracy)
    precision = precision_score(y_test, p_classes)
    print('precision: ', precision)
    recall = recall_score(y_test, p_classes)
    print('recall: ', recall)
    f1 = f1_score(y_test, p_classes)
    print('F1 score: ', f1)
    auc = roc_auc_score(y_test, p_test)
    print('AUC: ', auc)
    matrix = confusion_matrix(y_test, p_classes)
    print(matrix)
    

    return





if __name__ == '__main__':
    w_path2 = '/prj0129/mil4012/glaucoma/weights/glaucoma_DenseNet201.h5'
    w_path1 = '/prj0129/mil4012/glaucoma/weights/glaucoma_ResNet152.h5'
#     w_path2 = 'glaucoma_DenseNet201LAG_5.h5'
#     w_path1 = 'glaucoma_ResNet152LAG_5.h5'
    #model = vgg16(img_size=(224, 224, 3), scale=1,dropout=False)
    #model.load_weights('vgg16_glaucoma.h5')
    #model.summary()
   # model = vgg_16(vgg_en='vgg_16',img_size=(224, 224, 3), dropout=False)
  #  model = nas(nas_en ='nasmobile',img_size=(224, 224, 3), dropout=False)
  #  model = naslarge(naslarge_en = 'naslarge',img_size=(331, 331, 3), dropout=False)
   # model = xception(xcep_en = 'xception',img_size=(299, 299, 3), dropout=False)
   # model = mobv2(mob_en='mobv2',img_size=(224, 224, 3), dropout=False)
   # model = ensemble_vgg(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
   # model = ensemble_res(res_en=['res50','res101','res152'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
    #model = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
   # model = Res1(res_en='res152',img_size=(224, 224, 3), dropout=False)
   # model = Den1(den_en='den201',img_size=(224, 224, 3), dropout=False)
#     model = Res(res_en='res50',img_size=(224, 224, 3), dropout=False)
  #  model = ensemble_model(model_en=['res152','den201'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
  #  model = ensemble_resden(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
    #proposed
    model = ensemble_resden1(w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
  #  model = multiscale_Net(net='res152',img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = Multiscale_multimodel(img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = triplescale_Net(net='den201',img_size=(224, 224, 3), dropout=False, flag=0)
    learning_rate = 1e-4
    epochs = 15
    weights_path = '/prj0129/mil4012/glaucoma/weights/glaucoma_MultiNet1sp_5.h5'
    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy)
    
    label_path1 = os.path.join(path,'glaucoma_list_patient.csv')
    tmp = np.loadtxt(label_path1, dtype=np.str, delimiter=",")

    label_path2 = os.path.join(path,'normal_list_patient.csv')
    tmp_1 = np.loadtxt(label_path2, dtype=np.str, delimiter=",")

    tmp = tmp[1:,:] 
    tmp_1 = tmp_1[1:,:]
    fold = 1
    total_num_fold = 5
    x_size = 224
    y_size = 224
    train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal = get_train_test_p_id(tmp, tmp_1, fold, total_num_fold)
    

    val_images,val_labels,test_images,test_labels,test_images_s, test_labels_s,test_images_un, test_labels_un = load_data(x_size,y_size, data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),
                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)

    test_images_vf, test_labels_vf = load_data_vf(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),vf_path=os.path.join(path,'patient_vf1.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)

    
    
    train_generator, train_labels = DataGenerator(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)

    train_labels = train_labels.astype(np.float)
    val_labels = val_labels.astype(np.float)
    test_labels = test_labels.astype(np.float)
    test_labels_s = test_labels_s.astype(np.float)
    test_labels_un = test_labels_un.astype(np.float)
    test_labels_vf = test_labels_vf.astype(np.float)

   
 
    train(train_generator, train_labels, val_images, val_labels, model, epochs, weights_path)


    test(test_images, test_labels, test_images_s, test_labels_s, test_images_vf, test_labels_vf, model, weights_path)

