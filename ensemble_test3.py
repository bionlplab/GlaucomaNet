import os
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from data_load_cv import load_data
from data_load_cv_vf import load_data_vf
from ImageGenerator_cv import DataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, Lasso
path = '/prj0129/mil4012/glaucoma' 



def get_train_test_p_id(glaucoma_list,normal_list, fold, total_num_fold):
    num_glaucoma = len(glaucoma_list) // 2
    test_num_glaucoma = num_glaucoma // total_num_fold * 2
    
    num_normal = len(normal_list) // 2
    test_num_normal = num_normal // total_num_fold * 2

    if fold == total_num_fold:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):,:]
        test_normal = normal_list[((fold-1) * test_num_normal):,:]
    else:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):fold * test_num_glaucoma,:]
        test_normal = normal_list[((fold-1) * test_num_normal):fold * test_num_normal,:]
    
    train_glaucoma = np.concatenate((glaucoma_list[0:((fold-1) * test_num_glaucoma),:], glaucoma_list[(fold * test_num_glaucoma):,:]), axis=0)
  #  train_glaucoma = glaucoma_list[0:((fold-1) * test_num_glaucoma),:] + glaucoma_list[(fold * test_num_glaucoma):,:]
  #  train_normal = normal_list[0:((fold-1) * test_num_normal),:] + normal_list[(fold * test_num_normal):,:]   
    train_normal = np.concatenate((normal_list[0:((fold-1) * test_num_normal),:], normal_list[(fold * test_num_normal):,:]), axis=0)

    valiation_glaucoma = train_glaucoma[int(0.8*len(train_glaucoma) // 2) * 2:,:]
    validation_normal = train_normal[(len(train_normal) - len(valiation_glaucoma)):,:] 
   # validation_normal = train_normal[(len(train_normal) - int(0.2*len(train_glaucoma) // 2) * 2):,:] 
   # validation_normal = train_normal[int(0.8*len(train_normal)):,:]
    train_glaucoma = train_glaucoma[0:(len(train_glaucoma)-len(valiation_glaucoma)) :]
   # train_glaucoma = train_glaucoma[0:int(0.8*len(train_glaucoma) // 2) * 2,:]
   # train_normal = train_normal[0:int(0.8*len(train_normal)),:]
    train_normal = train_normal[0:(len(train_normal) - len(validation_normal)),:]
    le_train_glaucoma = len(train_glaucoma)
    le_train_normal = len(train_normal)
    le_validation_glaucoma = len(valiation_glaucoma)
    le_validation_normal = len(validation_normal)
    
    le_test_glaucoma = len(test_glaucoma)
    le_test_normal = len(test_normal)
    
#    train_name = train_normal + train_glaucoma
#    test_name = test_normal + test_glaucoma
    
    train_name = np.concatenate((train_normal, train_glaucoma), axis=0)
    validation_name = np.concatenate((validation_normal, valiation_glaucoma), axis=0)
    test_name = np.concatenate((test_normal, test_glaucoma), axis=0)
    return train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal


def get_train_test_id(glaucoma_list,normal_list, fold, total_num_fold):
    num_glaucoma = len(glaucoma_list)
    test_num_glaucoma = num_glaucoma // total_num_fold
    
    num_normal = len(normal_list)
    test_num_normal = num_normal // total_num_fold

    if fold == total_num_fold:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):,:]
        test_normal = normal_list[((fold-1) * test_num_normal):,:]
    else:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):fold * test_num_glaucoma,:]
        test_normal = normal_list[((fold-1) * test_num_normal):fold * test_num_normal,:]
    
    train_glaucoma = np.concatenate((glaucoma_list[0:((fold-1) * test_num_glaucoma),:], glaucoma_list[(fold * test_num_glaucoma):,:]), axis=0)
  #  train_glaucoma = glaucoma_list[0:((fold-1) * test_num_glaucoma),:] + glaucoma_list[(fold * test_num_glaucoma):,:]
  #  train_normal = normal_list[0:((fold-1) * test_num_normal),:] + normal_list[(fold * test_num_normal):,:]   
    train_normal = np.concatenate((normal_list[0:((fold-1) * test_num_normal),:], normal_list[(fold * test_num_normal):,:]), axis=0)

    valiation_glaucoma = train_glaucoma[int(0.8*len(train_glaucoma)):,:]
    validation_normal = train_normal[(len(train_normal) - int(0.2*len(train_glaucoma))):,:] 
   # validation_normal = train_normal[int(0.8*len(train_normal)):,:]
    train_glaucoma = train_glaucoma[0:int(0.8*len(train_glaucoma)),:]
   # train_normal = train_normal[0:int(0.8*len(train_normal)),:]
    train_normal = train_normal[0:(len(train_normal) - len(validation_normal)),:]
    le_train_glaucoma = len(train_glaucoma)
    le_train_normal = len(train_normal)
    le_validation_glaucoma = len(valiation_glaucoma)
    le_validation_normal = len(validation_normal)
    
    le_test_glaucoma = len(test_glaucoma)
    le_test_normal = len(test_normal)
    
#    train_name = train_normal + train_glaucoma
#    test_name = test_normal + test_glaucoma
    
    train_name = np.concatenate((train_normal, train_glaucoma), axis=0)
    validation_name = np.concatenate((validation_normal, valiation_glaucoma), axis=0)
    test_name = np.concatenate((test_normal, test_glaucoma), axis=0)
    return train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal

def generate_weight(y_test):
    len_non = len(np.where(y_test==0))
    len_g = len(np.where(y_test==1))
    s_weight = np.zeros(len(y_test),)
    for i in range(len(y_test)):
        if y_test[i] == 0 :
            s_weight[i] = 1/len(y_test)
        else:
            s_weight[i] = (1/len(y_test)) * (len_non/len_g)
    return s_weight
    

def test_train(y_test, data_validation, regr):
#def test(x_test, y_test, model, weights):
    p_predict = np.zeros((len(y_test),len(data_validation)))
    for i in range(len(data_validation)):
        label = np.loadtxt(data_validation[i]) 
        p_predict[:,i] = np.reshape(label,(len(label),))
#        np.savetxt(weights[i][:-3]+'val.txt', np.reshape(label,(len(label),)))
#    p_test = get_test1(x_test, y_test, model, weights)
#    p_test = p_test/len(weights)
#     sample_weight = generate_weight(y_test)
#     regr.fit(p_predict,y_test,sample_weight)
    regr.fit(p_predict,y_test)
    p_test = regr.predict(p_predict)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]
#    print(p_test)
#    print(p_classes)
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
    
    return regr


def test(y_test, data_test, regr):
#def test(x_test, y_test, model, weights):
    p_predict = np.zeros((len(y_test),len(data_test)))
    for i in range(len(data_test)):
        label = np.loadtxt(data_test[i]) 
        p_predict[:,i] = np.reshape(label,(len(label),))
#        np.savetxt(weights[i][:-3]+'.txt', np.reshape(label,(len(label),)))
       # np.savetxt(weights[0][:-3]+'.txt', p_predict[:,i])
#    p_test = get_test1(x_test, y_test, model, weights)
#    p_test = p_test/len(weights)
    p_test = regr.predict(p_predict)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]
#    print(p_test)
#    print(p_classes)
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
    

    
def test_mean(y_test, data_test):
    p_predict = np.zeros((len(y_test),len(data_test)))
    for i in range(len(data_test)):
        label = np.loadtxt(data_test[i]) 
        p_predict[:,i] = np.reshape(label,(len(label),))
#    p_test = get_test1(x_test, y_test, model, weights)
    p_test = np.mean(p_predict,axis=1)
#    p_test = p_test/len(weights)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]
#    print(p_test)
#    print(p_classes)
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
    
if __name__ == '__main__':
    
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
    
    #print(test_name)

    #/prj0129/mil4012/   /home/mil4012/glaucoma
   # val_images,val_labels,test_images,test_labels = load_data(data_path='/prj0129/mil4012/image_crop/',label_path='/prj0129/mil4012/lab_new.csv',validation_name=validation_name,test_name=test_name)
    val_images,val_labels,test_images,test_labels,test_images_s, test_labels_s,test_images_un, test_labels_un = load_data(x_size,y_size, data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),
                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)
#    val_images1,val_labels,test_images1,test_labels,test_images_s1, test_labels_s,test_images_un1, test_labels_un = load_data(data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),
#                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
#                                                                                                                          validation_name=validation_name,test_name=test_name)
#    val_images2,val_labels,test_images2,test_labels,test_images_s2, test_labels_s,test_images_un2, test_labels_un = load_data(data_path=os.path.join(path,'image_crop3/'),label_path=os.path.join(path,'lab_new.csv'),
#                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
#                                                                                                                          validation_name=validation_name,test_name=test_name)
    test_images_vf, test_labels_vf = load_data_vf(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),vf_path=os.path.join(path,'patient_vf1.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)
#    test_images_vf1, test_labels_vf = load_data_vf(data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),vf_path=os.path.join(path,'patient_vf1.csv'),
#                                                                                                                          validation_name=validation_name,test_name=test_name)
    #val_labels,test_labels = load_data(data_path='/prj0129/mil4012/image_crop/',label_path='/prj0129/mil4012/lab_new.csv',validation_name=validation_name,test_name=test_name)

    print('the shape of testing image:', np.shape(test_images))



    
    train_generator, train_labels = DataGenerator(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)
    train_labels= train_labels.astype(np.float)    
#     train_generator, train_labels = DataGenerator(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal[le_train_glaucoma*2:le_train_glaucoma*3],train_glaucoma=train_glaucoma)
#    train_generator1, train_labels = DataGenerator(data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)
   # train_generator2, train_labels = DataGenerator(data_path=os.path.join(path,'image_crop3/'),label_path=os.path.join(path,'lab_new.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)
#     train_generator = resize(train_generator, (np.size(train_generator,0),331, 331, 3))
#     train_labels = train_labels.astype(np.float)
    val_labels = val_labels.astype(np.float)
    test_labels = test_labels.astype(np.float)
    test_labels_s = test_labels_s.astype(np.float)
    test_labels_un = test_labels_un.astype(np.float)
    test_labels_vf = test_labels_vf.astype(np.float)
    
    np.savetxt('/prj0129/mil4012/glaucoma/fold_p/foldtrain1'+'.txt', np.reshape(train_labels,(len(train_labels),)))
#     np.savetxt('/prj0129/mil4012/glaucoma/fold_p/foldval5'+'.txt', np.reshape(val_labels,(len(val_labels),)))
#     np.savetxt('/prj0129/mil4012/glaucoma/fold_p/foldtest5'+'.txt', np.reshape(test_labels,(len(test_labels),)))
#     np.savetxt('/prj0129/mil4012/glaucoma/fold_p/foldvf5'+'.txt', np.reshape(test_labels_vf,(len(test_labels_vf),)))

#     data_test = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f5_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f5_55.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f5_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f5new.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f5_5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f5_55.txt']
    
#     data_validation = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f5_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f5_55val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f5_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f5newval.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f5_5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f5_55val.txt']
    
#     data_test = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f4_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f4_55.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f4.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f4_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f4new.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f4_5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f4_55.txt']
    
#     data_validation = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f4_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f4_55val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f4val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f4_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f4newval.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f4_5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f4_55val.txt']

#     data_test = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f3_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f3_55.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f3_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f3new1.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f3_5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f3_55.txt']
    
#     data_validation = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f3_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f3_55val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f3_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f3new1val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f3_5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f3_55val.txt']
    
#     data_test = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f2_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f2_55.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f2new.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f2_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f2new2.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f2_5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f2_55.txt']
    
#     data_validation = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_f2_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_f2_55val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152_f2newval.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_f2_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201_f2new2val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_f2_5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_f2_55val.txt']
    
    
    data_test = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_55.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_3.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_5.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_55.txt']
    
    data_validation = ['/prj0129/mil4012/glaucoma/result_p/glaucoma_NASNetMobile_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Vgg16_55val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_ResNet152val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Mobilev2_3val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_DenseNet201val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_MultiNet1sp_5val.txt','/prj0129/mil4012/glaucoma/result_p/glaucoma_Xceptionorignial_55val.txt']
#     # random forest
#     regr = RandomForestRegressor(n_estimators=500,max_depth=2)
#     regr = test_train(val_labels, data_validation, regr)
#     test(test_labels, data_test, regr)
    
#     # macro averaging
#     test_mean(test_labels, data_test)
    
    # linear regression
    regr = LinearRegression()
    regr = test_train(val_labels, data_validation, regr)
    test(test_labels, data_test, regr)
    
    
#     # linear LASSO
#     regr = Lasso(alpha=0.1)
#     regr = test_train(val_labels, data_validation, regr)
#     test(test_labels, data_test, regr)
    
    