from utilst import generate_results_csv
from utilst import create_directory
from utilst import read_dataset
from utilst import transform_mts_to_ucr_format
from utilst import visualize_filter
from utilst import viz_for_survey_paper
from utilst import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from constants import CLASSIFIERS
from constants import ARCHIVE_NAMES
from constants import ITERATIONS
from utilst import read_all_datasets
from collections import Counter
from imblearn.over_sampling import SMOTE
from scipy.io import loadmat
from imblearn.combine import SMOTEENN

classifier_name = 'fcn'
root_dir= '/home/awasthi/Task2/data-indus1/'
output_directory = '/home/awasthi/Task2/data-indus1/'


def oversampling(x_t,y_t):
                
    y = np.repeat(y_t,x_t.shape[1])
    x = np.reshape(x_t,(x_t.shape[0]*x_t.shape[1],x_t.shape[2]))
    oversample = SMOTE(random_state=40)
    X, Y = oversample.fit_resample(x, y)
    y_new = Y[1::x_t.shape[1]]
    x_new = np.reshape(X,(len(y_new),x_t.shape[1],x_t.shape[2]))
    return x_new,y_new

def data_indus1():
            
    data= loadmat('S:/Job/Time Series analysis/Task 2/data-sets/data-sets/data-indus1/data-indus1.mat')
    x_train= data['train']
    x_test = data['test']
    y_train = data['trainlabels']
    y_test = data['testlabels']
    dataset_name = 'data-indus1'
    classifier_name = 'fcn'
    output_directory = 'S:/Job/Time Series analysis/Task 2/data-sets/data-sets/data-indus1'


def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    x_train, y_train = oversampling(x_train,y_train)
    #count = Counter(y_train)
   
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### 
transform_mts_to_ucr_format()
a = loadmat('S:/Job/Time Series analysis/Task 2/dl-4-tsc-master/dl-4-tsc-master/results/data-indus1/fcn/data-indus1.mat')
x_train = np.load('S:/Job/Time Series analysis/Task 2/dl-4-tsc-master/dl-4-tsc-master/results/data-indus1/fcn/x_train.npy')
y_train = np.load('S:/Job/Time Series analysis/Task 2/dl-4-tsc-master/dl-4-tsc-master/results/data-indus1/fcn/y_train.npy')
x_test = np.load('S:/Job/Time Series analysis/Task 2/dl-4-tsc-master/dl-4-tsc-master/results/data-indus1/fcn/x_test.npy')
y_test = np.load('S:/Job/Time Series analysis/Task 2/dl-4-tsc-master/dl-4-tsc-master/results/data-indus1/fcn/y_test.npy')

x_train,y_train = oversampling(x_train, y_train)
archive_name = 'mts_archive'
dataset_name = 'data-indus1'
datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

fit_classifier()
print('DONE')

       