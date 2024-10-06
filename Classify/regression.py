

import os
import io
import math

import cPickle as pickle
import numpy as np


all_train_data = './dataset/all_train_data'
train_data_1 = './dataset/train_data_1'
train_data_2 = './dataset/train_data_2'
test_data = './dataset/test_data'
                            

def sigmoid(x):
    try:
        res = 1 / (1 + math.exp(-x))
    except OverflowError:
        res = 1 if x>0 else 0
    return res


def logistic_regression(train_data_path, step_size= 0.01, iterations = 100, train_method = 'sgd', lambda_ = 0.1, sample_weights = None):

    data = train_data_path.split('/')[-1]
    model_path = './models/lr_%s(%s, %s)_on_%s'%(train_method, iterations, step_size, data)
    if not os.path.exists('./models/'):
        os.mkdir('./models/')
    if sample_weights != None or not os.path.exists(model_path):
        w = np.zeros([1, 2331]) # length of features plus one for the bias
        if train_method == 'sgd': # train with sgd
            old_w = None
            for i in xrange(iterations):
                if i > 0:
                    diff = np.abs(w - old_w).sum()
                 
                old_w = np.array(w)
                with io.open(train_data_path, mode = 'r', encoding = 'utf8') as rf:
                    idx = 0
                    line = rf.readline()
                    while line:
                        fields = map(lambda x: float(x), line.strip().split())
                        y = fields[0]
                        fields[0] = 1.0 # bias item
                        x = np.array(fields)
                        if sample_weights != None:  # adaboost
                            w += step_size*sample_weights[idx]*(y - sigmoid(np.dot(w, x)))*x
                            idx += 1
                        else:
                            w += step_size*(y - sigmoid(np.dot(w, x)))*x
                        line = rf.readline()
        elif train_method == 'bgd':  # train with bgd
            for i in xrange(iterations):
                if i > 0:
                    diff = np.abs(w - old_w).sum()
                    print 'bgd, %s iterations, coefficients abs difference: %s' %(i+1, diff)
                old_w = np.array(w)
                offset = np.zeros([1, 2331])
                with io.open(train_data_path, mode = 'r', encoding = 'utf8') as rf:
                    line = rf.readline()
                    idx = 0
                    while line:
                        fields = map(lambda x: float(x), line.strip().split())
                        y = fields[0]
                        fields[0] = 1.0 # bias item
                        x = np.array(fields)
                        if sample_weights != None:
                            offset += sample_weights[idx]*(y - sigmoid(np.dot(w, x)))*x
                            idx += 1
                        else:
                            offset += (y - sigmoid(np.dot(w, x)))*x
                        line = rf.readline()
                # regularization
                offset += 2 * lambda_ * w
                w += step_size * offset
        else:
            print 'train method %s is not available '%(train_method)
            return 
   
        if sample_weights != None:
            return w 
        pickle.dump(w, open(model_path, 'w'))
    return model_path


def predict(test_data_file, model_path):
    if not os.path.exists(model_path):
        print 'model not ready'
        return
    w = pickle.load(open(model_path))[0]
    TP, FP, TN, FN = 0, 0, 0, 0
    with io.open(test_data_file, mode = 'r', encoding = 'utf8') as rf:
        line = rf.readline()
        count = 1
        while line:
            fields = map(lambda x: float(x), line.strip().split())
            y = fields[0]
            fields[0] = 1.0 # bias item
            x = np.array(fields)
            predict = 1 if sigmoid(np.dot(w, x)) >= 0.5 else 0
            line = rf.readline()
            # evaluate predicting result
            if y == 1:
                if predict == y:
                    TP += 1
                else:
                    FN += 1
            if y == 0:
                if predict == y:
                    TN += 1
                else:
                    FP += 1
    print 'TP:%s, FN:%s, TN:%s, FP:%s'%(TP, FN, TN, FP)           
    precision, recall = 1.0*TP/(TP+FP), 1.0*TP/(TP+FN)
    print "precision: %s, recall: %s"%(precision, recall)
    return precision, recall




if __name__ == '__main__':
    
    # customized parameters
    step_size = 0.01
    iterations =  1500
    train_method = 'acf'
    lambda_ = 0.1  # parameter for regularization
    
    print '=============training on train data 1======================'
    model_path_1 = logistic_regression(train_data_1, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    print '=============training on train data 2======================'
    model_path_2 = logistic_regression(train_data_2, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    
    print '=============performance on train data 1======================'
    p1, r1 = predict(train_data_1, model_path_2)
    print '=============performance on train data 2======================'
    p2, r2 = predict(train_data_2, model_path_1)
    print '=============average performance on train data 1 and 2======================'
    p, r = (p1+p2)/2.0, (r1+r2)/2.0
    print "precision: %s, recall: %s"%(p, r)
    """
    print '=============training on all data ======================'
    model_path = logistic_regression(all_train_data, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    p, r = predict(test_data, model_path)
    print "precision: %s, recall: %s"%(p, r)
    """



