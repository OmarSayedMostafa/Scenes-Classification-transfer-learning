
import os, numpy as np, glob

def get_classes_weight(generator):
  
    classes = generator.classes # array[0,0,0,...,1,1,1,....,2,2,2,..]# get all data classes ids 
    classes, data_count = np.unique(classes, return_counts=True) # get the count of unique values in the data list 
#     print(sorted(data_count))
    class_weight = dict(zip(classes, data_count))
   
    maxValue = np.max(list(class_weight.values()))
    
    for key in class_weight.keys():
        class_weight[key]=float(maxValue)/float(class_weight[key])
    
    return class_weight


def to_one_hot(label_list):
    label_array = np.array(label_list)
    oneHot_array = np.zeros((label_array.size, label_array.max()+1))
    oneHot_array[np.arange(label_array.size),label_array] = 1
    return oneHot_array