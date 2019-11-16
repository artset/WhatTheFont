import numpy
import bcfstore

def read_bcf_memory(data_path, test):
    file_base = {0:'train',1:'val',2:'test'}
    bcf_file = '%s/%s.bcf'%(data_path, file_base[test])
    lbl_file = '%s/%s.label'%(data_path, file_base[test])
    label_store = read_label(lbl_file)
    data_store = bcfstore.bcf_store_memory(bcf_file)
    return data_store, label_store

def read_bcf_file(data_path, test):
    file_base = {0:'train',1:'val',2:'test'}
    bcf_file = '%s/%s.bcf'%(data_path, file_base[test])
    lbl_file = '%s/%s.label'%(data_path, file_base[test])
    label_store = read_label(lbl_file)
    data_store = bcfstore.bcf_store_file(bcf_file)
    return data_store, label_store

def read_label(label_file):
    fi = open(label_file, 'rb')
    labels = numpy.fromstring(fi.read(), dtype=numpy.uint32)
    fi.close()
    return labels

def batches_from_list(image_list, numBatches, imagesInBatch):
    batches = []
    for i in xrange(0, numBatches):
        start = imagesInBatch*(i  )
        end   = imagesInBatch*(i+1)
        batches.append(image_list[start:end])
    return batches
