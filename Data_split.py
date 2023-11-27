import argparse
import glob
import os
from itertools import chain
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
    for ext in ['dcm']]))
    return sorted(fnames)

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def changeCTFileName(directory):
    print(os.listdir(directory))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/media/shkim/7cf97277-d9cd-454f-994f-59734f6775d0/GNG_extraction', help='root directory of the dataset')
    parser.add_argument('--all', type=str, default='all', help='All CT directory')
    parser.add_argument('--all_masks', type=str, default='all_masks', help='All masks directory')   

    opt = parser.parse_args()    
    
    #os.path.join(opt.dataroot, opt.output)
    patients = os.listdir(os.path.join(opt.dataroot, opt.all))
    masks = os.listdir(os.path.join(opt.dataroot, opt.all_masks))
    
    #patients = glob.glob(dataroot + train)
    #masks = glob.glob(dataroot + train_masks)
    
    X_train, X_temp, y_train, y_temp = train_test_split(patients, masks, train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5)
    
    print(len(patients))
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))
        
    f_train = open(opt.dataroot +"/BONE_fullsize_trainlist.text", "w+")
    f_val = open(opt.dataroot + "/BONE_fullsize_vallist.text", "w+")
    f_test = open(opt.dataroot + "/BONE_fullsize_testlist.text", "w+")
    
    for idx, patient in enumerate(X_train):
        f_train.write(patient+"\n")
        shutil.move(os.path.join(opt.dataroot, opt.all)+"/" + patient, opt.dataroot + "/" + "train")
        shutil.move(os.path.join(opt.dataroot, opt.all_masks)+"/" + patient, opt.dataroot + "/" + "train_masks")
        
    for idx, patient in enumerate(X_val):
        f_val.write(patient+"\n")
        shutil.move(os.path.join(opt.dataroot, opt.all)+"/" + patient, opt.dataroot + "/" + "val")
        shutil.move(os.path.join(opt.dataroot, opt.all_masks)+"/" + patient, opt.dataroot + "/" + "val_masks")
        
    for idx, patient in enumerate(X_test):
        f_test.write(patient+"\n")
        shutil.move(os.path.join(opt.dataroot, opt.all)+"/" + patient, opt.dataroot + "/" + "test")
        shutil.move(os.path.join(opt.dataroot, opt.all_masks)+"/" + patient, opt.dataroot + "/" + "test_masks")
    
    f_train.close()
    f_val.close()
    f_test.close()
    
    