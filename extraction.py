import argparse
import itertools
import glob
import random
import os
from itertools import chain
from pathlib import Path
import numpy as np
import zipfile
import natsort

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
    parser.add_argument('--dataroot', type=str, default='/media/shkim/7cf97277-d9cd-454f-994f-59734f6775d0', help='root directory of the dataset')
    parser.add_argument('--target', type=str, default='GNG', help='target dataset')
    parser.add_argument('--output', type=str, default='GNG_extraction', help='output root directory')
    parser.add_argument('--train', type=str, default='train', help='output CT directory')
    parser.add_argument('--train_masks', type=str, default='train_masks', help='output mask directory')
    parser.add_argument('--train_masks_BONE', type=str, default='train_masks_BONE', help='output BONE mask directory')    

    opt = parser.parse_args()    
    targetroot = os.path.join(opt.dataroot, opt.target)
    outputroot = os.path.join(opt.dataroot, opt.output)
    
    patients = os.listdir(targetroot)
    patients = natsort.natsorted(patients)
    
    for idx, patient in enumerate(patients):
        zipfiles = os.listdir(os.path.join(targetroot, patient))
        try:
            BONE_MASK, SKIN_MASK, CT = natsort.natsorted(zipfiles)
            patientNum = patient.split("_")[0]
            
            BONE_MASK_zip = zipfile.ZipFile(os.path.join(os.path.join(targetroot, patient), BONE_MASK))
            #SKIN_MASK_zip = zipfile.ZipFile(os.path.join(os.path.join(targetroot, patient), SKIN_MASK))
            CT_zip = zipfile.ZipFile(os.path.join(os.path.join(targetroot, patient), CT))
            
            BONE_MASK_DIR = os.path.join(os.path.join(outputroot, opt.train_masks_BONE), opt.target + '_' + patientNum)    
            #SKIN_MASK_DIR = os.path.join(os.path.join(outputroot, opt.train_masks), opt.target + '_' + patientNum)  
            CT_DIR = os.path.join(os.path.join(outputroot, opt.train), opt.target + '_' + patientNum)            
            
            createDirectory(BONE_MASK_DIR)
            #createDirectory(SKIN_MASK_DIR)
            createDirectory(CT_DIR)            
            
            BONE_MASK_zip.extractall(BONE_MASK_DIR)
            #SKIN_MASK_zip.extractall(SKIN_MASK_DIR)
            CT_zip.extractall(CT_DIR)
            
            #changeCTFileName("/media/shkim/7cf97277-d9cd-454f-994f-59734f6775d0/seg_data/skin/train/WR_HwangYuNa") 
            #changeCTFileName(BONE_MASK_DIR)
            #changeCTFileName(SKIN_MASK_DIR)
            #changeCTFileName(CT_DIR)
            
            BONE_MASK_zip.close()
            #SKIN_MASK_zip.close()
            CT_zip.close()
            
        except:
            print("Not enough image in " + opt.dataroot+patient)
           
        #fantasy_zip.extract('Fantasy Jungle.pdf', 'C:\\Stories\\Fantasy')
        #fantasy_zip.close()
        
    #print(os.listdir(opt.dataroot))
    #print(glob.glob(opt.dataroot+'*'))
    
    #listdir(opt.dataroot)