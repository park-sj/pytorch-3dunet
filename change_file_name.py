import argparse
import glob
import os
from itertools import chain
from pathlib import Path
import natsort

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
    for ext in ['dcm']]))
    #return sorted(fnames)
    return natsort.natsorted(fnames)

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
    parser.add_argument('--all', type=str, default='test', help='CT directory')
    #parser.add_argument('--all_masks', type=str, default='train_masks', help='masks directory')   

    opt = parser.parse_args()    
    
    #os.path.join(opt.dataroot, opt.output)
    patients = natsort.natsorted(os.listdir(os.path.join(opt.dataroot, opt.all)))
    #masks = natsort.natsorted(os.listdir(os.path.join(opt.dataroot, opt.all_masks)))  
    
    for idx, patient in enumerate(patients):
        CTlist = listdir(os.path.join(opt.dataroot, opt.all)+ "/" + patient)
        for idx, dcm in enumerate(CTlist):
            newfilename = os.path.join(os.path.join(opt.dataroot, opt.all), patient) + "/DCT" + str(idx).zfill(4) + '.dcm'
            os.rename(dcm, newfilename)
            
     #for idx, patient_mask in enumerate(masks):
        #masklist = listdir(os.path.join(opt.dataroot, opt.all)+ "/" + patient_mask)
        #for idx, mask in enumerate(masklist):
            #newfilename = os.path.join(os.path.join(opt.dataroot, opt.all), mask) + "/IM" + str(idx).zfill(3) + '.dcm'
            #os.rename(dcm, newfilename)   

        
    #for idx, mask in enumerate(masks):
        
    
    