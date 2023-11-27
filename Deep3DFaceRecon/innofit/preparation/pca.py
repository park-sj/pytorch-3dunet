#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:48:15 2022

@author: workstation
"""

import sys
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


def readobj(file_path):
    def yield_file(in_file):
        f = open(in_file, encoding='latin1')
        buf = f.read()
        f.close()
        for b in buf.split('\n'):
            b = b.strip()
            if b.startswith('v '):
                yield ['v', [float(x) for x in b.split()[1:]]]
            elif b.startswith('vt '):
                yield ['vt', [float(x) for x in b.split()[1:]]]
            elif b.startswith('f '):
                triangle = b.split(' ')[1:]
                # -1 as .obj is base 1 but the Data class expects base 0 indices
                yield ['f', [[int(t.split("/")[0]) - 1 for t in triangle], [int(t.split("/")[1]) - 1 for t in triangle]]]
                # yield ['f', [[int(i) - 1 for i in t.split("/")] for t in triangle]]
            else:
                yield ['', ""]

    verts = []
    faces = []
    vts = []
    uvtris = []

    for k, v in yield_file(file_path):
        if k == 'v':
            verts.append(v)
        elif k == 'vt':
            vts.append(v)
        elif k == 'f':
            faces.append(v[0])
            uvtris.append(v[1])

    if not len(faces) or not len(verts):
        return None, None

    return np.array(verts), np.array(faces), np.array(vts), np.array(uvtris)


RESOLUTION = 128

if __name__ == "__main__":
    if len(sys.argv) != 3:
        file_dir = "/home/workstation/Libraries/jwson/data/mesh_data_for_PCA_registration-master/texture_selected_improved_aligned_registered_uvtextured"
        save_dir = "/home/workstation/Libraries/jwson/"
    else:
        file_dir = sys.argv[1]
        save_dir = sys.argv[2]
    
    patients = os.listdir(file_dir)
    
    if True:
        vert, face, vt, uvtri = readobj(os.path.join(file_dir, patients[0], "result_trimmed_baked.obj"))
        
        np.save(os.path.join(save_dir, "tri"), face)
        np.save(os.path.join(save_dir, "vt"), vt)
        np.save(os.path.join(save_dir, "uvtri"), uvtri)
        
        verts = np.zeros((len(patients), len(vert.reshape(-1))))
        
        rot = R.from_euler('xyz', [-90, 0, 0], degrees=True)
        for i, p in enumerate(patients):
            vert, _, _, _ = readobj(os.path.join(file_dir, p, "result_trimmed_baked.obj"))
            vert = rot.apply(vert)
            verts[i, :] = vert.reshape(-1)
            
        pca = PCA(n_components=80)
        pca.fit(verts)
        
        np.save(os.path.join(save_dir, "shapePC"), pca.components_.transpose())
        np.save(os.path.join(save_dir, "shapeMU"), pca.mean_)
        np.save(os.path.join(save_dir, "shapeEV"), pca.singular_values_)
    
    if True:
        colors = np.zeros((len(patients), RESOLUTION*RESOLUTION*3))
        for i, p in enumerate(patients):
            image = cv2.imread(os.path.join(file_dir, p, "result_trimmed_baked.png"))
            image = image.transpose(1,0,2)
            image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = image.astype(np.float32)/255.0
            colors[i, :] = image.reshape(-1)
        
        pca = PCA(n_components=50)
        pca.fit(colors)
        
        np.save(os.path.join(save_dir, "uvPC"), pca.components_.transpose())
        np.save(os.path.join(save_dir, "uvMU"), pca.mean_)
        np.save(os.path.join(save_dir, "uvEV"), pca.singular_values_)
    
        if True:
            # PCA 복원 퀄리티 확인
            restored = pca.inverse_transform(pca.transform(colors[-1:,:]))
            cv2.imwrite(os.path.join(save_dir, "pca128.png"), np.clip(255. * restored, 0, 255).astype(np.uint8).reshape(RESOLUTION,RESOLUTION,3))