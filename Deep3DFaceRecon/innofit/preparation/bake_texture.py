import sys
import os
import bpy
#import imageio
import numpy as np
import cv2
import re

EXTRUDE=[15., 40., 55.]
N_PNG=len(EXTRUDE)
RESOLUTION = 128

def execute_quiet(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    result = func(*args, **kwargs)
    sys.stdout = old_stdout
    return result

def bake_texture(scanner_loc, innofit_loc, save_dir):
    # First delete all the default objects - cube, camera, light
    bpy.ops.object.select_all(action='SELECT')
    execute_quiet(bpy.ops.object.delete)

    ## Import innofit mesh and prepare to bake
    # 1. import
    imported_object = execute_quiet(bpy.ops.import_scene.obj, filepath=innofit_loc, split_mode='OFF')
    assert 'FINISHED' in imported_object, 'Failed to load innofit mesh!'
    innofit_obj = bpy.data.objects[0]    
    # 2. unwrap
    bpy.context.view_layer.objects.active = innofit_obj
    bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')
    print('Innofit mesh is unwrapped')
    # 3. make and connect nodes
    for mat in innofit_obj.data.materials:
        print(f'mat: {mat}')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        ps = nodes.get('Principled BSDF')
        # 3-a. Add image texture node and make new image
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Image Texture'
        texture_node.select = True
        nodes.active = texture_node
        image_name = innofit_obj.name + '_BakedTexture'
        img = bpy.data.images.new(image_name, RESOLUTION, RESOLUTION)
        texture_node.image = img
        links.new(texture_node.outputs['Color'], ps.inputs['Base Color'])
        # 3-b. Add Mapping node
        mapping_node = nodes.new('ShaderNodeMapping')
        mapping_node.select = True
        links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        # 3-c. Add UV Map node
        uvmap_node = nodes.new('ShaderNodeUVMap')
        uvmap_node.select = True
        links.new(uvmap_node.outputs['UV'], mapping_node.inputs['Vector'])

    # import scanner mesh
    imported_object = execute_quiet(bpy.ops.import_scene.obj, filepath=scanner_loc)
    assert 'FINISHED' in imported_object
    scanner_object = bpy.data.objects[1]

    # bake
    print("Selected objects: ", bpy.context.selected_objects)
    bpy.context.scene.render.engine = 'CYCLES'
    
    scanner_object.select_set(True)
    
    print(bpy.data.objects[0], bpy.data.objects[1])
    for i, extrude in enumerate(EXTRUDE):
        print(f"Baking texture with extrude {extrude}")
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={'COLOR'}, use_selected_to_active=True, target='IMAGE_TEXTURES', max_ray_distance=1000, cage_extrusion=extrude)
        img.save_render(filepath=os.path.join(save_dir, 'result_trimmed_baked'+str(i+1) +'.png'))
    
    bpy.ops.object.select_all(action='DESELECT')
    scanner_object.select_set(True)
    bpy.ops.object.delete()

    innofit_obj.select_set(True)

    output_loc = os.path.join(save_dir, 'result_trimmed_baked.obj')
    execute_quiet(bpy.ops.export_scene.obj, filepath=output_loc, use_uvs=True, use_normals=False, use_materials=True, keep_vertex_order=True, use_selection=True)
    
    with open(os.path.join(save_dir, 'result_trimmed_baked.mtl'), 'r') as f:
        filedata = f.read()
    filedata = filedata.replace('map_Kd .', 'map_Kd result_trimmed_baked.png')

    with open(os.path.join(save_dir, 'result_trimmed_baked.mtl'), 'w') as f:
        f.write(filedata)

    bpy.ops.wm.quit_blender()

'''
def merge2(save_dir):
    im_main = imageio.imread(os.path.join(save_dir, 'result_trimmed_baked1.png'))
    images = []
    start = time.time()
    for i in range(1,N_PNG):
        print(f'Merging - result_trimmed_baked'+str(i+1)+'.png')
        images.append(imageio.imread(os.path.join(save_dir, 'result_trimmed_baked'+str(i+1)+'.png')))
    for i, j in np.ndindex(im_main.shape[0], im_main.shape[1]):
        if all(im_main[i,j,:-1] == np.array([0,0,0],dtype=np.uint8)):
            for k, image in enumerate(images):
                if not all(image[i,j,:-1] == np.array([0,0,0],dtype=np.uint8)):
                    im_main[i,j,:] = image[i,j,:]
                    break
    end = time.time()
    print(f"{end-start : 5f} sec")
    imageio.imsave(os.path.join(save_dir, "result_trimmed_baked.png"), im_main)
    print("Merged image has been saved.")
'''
def merge(save_dir):

    make = np.ones(shape=(RESOLUTION,RESOLUTION,3), dtype=np.uint8)
    img = cv2.imread(os.path.join(save_dir, 'result_trimmed_baked1.png'))
    img2 = cv2.imread(os.path.join(save_dir, 'result_trimmed_baked2.png'))
    img3 = cv2.imread(os.path.join(save_dir, 'result_trimmed_baked3.png'))

    mask = cv2.subtract(make,img)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    img[mask>0] = img2[mask>0]
    mask = cv2.subtract(make,img)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    img[mask>0] = img3[mask>0]

    cv2.imwrite(os.path.join(save_dir, "result_trimmed_baked.png"), img)
    print("Merged image has been saved.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        scanner_loc = "/home/workstation/Libraries/jwson/data/mesh_data_for_PCA_registration-master/texture_selected_original_aligned2innofit"
        innofit_loc = "/home/workstation/Libraries/jwson/data/mesh_data_for_PCA_registration-master/texture_selected_improved_aligned_registered_uvmapped"
        save_dir = '/home/workstation/Libraries/jwson/data/mesh_data_for_PCA_registration-master/texture_selected_improved_aligned_registered_uvtextured'
    else:
        scanner_loc = sys.argv[1]
        innofit_loc = sys.argv[2]
        save_dir = sys.argv[3]
    # assert os.path.isfile(scanner_loc), f'{scanner_loc} file does not exist!'
    # assert os.path.isfile(innofit_loc), f'{innofit_loc} file does not exist!'
    
    patients = os.listdir(innofit_loc)
    
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]') # 한글과 띄어쓰기를 제외한 모든 글자
    for p in patients:
        p_hangul = hangul.sub('', p)
        
        innofit_mesh = os.path.join(innofit_loc, p)
        scanner_mesh = os.path.join(scanner_loc, p[6:-26], p_hangul+'.obj')
        assert os.path.isfile(innofit_mesh), f'{innofit_mesh} file does not exist!'
        assert os.path.isfile(scanner_mesh), f'{scanner_mesh} file does not exist!'
        
        os.makedirs(os.path.join(save_dir, p), exist_ok=True)
        bake_texture(scanner_mesh, innofit_mesh, os.path.join(save_dir, p))
        merge(os.path.join(save_dir, p))