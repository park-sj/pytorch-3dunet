import numpy as np
import  torch
import torch.nn.functional as F
from scipy.io import loadmat
from util.load_mats import transferBFM09
import os
from pytorch3d.renderer import TexturesUV


def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3)
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]
        
        
class ParametricFaceModel:
    def __init__(self,
                param_folder='./innofit',
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,
                is_train=True,
                **kwargs):
        # model = loadmat(os.path.join(param_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = np.load(os.path.join(param_folder, 'shapeMU.npy')).astype(np.float32)/1e2
        # identity basis. [3*N,80]
        self.id_base = np.load(os.path.join(param_folder, 'shapePC.npy'))[:,:30].astype(np.float32)*np.reshape(np.load(os.path.join(param_folder, 'shapeEV.npy'))[:30].astype(np.float32), (-1, 30))/1e2
        # # expression basis. [3*N,64]
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = np.load(os.path.join(param_folder, 'uvMU.npy')).astype(np.float32)
        self.tex_resolution = int(np.sqrt(self.mean_tex.size//3))
        # texture basis. [3*N,80]
        self.tex_base = np.load(os.path.join(param_folder, 'uvPC.npy'))[:,:30].astype(np.float32)*np.reshape(np.load(os.path.join(param_folder, 'uvEV.npy'))[:30].astype(np.float32),(-1, 30))
        # face indices for each vertex that lies in. starts from 0. [N,8]
        # self.point_buf = model['point_buf'].astype(np.int64) - 1
        self.point_buf = np.load(os.path.join(param_folder, 'point_buf.npy')).astype(np.int64)
        # vertex indices for each face. starts from 0. [F,3]
        # self.face_buf = model['tri'].astype(np.int64) - 1
        self.face_buf = np.load(os.path.join(param_folder, 'tri.npy')).astype(np.int64)
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(np.load(os.path.join(param_folder, 'keypoints.npy'))).astype(np.int64)
        self.uvtri = np.load(os.path.join(param_folder, 'uvtri.npy')).astype(np.int64)
        self.vt = np.load(os.path.join(param_folder, 'vt.npy')).astype(np.float32)
        
        if is_train:
            # # vertex indices for small face region to compute photometric error. starts from 0.
            # self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
            # # vertex indices for each face from small face region. starts from 0. [f,3]
            # self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # # vertex indices for pre-defined skin region to compute reflectance loss
            # self.skin_mask = np.squeeze(model['skinmask'])
            self.front_mask = np.arange(4571)
            self.front_face_buf = self.face_buf
            self.skin_mask = np.arange(self.tex_resolution**2)
            # TODO: 모든 vertex를 다 마스크에 넣었는데 뒷통수 부분은 제외하고 얼굴에 해당하는 vertex만 넣어야함
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])
        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        
    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))
                
    def compute_shape(self, id_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)
        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        face_shape = id_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])
    
    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)
        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
            face_texture = torch.clamp(face_texture, min=0.0, max=1.0)
        
        vt = self.vt.unsqueeze(0).repeat(batch_size, 1, 1)
        uvtri = self.uvtri.unsqueeze(0).repeat(batch_size, 1, 1)
        face_texture = torch.flip(face_texture.reshape([batch_size, self.tex_resolution, self.tex_resolution, 3]), [1])
        texture = TexturesUV(verts_uvs=vt, faces_uvs=uvtri, maps=face_texture)
            
        return texture
    
    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm
    
    
    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, F, 3, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1)
        # face_color = face_color.transpose(0, 1)
        # face_color = torch.permute(face_color[self.face_buf], (2, 0, 1, 3))
        return face_color
    
    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])
        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])
        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)
    
    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape
    
    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj
    
    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)
    
    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)
        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints]
    
    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors
        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :30]
        tex_coeffs = coeffs[:, 194: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        coef_dict = self.split_coeff(coeffs)
        print("coef", coef_dict)
        face_shape = self.compute_shape(coef_dict['id'])
        rotation = self.compute_rotation(coef_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)
        face_proj = self.to_image(face_vertex)
        print("face_vertex", face_vertex)
        print("face_proj", face_proj)
        landmark = self.get_landmarks(face_proj)
        face_texture = self.compute_texture(coef_dict['tex'], normalize=False)
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])
        return face_vertex, face_texture, face_color, landmark