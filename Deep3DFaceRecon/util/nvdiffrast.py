"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""
import pytorch3d.ops
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
from scipy.io import loadmat
from torch import nn

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, SoftPhongShader, FoVPerspectiveCameras, PointLights
from pytorch3d.ops import padded_to_packed
"""
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
"""

# def ndc_projection(x=0.1, n=1.0, f=50.0):
#     return np.array([[n/x,    0,            0,              0],
#                      [  0, n/-x,            0,              0],
#                      [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
#                      [  0,    0,           -1,              0]]).astype(np.float32)

class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        super(MeshRenderer, self).__init__()

        # x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        # self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
        #         torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.fov = rasterize_fov
        self.znear = znear
        self.zfar = zfar

    def forward(self, rasterizer, vertex, tri, textures=None, gamma=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N ,C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        # ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]

        # vertex_ndc = vertex @ ndc_proj.t()
        """
        if self.rasterizer is None:
            self.rasterizer = MeshRasterizer()
            print("create rasterizer on device cuda:%d"%device.index)
        """
        
        # ranges = None
        # if isinstance(tri, List) or len(tri.shape) == 3:
        #     vum = vertex_ndc.shape[1]
        #     fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
        #     fstartidx = torch.cumsum(fnum, dim=0) - fnum
        #     ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
        #     for i in range(tri.shape[0]):
        #         tri[i] = tri[i] + i*vum
        #     vertex_ndc = torch.cat(vertex_ndc, dim=0)
        #     tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()

        # rasterize
        """
        cameras = FoVPerspectiveCameras(
            device=device,
            fov=self.fov,
            znear=self.znear,
            zfar=self.zfar,
        )

        raster_settings = RasterizationSettings(
            image_size=rsize
        )
        """

        # print(vertex.shape, tri.shape)
        batch_size = vertex.size()[0]
        tri = tri.unsqueeze(0)
        tri = tri.expand(batch_size, tri.size()[1], tri.size()[2])

        if textures is None:
            textures = TexturesVertex(torch.ones_like(vertex).to(device))
        mesh = Meshes(vertex.contiguous()[...,:3], tri, textures=textures)

        #fragments = rasterizer(mesh, cameras = cameras, raster_settings = raster_settings)
        #print(vertex.get_device())
        #if rasterizer.get_device() != mesh.get_device():
        #    rasterizer.to(mesh.get_device())

        fragments = rasterizer(mesh)
        rast_out = fragments.pix_to_face.squeeze(-1)
        depth = fragments.zbuf

        # render depth
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if gamma is not None:
            cameras = FoVPerspectiveCameras(device=device, znear=self.znear, zfar=self.zfar, fov=self.fov)
            lights = PointLights(device=device, location=[[0.0, 0.0, 15.0]])
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
            image = shader(fragments, mesh)
            image = image[...,:3].permute(0, 3, 1, 2)
            print("image", image, torch.max(image), torch.min(image))
            
            # split_sizes = mesh.num_verts_per_mesh()[0].item()
            # first_idxs = torch.tensor([i*split_sizes for i in range(batch_size)], dtype=torch.int64).to(device)
            # total_faces = mesh.num_faces_per_mesh()[0].item()*batch_size
            # vertex_features_packed = padded_to_packed(gamma, first_idxs, total_faces)
            # face_attributes = vertex_features_packed[mesh.faces_packed(), ...]
            # gamma_pixel = pytorch3d.ops.interpolate_face_attributes(fragments.pix_to_face,
            #                                           fragments.bary_coords,
            #                                           face_attributes)
            # gamma_pixel = gamma_pixel.squeeze(-2).permute(0, 3, 1, 2)
            # print("gamma_pixel", gamma_pixel, torch.max(gamma_pixel), torch.min(gamma_pixel))
            # image = mask * image * gamma_pixel
            # image = mask * gamma_pixel
        
        return mask, depth, image

