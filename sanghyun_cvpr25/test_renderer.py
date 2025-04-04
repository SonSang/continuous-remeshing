import torch as th
import nvdiffrast.torch as dr
from torch.utils.tensorboard import SummaryWriter
import trimesh

LIGHT_DIR = [0., 0., -1.]    #3

'''
Functions from continuous remeshing
'''
def _translation(x, y, z, device):
    return th.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = th.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None, n=None, f=None, image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    if n is None:
        n = 1
    if f is None:
        f = 50
    A = az_count
    P = pol_count
    C = A * P

    phi = th.arange(0,A) * (2*th.pi/A)
    phi_rot = th.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = th.arange(1,P+1) * (th.pi/(P+1)) - th.pi/2
    theta_rot = th.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = th.empty((C,4,4), device=device)
    mv[:] = th.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device, n=n, f=f)

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return th.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=th.float32)
    tri = tensor([[0, 1, 2]], dtype=th.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: th.Tensor, #C,4,4
            proj: th.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        try:
            self._glctx = dr.RasterizeGLContext()
        except Exception as e:
            print("Could not use GL rasterizer, falling back to CUDA rasterizer.")
            self._glctx = dr.RasterizeCudaContext()
        _warmup(self._glctx)

    def render(self,
            vertices: th.Tensor, #V,3 float
            normals: th.Tensor, #V,3 float
            faces: th.Tensor, #F,3 long
            ) ->th.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(th.int32)
        vert_hom = th.cat((vertices, th.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = th.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = th.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

def calc_face_normals(
        vertices:th.Tensor, #V,3 first vertex may be unreferenced
        faces:th.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->th.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = th.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = th.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
    return face_normals #F,3

def calc_vertex_normals(
        vertices:th.Tensor, #V,3 first vertex may be unreferenced
        faces:th.Tensor, #F,3 long, first face may be all zero
        face_normals:th.Tensor=None, #F,3, not normalized
        )->th.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = th.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    return th.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)

'''
Our adaptations
'''

class AlphaRenderer(NormalsRenderer):
    '''
    Renderer that renders
    * normal
    * depth
    * shillouette
    '''
    
    def __init__(
            self,
            mv: th.Tensor, #C,4,4
            proj: th.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        super().__init__(mv,proj,image_size)
        self._mv = mv
        self._proj = proj

    def forward(self,
                verts: th.Tensor,
                normals: th.Tensor,
                faces: th.Tensor,
                batches=None):

        if batches is None:
            mv = self._mv
            mvp = self._mvp
        else:
            mv = self._mv[batches]
            mvp = self._mvp[batches]
        
        '''
        Single pass without transparency.
        '''
        V = verts.shape[0]
        faces = faces.type(th.int32)
        vert_hom = th.cat((verts, th.ones(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        verts_clip = vert_hom @ mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, 
                                verts_clip, 
                                faces, 
                                resolution=self._image_size, 
                                grad_db=False) #C,H,W,4

        # view space normal;
        vert_normals_hom = th.cat((normals, th.zeros(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        vert_normals_view = vert_normals_hom @ mv.transpose(-2,-1) #C,V,4
        vert_normals_view = vert_normals_view[..., :3] #C,V,3
        # in the view space, normals should be oriented toward viewer, 
        # so the z coordinates should be negative;
        vert_normals_view[vert_normals_view[..., 2] > 0.] = \
            -vert_normals_view[vert_normals_view[..., 2] > 0.]
        vert_normals_view = vert_normals_view.contiguous()

        # view space lightdir;
        lightdir = th.tensor(LIGHT_DIR, dtype=th.float32, device=verts.device) #3
        lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3

        # normal;
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  #C,H,W,3
        pixel_normals_view = pixel_normals_view / th.clamp(th.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
        diffuse = th.sum(lightdir * pixel_normals_view, -1, keepdim=True)           #C,H,W,1
        diffuse = th.clamp(diffuse, min=0.0, max=1.0)
        diffuse = diffuse[..., [0, 0, 0]] #C,H,W,3

        # depth;
        verts_depth = (verts_clip[..., [2]] / verts_clip[..., [3]])     #C,V,1
        depth, _ = dr.interpolate(verts_depth, rast_out, faces) #C,H,W,1    range: [-1, 1], -1 is near, 1 is far
        depth = (depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]
        
        depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        depth = 1 - depth                           # C,H,W,1
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # exclude background;
        depth_info = {'raw': depth, 'max': max_depth, 'min': min_depth}

        # shillouette;
        alpha = th.clamp(rast_out[..., [-1]], max=1) #C,H,W,1
        
        col = th.concat((diffuse, depth, alpha),dim=-1) #C,H,W,5
        col = dr.antialias(col, rast_out, verts_clip, faces) #C,H,W,5
        return col, depth_info

class GTInitializer:

    def __init__(self, verts: th.Tensor, faces: th.Tensor, device: str):

        # geometry info;
        self.gt_vertices = verts
        self.gt_faces = faces
        self.gt_vertex_normals = calc_vertex_normals(verts, faces)

        # rendered images;
        self.gt_images = None
        self.gt_depth_info = None

    def render(self, renderer: AlphaRenderer):

        target_images, target_depth_info = \
            renderer.forward(self.gt_vertices, self.gt_vertex_normals, self.gt_faces)

        self.gt_images = target_images
        self.gt_depth_info = target_depth_info

        return self.gt_images

    def write(self, writer: SummaryWriter):

        if self.gt_images is None:
            raise ValueError("Ground truth image is None")

        for i in range(len(self.gt_images)):
            writer.add_image(f"gt/diffuse_{i}", self.diffuse_images()[i], 0, dataformats="HWC")
            writer.add_image(f"gt/depth_{i}", self.depth_images()[i], 0, dataformats="HWC")
            writer.add_image(f"gt/shillouette_{i}", self.shillouette_images()[i], 0, dataformats="HWC")
        writer.flush()

    def diffuse_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,:-2]

    def depth_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[-2,-2,-2]]

    def shillouette_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[-1,-1,-1]]


def import_mesh(fname, device, scale=0.8):
    if fname.endswith(".obj"):
        target_mesh = trimesh.load_mesh(fname, 'obj')
    elif fname.endswith(".stl"):
        target_mesh = trimesh.load_mesh(fname, 'stl')    
    elif fname.endswith(".ply"):
        target_mesh = trimesh.load_mesh(fname, 'ply')
    else:
        raise ValueError(f"unknown mesh file type: {fname}")

    target_vertices, target_faces = target_mesh.vertices,target_mesh.faces
    target_vertices, target_faces = \
        th.tensor(target_vertices, dtype=th.float32, device=device), \
        th.tensor(target_faces, dtype=th.long, device=device)
    
    # normalize to fit mesh into a sphere of radius [scale];
    if scale > 0:
        target_vertices = target_vertices - target_vertices.mean(dim=0, keepdim=True)
        max_norm = th.max(th.norm(target_vertices, dim=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale

    vertex_normals = th.tensor(target_mesh.vertex_normals, dtype=th.float32, device=device)
    vertex_colors = th.ones_like(target_vertices)
    
    ### get duplicate verts and define vertex normal from face normals
    face_vert_0 = target_vertices[target_faces[:, 0]]   # [F, 3]
    face_vert_1 = target_vertices[target_faces[:, 1]]   # [F, 3]
    face_vert_2 = target_vertices[target_faces[:, 2]]   # [F, 3]
    face_normal = th.cross(face_vert_1 - face_vert_0, face_vert_2 - face_vert_0, dim=-1)    # [F, 3]
    face_normal = face_normal / (th.norm(face_normal, dim=-1, p=2, keepdim=True) + 1e-6)

    n_verts = th.cat([face_vert_0, face_vert_1, face_vert_2], dim=0)   # [3F, 3]
    n_verts_normal = th.cat([face_normal, face_normal, face_normal], dim=0)   # [3F, 3]
    n_faces = th.arange(0, target_faces.shape[0], device=device)        # [F]
    n_faces = th.stack([n_faces, n_faces + target_faces.shape[0], n_faces + 2 * target_faces.shape[0]], dim=-1)  # [F, 3]
    n_verts_colors = th.ones_like(n_verts)   # [3F, 3]

    return n_verts, n_faces, n_verts_normal, n_verts_colors