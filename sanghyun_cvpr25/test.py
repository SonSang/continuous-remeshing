import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os
import imageio
import time
from PIL import Image
import trimesh

import sys
sys.path.append('..')

from tqdm import tqdm
from test_renderer import AlphaRenderer, make_star_cameras, GTInitializer, calc_vertex_normals, import_mesh
from util.func import make_sphere
from core.opt import MeshOptimizer

from torch.utils.tensorboard import SummaryWriter

DOMAIN = 1.0
NUM_VIEWPOINTS = 8
IMAGE_SIZE = 512
DEVICE = 'cuda:0'

def save_image(img, path):
    with torch.no_grad():
        img = img.detach().cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='remeshing optimization')
    parser.add_argument('-o', '--out_dir', type=str, default='out')
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    
    parser.add_argument('-i', '--iter', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-domain', '--domain', type=float, default=None)
    FLAGS = parser.parse_args()
    device = 'cuda:0'

    logdir = FLAGS.out_dir
    logdir = os.path.join(logdir, time.strftime("%Y-%m-%d-%H-%M-%S"))

    if FLAGS.domain is not None:
        DOMAIN = FLAGS.domain
    
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, "flags.txt"), "w") as f:
        f.write(str(FLAGS))

    writer = SummaryWriter(logdir)
    
    # Load GT mesh
    gt_verts, gt_faces, gt_normals, gt_colors = import_mesh(FLAGS.ref_mesh, device, scale=DOMAIN)

    print("===== Ground truth mesh =====")
    print("Number of vertices: ", gt_verts.shape[0])
    print("Number of faces: ", gt_faces.shape[0])
    print("=============================")

    # save gt mesh;
    mesh = trimesh.base.Trimesh(vertices=gt_verts.cpu().numpy(), faces=gt_faces.cpu().numpy(), vertex_normals=gt_normals.cpu().numpy(), vertex_colors=gt_colors.cpu().numpy())
    mesh.export(os.path.join(logdir, "gt_mesh.obj"))

    '''
    Ground truth renderings, used for mv recon.
    '''
    num_viewpoints = NUM_VIEWPOINTS
    image_size = IMAGE_SIZE
    
    mv, proj = make_star_cameras(num_viewpoints, num_viewpoints, distance=2.0, r=0.6, n=1.0, f=3.0)
    proj = proj.unsqueeze(0).expand(mv.shape[0], -1, -1)
    renderer = AlphaRenderer(mv, proj, [image_size, image_size])

    gt_manager = GTInitializer(gt_verts, gt_faces, DEVICE)
    gt_manager.render(renderer)
    
    gt_diffuse_map = gt_manager.diffuse_images()#.cpu().clone()
    gt_depth_map = gt_manager.depth_images()#.cpu().clone()

    del gt_manager

    # save gt images;
    image_save_path = os.path.join(logdir, "gt_images")
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    for i in range(len(gt_diffuse_map)):
        save_image(gt_diffuse_map[i], os.path.join(image_save_path, "diffuse_{}.png".format(i)))
        save_image(gt_depth_map[i], os.path.join(image_save_path, "depth_{}.png".format(i)))

    '''
    Optimization
    '''
    vertices,faces = make_sphere(level=2,radius=.5)

    opt = MeshOptimizer(vertices, faces, lr=FLAGS.learning_rate, edge_len_lims=(0.02, 0.15))
    vertices = opt.vertices
    snapshots = []

    start_time = time.time()

    steps = FLAGS.iter
    bar = tqdm(range(steps))
    for i in bar:
        opt.zero_grad()

        batches = torch.randperm(gt_diffuse_map.shape[0])[:FLAGS.batch]
        
        normals = calc_vertex_normals(vertices, faces)
        cols, _ = renderer.forward(vertices, normals, faces, batches)

        diffuse = cols[..., :3]
        depth = cols[..., [3, 3, 3]]

        b_gt_diffuse_map = gt_diffuse_map[batches].to(DEVICE)
        b_gt_depth_map = gt_depth_map[batches].to(DEVICE)

        diffuse_loss = (diffuse - b_gt_diffuse_map).abs().mean()
        depth_loss = (depth - b_gt_depth_map).abs().mean()

        loss = diffuse_loss + depth_loss
        loss.backward()
        opt.step()

        vertices,faces = opt.remesh()

        bar.set_description("Loss: {:.6f}".format(loss.item()))

        with torch.no_grad():
            writer.add_scalar('loss', loss.item(), i)
            writer.add_scalar('diffuse_loss', diffuse_loss.item(), i)
            writer.add_scalar('depth_loss', depth_loss.item(), i)

        if i % 200 == 0:
            with torch.no_grad():
                mesh = trimesh.base.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
                mesh.export(os.path.join(logdir, "iter_{}.obj".format(i)))

    end_time = time.time()
    with open(os.path.join(logdir, "time.txt"), 'w') as f:
        f.write("Time: {:.6f} sec".format(end_time - start_time))

    with torch.no_grad():
        final_mesh = trimesh.base.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
        final_mesh.export(os.path.join(logdir, "final_mesh.obj"))