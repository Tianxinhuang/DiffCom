import argparse
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from pointnet2_ops.pointnet2_utils import QueryAndGroup

sys.path.append('mpeg-pcc-tmc13')
from testgpcc import gpcc_encode, gpcc_decode  # noqa: E402

from randloader import *  # noqa: F401,F403

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
workspace = './compress/workspace'
query_group = None


def init_query_group(radius: float, max_neighbors: int) -> None:
    global query_group
    query_group = QueryAndGroup(radius, max_neighbors)


def write_ply(filename, points_colors):
    with open(filename, 'w') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write(f'element vertex {len(points_colors)}\n')
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        if points_colors.shape[-1] >= 6:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
        file.write('end_header\n')
        if points_colors.shape[-1] >= 6:
            for p in points_colors:
                file.write(f'{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n')
        else:
            for p in points_colors:
                file.write(f'{p[0]} {p[1]} {p[2]}\n')


def read_ply(filename):
    points = []
    colors = []
    with open(filename, 'r') as file:
        is_header = True
        has_colors = True
        for line in file:
            if is_header:
                if 'property uchar red' in line:
                    has_colors = True
                if line.strip() == 'end_header':
                    is_header = False
            else:
                values = line.split()
                if has_colors and len(values) <= 4:
                    has_colors = False
                points.append([float(values[0]), float(values[1]), float(values[2])])
                if has_colors:
                    colors.append([int(values[3]), int(values[4]), int(values[5])])
    points = np.array(points)
    colors = np.array(colors) if has_colors else None
    if has_colors:
        points = np.concatenate([points, colors], axis=-1)
    return points


def read_ply_point_cloud(filepath):
    plydata = PlyData.read(filepath)
    vertex_data = plydata['vertex']
    x = np.array(vertex_data['x'])
    y = np.array(vertex_data['y'])
    z = np.array(vertex_data['z'])
    points = np.vstack((x, y, z)).T
    if {'red', 'green', 'blue'}.issubset(vertex_data.data.dtype.names):
        r = np.array(vertex_data['red'])
        g = np.array(vertex_data['green'])
        b = np.array(vertex_data['blue'])
        colors = np.vstack((r, g, b)).T
        return np.concatenate([points, colors], axis=-1)
    return points


def tran_ply(filename):
    data = read_ply_point_cloud(filename)
    os.remove(filename)
    write_ply(filename, data)


def get_voxs(fpsfunc, gt, mask, depth, numarr=None):
    if numarr is not None:
        numarr = np.array(numarr).reshape([depth, depth, depth])
    points = gt[:, :3]
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    else:
        points = points.float()
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    bbox_size = max_coords - min_coords
    voxel_size = bbox_size / depth
    voxel_indices = torch.floor((points - min_coords) / voxel_size).long()
    voxel_indices = torch.clamp(voxel_indices, min=0, max=depth - 1)
    numlist = np.zeros([depth, depth, depth])
    voxel_dict = defaultdict(list)
    for i, idx in enumerate(voxel_indices):
        key = (idx[0].item(), idx[1].item(), idx[2].item())
        voxel_dict[key].append(i)
    fgts_list = []
    ceni_list = []
    gt_list = []
    bboxlist = []
    for key, idxs in sorted(voxel_dict.items()):
        num_points_in_voxel = len(idxs)
        voxnum = numarr[key[0], key[1], key[2]] if numarr is not None else 0
        if num_points_in_voxel == 0:
            numlist[key[0], key[1], key[2]] = 0
            continue
        voxel_points = gt[idxs]
        voxel_min = voxel_points[:, :3].min(dim=0)[0]
        voxel_max = voxel_points[:, :3].max(dim=0)[0]
        voxel_bbox = torch.cat([voxel_min, voxel_max], dim=0)
        cennum = 1024
        if num_points_in_voxel < 4 * cennum and voxnum < 4:
            fgts_list.append(voxel_points)
            numlist[key[0], key[1], key[2]] = 0
        else:
            bboxlist.append(voxel_bbox)
            ceni_points = fpsfunc(voxel_points.unsqueeze(0), cennum) / mask.permute(0, 2, 1)
            ceni_list.append(ceni_points)
            gt_list.append(voxel_points)
            if numarr is not None:
                numlist[key[0], key[1], key[2]] = voxnum
            else:
                numlist[key[0], key[1], key[2]] = num_points_in_voxel // 1024
    fgts = torch.cat(fgts_list, dim=0) if fgts_list else torch.tensor([]).cuda()
    ceni = torch.cat(ceni_list, dim=0) if ceni_list else torch.tensor([]).cuda()
    voxbbox = torch.stack(bboxlist, dim=0) if bboxlist else torch.tensor([]).cuda()
    numlist = list(numlist.reshape([-1]))
    return [ceni, voxbbox, fgts, gt_list, numlist]


def preprocess(fpsfunc, gt, level, numarray=None):
    mask = torch.tensor([1, 1, 1, 255.0, 255.0, 255.0]).float().cuda().unsqueeze(0).unsqueeze(0)
    if numarray is None:
        result_octs = get_voxs(fpsfunc, gt.squeeze(0), mask, depth=level)
    else:
        result_octs = get_voxs(fpsfunc, gt.squeeze(0), mask, depth=level, numarr=numarray)
    final = result_octs
    numarray = final[-1]
    return final, numarray


def get_points(fixedpts, vpts, gts, mask):
    result = vpts.permute(0, 2, 1).reshape(-1, 6)
    result *= mask
    if len(fixedpts) > 0:
        result = torch.cat([result, fixedpts], dim=0)
    return result


def upscale(pts, use_color, reso=512):
    if use_color:
        coors, colors = pts[:, :3], pts[:, 3:]
        coors = reso * coors + reso
        return np.concatenate([coors, colors], axis=-1)
    return pts * reso + reso


def downscale(pts, use_color, reso=512):
    if use_color:
        coors, colors = pts[:, :3], pts[:, 3:]
        coors = (coors - reso) / reso
        return np.concatenate([coors, colors], axis=-1)
    return (pts - reso) / reso


def norm(pts, cen=None, r=None):
    coors, colors = pts[:, :3], pts[:, 3:]
    if cen is None or r is None:
        cens = np.mean(coors, axis=0, keepdims=True)
        radius = 0.5 * np.sqrt(np.max(np.sum((coors - cens) ** 2, axis=-1))) + 1e-8
    else:
        cens, radius = cen, r
    coors = (coors - cens) / radius
    results = np.concatenate([coors, colors], axis=-1)
    return results, cens, radius


def denorm(pts, cens, radius):
    coors, colors = pts[:, :3], pts[:, 3:]
    results = coors * radius + cens
    results = np.concatenate([results, colors], axis=-1)
    return results


def encode_pts(level, normstr, numarray, pts, drcpath, plyname='temp.ply', drcname='temp.drc'):
    numarray = np.reshape(np.array(numarray), [-1])
    plyname = os.path.join(workspace, plyname)
    drcname = os.path.join(workspace, drcname)
    use_color = pts.shape[-1] > 3
    write_ply(plyname, pts)
    gpcc_encode(plyname, drcname, False, rate=3, mtype=0, use_color=use_color)
    with open(drcname, 'rb') as f:
        censbit = f.read()
    with open(drcpath, 'wb') as f:
        f.write(level.to_bytes(1, 'big'))
        for value in numarray:
            f.write(int(value).to_bytes(2, 'big'))
        for istr in normstr:
            f.write(struct.pack('f', istr))
        f.write(censbit)


def decode_pts(drcpath):
    normstr = []
    numarray = []
    with open(drcpath, 'rb') as f:
        level = int.from_bytes(f.read(1), 'big')
        numlen = level ** 3
        for _ in range(numlen):
            numarray.append(int.from_bytes(f.read(2), 'big'))
        for _ in range(4):
            normstr.append(struct.unpack('f', f.read(4))[0])
        drcstr = f.read()
    fdrcname = os.path.join(workspace, 'decoded.drc')
    with open(fdrcname, 'wb') as f:
        f.write(drcstr)
    fplyname = os.path.join(workspace, 'decoded.ply')
    gpcc_decode(fdrcname, fplyname, False, mtype=0)
    tran_ply(fplyname)
    fcens = read_ply(fplyname)
    return level, fcens, np.array(normstr), np.array(numarray)


def decompress(sampler, ldecens, ldeboxes, ldegts, numlist, defgts, level):
    dresults = [torch.tensor([]).cuda()]
    resnum = 0
    for i in range(ldecens.shape[0]):
        igts = ldegts[i].cuda()
        addnum = numlist[i]
        if addnum == 0:
            dresults.append(ldecens[i].permute(1, 0) * sampler.mask[0])
            continue
        targetnum = addnum * 1024
        iternum = max((targetnum - 1024) // 3072, 1)
        box = ldeboxes[i:i + 1]
        jpc = []
        sampler.x_s = sampler.box_norm(ldecens[i:i + 1], box) * sampler.mask.permute(0, 2, 1)
        for _ in range(int(iternum)):
            npc = sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[None]))
            npc = sampler.box_denorm(npc, box)
            npc = npc.permute(0, 2, 1).reshape([-1, 6])
            jpc.append(npc)
        if len(jpc) < 1:
            jpc = ldecens[i].permute(1, 0) * sampler.mask[0]
        else:
            jpc = torch.cat(jpc, dim=0)
        centers = sampler.box_denorm(sampler.x_s, box).squeeze(0).permute(1, 0)
        jpc = torch.cat([jpc, centers], dim=0)
        resnum += targetnum
        dresults.append(jpc)
    dresults = torch.cat(dresults, dim=0)
    if dresults.shape[0] > 0:
        dresults = dresults[torch.randperm(dresults.shape[0])][:int(resnum)]
    if len(defgts) > 0:
        dresults = torch.cat([dresults, defgts], dim=0)
    return dresults


def knn_query(cens, points, bsize=16):
    if query_group is None:
        raise RuntimeError('QueryAndGroup is not initialized. Call init_query_group first.')
    xyz = []
    inpts = points.unsqueeze(0).repeat([bsize, 1, 1])
    num_full = cens.shape[0] // bsize
    for i in range(num_full):
        xyz.append(
            query_group(
                inpts[..., :3].contiguous(),
                cens[bsize * i:bsize * (i + 1), :3].permute(0, 2, 1).contiguous(),
                inpts[..., 3:].permute(0, 2, 1).contiguous(),
            )
        )
    remainder = cens.shape[0] % bsize
    if remainder > 0:
        start = num_full * bsize
        inpts = points.unsqueeze(0).repeat([remainder, 1, 1])
        xyz.append(
            query_group(
                inpts[..., :3].contiguous(),
                cens[start:, :3].permute(0, 2, 1).contiguous(),
                inpts[..., 3:].permute(0, 2, 1).contiguous(),
            )
        )
    knns = torch.cat(xyz, dim=0)
    knns = torch.cat([knns, cens.unsqueeze(-1)], dim=-1)
    wei = torch.zeros_like(knns.sum(1, keepdim=True)).repeat([1, 2, 1, 1])
    wei[:, :, :, -1] = 1.0
    wei[:, :, :, :-1] = 1e-4
    return wei, knns


def compress_pts(sampler, inpath, drcdir, dedir, level_index=10, use_scale=True, iterations=1500):
    levels = [1, 2, 3, 4, 5, 6, 7] #you can adjust it for better performance
    reso = levels[level_index - 1]
    level = level_index
    
    use_color =  sampler.readgt(inpath)
    name = os.path.basename(inpath)
    level_drc_dir = os.path.join(drcdir, str(level))
    level_de_dir = os.path.join(dedir, str(level))
    os.makedirs(level_drc_dir, exist_ok=True)
    os.makedirs(level_de_dir, exist_ok=True)
    drcpath = os.path.join(level_drc_dir, os.path.splitext(name)[0] + '.bin')
    depath = os.path.join(level_de_dir, name)
    
    gtpc = sampler.gts.detach()

    vgtpc = gtpc.squeeze().cpu().numpy()
    gtpc, cen, r = norm(gtpc.squeeze().cpu().numpy())
    vgtpc, cen, r = norm(vgtpc, cen, r)
    gtpc = torch.tensor(gtpc).float().cuda().unsqueeze(0)
    vgtpc = torch.tensor(vgtpc).float().cuda().unsqueeze(0)

    #Do the voxelization preprocessing
    octlist, numarray = preprocess(sampler.bdfunc, gtpc, reso)

    scale = 300 #you can adjust it for better performance
    varcens, varboxes, fgts, gts, numlist = octlist
    #if there are enough points as varcens, do the optimization, otherwise skip it
    if len(varcens) > 0:
        gpcc_gt = gtpc[0] / sampler.mask.squeeze(0)
        vgpcc = vgtpc[0] / sampler.mask.squeeze(0)
        wei, knns = knn_query(varcens, vgpcc, bsize=16)
        dataset = GTDataset([gts], 0, target_size=3072)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=gt_collate_fn, shuffle=True)
        sampler.training_setup([wei, knns])
        data_iter = iter(dataloader)
        for i in tqdm(range(iterations), desc='optimize', leave=False):
            try:
                batch, depthidx, pointidx = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch, depthidx, pointidx = next(data_iter)
            bsize = batch.shape[0]
            sampler.optimize(gpcc_gt, batch, [varboxes], depthidx, pointidx, bsize, i)
        fixedpts = fgts
        optpts = sampler.x_sp0[0]
        compressedpts = get_points(fixedpts, optpts, gts, sampler.mask.squeeze(0))
    else:
        compressedpts = fgts

    compressedpts = compressedpts.detach().cpu().numpy()

    #Do the scaling / normalization before encoding the G-PCC bitstream
    if use_scale:
        compressedpts = upscale(compressedpts, use_color, scale)
    else:
        compressedpts = denorm(compressedpts, cen, r)
    if not use_color:
        compressedpts = compressedpts[:, :3]

    normstr = np.concatenate([cen[0], np.array([r])])
    #Encode all the information into a bitstream
    encode_pts(reso, normstr, numarray, compressedpts, drcpath, plyname='temp.ply', drcname='temp.drc')

    #Sleep to imitate real data transmission delay
    time.sleep(1)

    #Decode the bitstream
    decoded_reso, depts, normstr, denumarr = decode_pts(drcpath)

    #Rescale / denormalize the decoded points
    cen, r = np.expand_dims(normstr[:3], axis=0), normstr[-1]
    if use_scale:
        depts = downscale(depts, use_color, scale)
    else:
        depts, cen, r = norm(depts.squeeze(), cen, r)

    depts = torch.tensor(depts).float().cuda()
    if depts.shape[1] < 6:
        padding = 255.0*torch.ones((depts.shape[0], 6 - depts.shape[1]), device=depts.device)
        depts = torch.cat([depts, padding], dim=1)

    start = time.time()
    #Do the voxelization preprocessing again for decompression
    deres, _ = preprocess(sampler.bdfunc, depts.unsqueeze(0), decoded_reso, denumarr)
    ldecens, ldeboxes, defgts, ldegts, denumlist = deres
    ldecens = ldecens.cuda()
    ldeboxes = ldeboxes.cuda()

    denumlist = [x for x in denumlist if x != 0]
    #Decompress/Upsample the point cloud in the spefic regions
    dresults = decompress(sampler, ldecens, ldeboxes, ldegts, denumlist, defgts, decoded_reso)
    
    #Back to original scales
    dresults = denorm(dresults.detach().cpu().numpy(), cen, r)

    write_ply(depath, dresults)


def parse_args():
    parser = argparse.ArgumentParser(description='Point cloud compression pipeline.')
    parser.add_argument('--level-list', type=int, nargs='+', default=[4, 5, 6, 7], help='Octree levels to process.')
    parser.add_argument('--drcdir', default='./compress/ours/bins/objects_voxs4', help='Directory for bitstreams.')
    parser.add_argument('--dedir', default='./compress/ours/outplys/objects_voxs4', help='Directory for reconstructed PLYs.')
    parser.add_argument('--indir', default='./compress/input/objects', help='Input directory with PLY files.')
    parser.add_argument('--iterations', type=int, default=1500, help='Optimizer iterations per sample.')
    parser.add_argument('--base-name', default='base40M', help='Base model config / checkpoint name.')
    parser.add_argument('--query-radius', type=float, default=0.05, help='QueryAndGroup radius.')
    parser.add_argument('--query-max-neighbors', type=int, default=31, help='QueryAndGroup neighbor count.')
    parser.add_argument('--workspace', default='./compress/workspace', help='Temp folder for gpcc I/O.')

    return parser.parse_args()


def load_base_components(base_name):
    print('creating base model...')
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_state_dict = load_checkpoint(base_name, device)
    base_model.load_state_dict(base_state_dict)
    return base_model, base_diffusion


def load_upsampler_components():
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    pretrained_state_dict = load_checkpoint('upsample', device)

    target_state_dict = upsampler_model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name in target_state_dict:
            target_state_dict[name].copy_(param)

    # upsampler_model.load_state_dict(pretrained_state_dict)
    upsampler_model.load_state_dict(target_state_dict)
    return upsampler_model, upsampler_diffusion


def build_sampler(device, upsampler_model, upsampler_diffusion):
    return PointCloudSampler(
        device=device,
        models=[upsampler_model],
        diffusions=[upsampler_diffusion],
        num_points=[4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[0.0],
        use_karras=[False],
        karras_steps=[128],
        sigma_min=[1e-3],
        sigma_max=[160],
        s_churn=[0],
    )


def main():
    args = parse_args()
    global workspace
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    init_query_group(args.query_radius, args.query_max_neighbors)
    # base_model, base_diffusion = load_base_components(args.base_name)
    upsampler_model, upsampler_diffusion = load_upsampler_components()
    sampler = build_sampler(device, upsampler_model, upsampler_diffusion)
    # sampler.base_model = base_model
    # sampler.base_diffusion = base_diffusion
    for ilevel in args.level_list:
        for name in os.listdir(args.indir):
            if not name.lower().endswith('.ply'):
                continue
            inpath = os.path.join(args.indir, name)
            if not os.path.isfile(inpath):
                continue
            compress_pts(
                sampler,
                inpath,
                args.drcdir,
                args.dedir,
                level_index=ilevel,
                use_scale=True,
                iterations=args.iterations,
            )


if __name__ == '__main__':
    main()