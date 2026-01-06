"""
Helpers for sampling from a single- or multi-stage point cloud diffusion model.
"""

from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from point_e.util.point_cloud import PointCloud

from .gaussian_diffusion import GaussianDiffusion
from .k_diffusion import karras_sample_progressive

import open3d as o3d
import numpy as np
import sys
sys.path.append('ChamferDistancePytorch')
import chamfer3D.dist_chamfer_3D
import chamfer6D.dist_chamfer_6D
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import random

import copy
class PointCloudSampler:
    """
    A wrapper around a model or stack of models that produces conditional or
    unconditional sample tensors.

    By default, this will load models and configs from files.
    If you want to modify the sampler arguments of an existing sampler, call
    with_options() or with_args().
    """

    def __init__(
        self,
        device: torch.device,
        models: Sequence[nn.Module],
        diffusions: Sequence[GaussianDiffusion],
        num_points: Sequence[int],
        aux_channels: Sequence[str],
        model_kwargs_key_filter: Sequence[str] = ("*",),
        guidance_scale: Sequence[float] = (3.0, 3.0),
        clip_denoised: bool = True,
        use_karras: Sequence[bool] = (True, True),
        karras_steps: Sequence[int] = (64, 64),
        sigma_min: Sequence[float] = (1e-3, 1e-3),
        sigma_max: Sequence[float] = (120, 160),
        s_churn: Sequence[float] = (3, 0),
    ):
        n = len(models)
        assert n > 0

        if n > 1:
            if len(guidance_scale) == 1:
                # Don't guide the upsamplers by default.
                guidance_scale = list(guidance_scale) + [1.0] * (n - 1)
            if len(use_karras) == 1:
                use_karras = use_karras * n
            if len(karras_steps) == 1:
                karras_steps = karras_steps * n
            if len(sigma_min) == 1:
                sigma_min = sigma_min * n
            if len(sigma_max) == 1:
                sigma_max = sigma_max * n
            if len(s_churn) == 1:
                s_churn = s_churn * n
            if len(model_kwargs_key_filter) == 1:
                model_kwargs_key_filter = model_kwargs_key_filter * n
        if len(model_kwargs_key_filter) == 0:
            model_kwargs_key_filter = ["*"] * n
        assert len(guidance_scale) == n
        assert len(use_karras) == n
        assert len(karras_steps) == n
        assert len(sigma_min) == n
        assert len(sigma_max) == n
        assert len(s_churn) == n
        assert len(model_kwargs_key_filter) == n

        self.device = device
        self.num_points = num_points
        self.aux_channels = aux_channels
        self.model_kwargs_key_filter = model_kwargs_key_filter
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.use_karras = use_karras
        self.karras_steps = karras_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn

        self.models = models
        self.diffusions = diffusions
        self.embed = None
        self.gts = None

        self.chamfer_dist = chamfer6D.dist_chamfer_6D.chamfer_6DDist()
        self.chamfer_3d = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        self.stage_model_kwargs = None

    @property
    def num_stages(self) -> int:
        return len(self.models)

    def sample_batch(self, batch_size: int, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        samples = None
        for x in self.sample_batch_progressive(batch_size, model_kwargs):
            samples = x
        return samples

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.diffusions[0].betas.shape[0]), batch_size)

        return torch.tensor(ts, device=self.device)
    
    def save_ply(self, points_with_colors, path):
        points_with_colors[:, 3:] = (points_with_colors[:, 3:]).astype(np.uint8)

        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(points_with_colors[:, :3])
        
        pcd.colors = o3d.utility.Vector3dVector(points_with_colors[:, 3:] / 255.0)
        
        o3d.io.write_point_cloud(path, pcd)
    
    def loss_oembed(self, reals, ptcond, batch_size):

        t = torch.tensor(np.random.choice(np.arange(1, self.startidx), batch_size), device=self.device)

        output = self.diffusions[0].training_losses(self.models[0], reals.permute(0,2,1), t, model_kwargs=dict(low_res=ptcond, hybrid=True))


        maxbox = self.x_s0[:,:3].max(-1)[0] #b*3
        minbox = self.x_s0[:,:3].min(-1)[0]
        boxloss = F.relu(self.x_s[:,:3]-maxbox.unsqueeze(-1)).max() + F.relu(minbox.unsqueeze(-1)-self.x_s[:,:3]).max()

        x_0 = output['pred_xstart']#[:1].squeeze(0).permute(1,0)
        reals = torch.cat([reals[:,:,:3], reals[:,:,3:]/255.0], dim=-1)
        x_0 = torch.cat([x_0[:,:3], x_0[:,3:]/255.0], dim=1)
        cd = self.chamfer_p(reals, x_0.permute(0, 2, 1))

        return output["mse"].sum() #+ 20*boxloss #+ 1*cd #+ 20.0*boxloss #+ 1*loss2 #+ 2*boxloss #+ 0.00001*loss2#+ 0.05*noutput["loss"].mean()

    def voxel_random(self, points, ptnum=1024, vsize=0.1):
        #idx = np.random.choice(np.arange(1, points.shape[1]), points.shape[0])
        idx = np.arange(1, points.shape[1])

        vpts = []
        for i in range(0, points.shape[0]):
            cen = self.gts[i,idx[i]:idx[i]+1]
            inid = (self.gts[i,:,:3]-cen[:,:3]).square().sum(-1).sqrt()<=vsize/2

            inpts = self.gts[i:i+1, inid]

            inidi = np.random.choice(np.array(range(inpts.shape[1])), ptnum, replace=True)
            vpts.append(inpts[:,inidi])
        vpts = torch.cat(vpts, dim=0)
        return vpts
        
    #L_{CDM}
    def loss_inverse(self, reals, ptcond, batch_size):
        #t = torch.tensor(np.random.choice(np.arange(1, self.diffusions[0].betas.shape[0]), batch_size), device=self.device)
        t = torch.tensor(np.random.choice(np.arange(1, self.startidx), batch_size), device=self.device)
        #t = torch.tensor([step],device=self.device)#self.uniform_sample_t(batch_size)
        x_t = self.diffusions[0].q_sample(self.diffusions[0].scale_channels(reals.permute(0, 2, 1)), t)
        real_t1 = self.diffusions[0].q_sample(self.diffusions[0].scale_channels(reals.permute(0, 2, 1)), t-1)

        x_t1 = self.diffusions[0].p_sample(self.models[0], x_t, t, model_kwargs=dict(low_res=ptcond))["sample"]
        self.x0 = x_t1[:1].squeeze(0).permute(1,0)


        real_t1 = torch.cat([real_t1[:,:3], real_t1[:,3:]], dim=1)
        x_t1 = torch.cat([x_t1[:,:3], x_t1[:,3:]], dim=1)

        cd = self.chamfer_p(real_t1.permute(0, 2, 1), x_t1.permute(0, 2, 1))
        return cd

    def loss_inversex0(self, reals, ptcond, batch_size):
        #t = torch.tensor(np.random.choice(np.arange(1, 128), batch_size), device=self.device)
        t = torch.tensor(np.random.choice(np.arange(1, self.startidx), batch_size), device=self.device)
        x_t = self.diffusions[0].q_sample(self.diffusions[0].scale_channels(reals.permute(0, 2, 1)), t)

        x_0 = self.diffusions[0].p_sample(self.models[0], x_t, t, model_kwargs=dict(low_res=ptcond))["pred_xstart"]
        self.x0 = x_0[:1].squeeze(0).permute(1,0)
        reals = torch.cat([reals[:,:,:3], reals[:,:,3:]/255.0], dim=-1)
        x_0 = torch.cat([x_0[:,:3], x_0[:,3:]/255.0], dim=1)
        cd = self.chamfer_p(reals, x_0.permute(0, 2, 1))
        return cd

    def box_detach(self, tensor, maxids = None, minids = None):
        result = tensor.clone()
        if maxids is None or minids is None:
            maxids = tensor.max(-1)[1]
            minids = tensor.min(-1)[1]

        for i in range(tensor.shape[0]):
            result[i,:,maxids[i]] = result[i,:,maxids[i]].detach().clone().requires_grad_(False)
            result[i,:,minids[i]] = result[i,:,minids[i]].detach().clone().requires_grad_(False)

        return result, maxids, minids
    
    #Detach points on the boundary box
    def list_detach(self, tensor_list):
        len_tensor = len(tensor_list)
        result = []
        
        for i in range(len_tensor):
            if len(self.maxids) < i+1 or len(self.minids) < i+1:
                iresult, imax, imin = self.box_detach(tensor_list[i])
                self.maxids.append(imax)
                self.minids.append(imin)
            else:
                iresult, _, _ = self.box_detach(tensor_list[i], self.maxids[i], self.minids[i])
            result.append(iresult)


        return result

    #b*2*1024*k, b*6*1024*k
    def wei2cens(self, wei, knns):
        if wei.shape[1]<=1:

            wei = wei.abs()+1e-5
            wei = wei/(wei.sum(-1, keepdim=True))
            cens = (wei * knns).sum(-1)
        else:
            wei = wei.abs()#+1e-3
            wei = wei/(wei.sum(-1, keepdim=True))

            coors, colors = knns[:,:3], knns[:,3:]
            coors = (coors*wei[:,:1]).sum(-1)
            colors = (colors*wei[:,1:]).sum(-1)
            cens = torch.cat([coors, colors], dim=1)
        return cens

    def training_setup(self, centers):

        self.wei, self.knns = centers
        self.x_sp0 = self.wei2cens(self.wei, self.knns)

        self.x_p0 = [self.x_sp0.clone().detach()]

        self.wei = nn.Parameter(self.wei)

        self.maxids = []
        self.minids = []

        self.startidx = 8
        self.optimizer = torch.optim.Adam([{'params': [self.wei], 'lr': 0.001, "name": "x_s"}])


    # Function to recursively convert all tensors in a nested list to Variables
    def convert_to_variable(self,nested_list):
        for i in range(len(nested_list)):
            if isinstance(nested_list[i], torch.Tensor):
                # Convert Tensor to Variable (now requires_grad=True for autograd)
                #nested_list[i] = nested_list[i].requires_grad_()
                nested_list[i] = nn.Parameter(nested_list[i])
                nested_list[i].requires_grad = True

            elif isinstance(nested_list[i], list):
                # Recursively apply to inner list
                convert_to_variable(nested_list[i])


    def fps_tensor(self, points, num):

        aidx = list(range(points.shape[1]))
        fidx = furthest_point_sample(points[:,:,:3].contiguous(), num)
        lidx = []

        for i in range(fidx.shape[0]):
            lidx.append(list(set(aidx)-set(fidx[i].cpu().numpy())))
        lidx = torch.tensor(np.array(lidx), dtype = torch.int32, device=self.device)

        fgts = gather_operation(points.permute(0,2,1).contiguous(), fidx)#B, 3, num
        #fgts = fgts.permute(0, 2, 1)

        lgts = gather_operation(points.permute(0,2,1).contiguous(), lidx)
        lgts = lgts.permute(0, 2, 1)

        return fgts, lgts

    def fpsfunc(self, points, num):        
        fidx = furthest_point_sample(points[:,:,:3].contiguous(), num)
        fgts = gather_operation(points.permute(0,2,1).contiguous(), fidx)#B, 3, num
        return fgts

    #Boundary sampling, points: b*n*3
    def bdfunc(self, points, num):

        cens = self.fpsfunc(points, num).permute(0,2,1)

        upcen = cens[:,:,:3].max(dim=1, keepdim=True)[0]
        downcen = cens[:,:,:3].min(dim=1, keepdim=True)[0] 

        upxyz = points[:,:,:3].max(dim=1, keepdim=True)[0]
        downxyz = points[:,:,:3].min(dim=1, keepdim=True)[0]

        result = torch.where(torch.greater_equal(cens[:,:,:3], upcen.repeat([1, num, 1])), upxyz.repeat([1,num, 1]), cens[:,:,:3])
        result = torch.where(torch.less_equal(result, downcen.repeat([1, num, 1])), downxyz.repeat([1,num, 1]), result)

        result = torch.cat([result, cens[:,:,3:]], dim=-1)

        return result.permute(0,2,1)

    def get_cen(self, cens, depthidx, pointidx):
        didx = np.array(depthidx,dtype=np.int32)
        pidx = np.array(pointidx,dtype=np.int32)
        result = []
        num = len(didx)
        for i in range(num):
            result.append(cens[didx[i]][pidx[i]].unsqueeze(0))
        result = torch.cat(result, dim=0)
        return result

    def get_info(self, cens, boxes, depthidx, pointidx):
 
        didx = np.array(depthidx,dtype=np.int32)
        pidx = np.array(pointidx,dtype=np.int32)
        result = []
        boxres = []
        num = len(didx)

        for i in range(num):
            result.append(cens[didx[i]][pidx[i]].unsqueeze(0))
            boxres.append(boxes[didx[i]][pidx[i]].unsqueeze(0))
        result = torch.cat(result, dim=0)
        boxres = torch.cat(boxres, dim=0).cuda()

        return result, boxres

    #pts:b*6*ptnum
    def box_norm(self, pts, boxes):
        coors = pts[:,:3]
        cens = (boxes[:,:3]+boxes[:,3:])/2
        r = (1e-8+(cens-boxes[:,:3]).square().sum(-1, keepdim=True)).sqrt()
        coors = 0.25 * (coors-cens.unsqueeze(-1))/r.unsqueeze(-1)

        result = torch.cat([coors, pts[:,3:]], dim=1)
        return result

    def box_denorm(self, pts, boxes):
        coors = pts[:,:3]
        cens = (boxes[:,:3]+boxes[:,3:])/2
        r = (cens-boxes[:,:3]).square().sum(-1, keepdim=True).sqrt()

        coors = 4 * coors * r.unsqueeze(-1) + cens.unsqueeze(-1)
        result = torch.cat([coors, pts[:,3:]], dim=1)
        return result

    #Optimize the weights to get the seeds appropriate to recover gt
    def optimize(self,gt, fgts, boxes0, didx, pidx,  bsize, iternum):
        self.x_sp00 = self.wei2cens(self.wei, self.knns)
        self.x_sp0 = [self.x_sp00]
        self.x_sp = self.x_sp0
        self.x_sp = self.list_detach(self.x_sp)

        self.x_sn, boxes = self.get_info(self.x_sp, boxes0, didx, pidx)
        self.x_s = self.box_norm(self.x_sn, boxes)
        
        self.x_s0n, _ = self.get_info(self.x_p0, boxes0, didx, pidx)
        self.x_s0 = self.box_norm(self.x_s0n, boxes)

        fgts = self.box_norm(fgts.permute(0,2,1), boxes).permute(0,2,1)

        self.x_s = self.x_s * self.mask.permute(0,2,1)
        self.x_s0 = self.x_s0 * self.mask.permute(0,2,1)

        #for check
        self.igts = fgts
        self.x_is = self.x_s

        loss = 1*self.loss_inverse(fgts, self.x_s, bsize)

        maxbox = self.x_s0n[:,:3].max(-1)[0] #b*3
        minbox = self.x_s0n[:,:3].min(-1)[0]
        
        loss.backward()
        if (iternum+1) % 1 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def normalize(self, data):
        mindata = data.min(axis=0, keepdims=True)
        maxdata = data.max(axis=0, keepdims=True)

        cendata = (mindata+maxdata)/2

        r = np.sqrt(np.square(data-cendata).sum(-1)).max()

        ndata = 0.5*(data-cendata)/r

        return ndata
    def tensor_norm(self, data):
        coors, colors = data[:,:3], data[:,3:]
        mindata = coors.min(axis=-1, keepdims=True)[0]
        maxdata = coors.max(axis=-1, keepdims=True)[0]
        cendata = (mindata+maxdata)/2
        r = (coors-cendata).square().sum(1).sqrt().max()
        ndata = 0.5*(coors-cendata)/r
        ndata = torch.cat([ndata, colors], dim=1)
        return r, cendata, ndata
    def tensor_denorm(self, data, cendata,  r):
        coors, colors = data[:,:3], data[:,3:]
        result = coors*r + cendata
        result = torch.cat([result, colors], dim=1)
        return result

    def save_embed(self, path):
        torch.save(self.embed, path)
    def load_embed(self, path):
        self.embed = torch.load(path)

    def readgt(self, path):
        pcd = o3d.io.read_point_cloud(path)

        if np.array(pcd.colors).shape[0] == 0:
            colors = 255.0*np.ones_like(np.array(pcd.points))
            use_color = False
        else:
            colors = 255.0*np.array(pcd.colors)
            use_color = True

        points = np.concatenate([np.array(pcd.points), colors], axis=-1)#[idx]
        self.points = points
        self.gts = torch.tensor(points).float().cuda()
        self.gts = self.gts.unsqueeze(0)

        self.mask = torch.tensor([1, 1, 1, 255.0, 255.0, 255.0]).float().cuda().unsqueeze(0).unsqueeze(0)
        return use_color

    def farthest_point_sampling(self, points, k):
        """
        Simple implementation of the Farthest Point Sampling algorithm.
    
        :param points: numpy.ndarray, the point coordinates from the point cloud
        :param k: int, the number of centroids (sample points) to select
        :return: numpy.ndarray, the indices of the selected points
        """
        n_points = points.shape[0]
        indices = np.zeros(k, dtype=int)
        distances = np.full(n_points, np.inf)
        # Randomly select the first point
        indices[0] = np.random.randint(0, n_points)
        # Compute distances from the first point
        for i in range(1, k):
            dist = np.linalg.norm(points - points[indices[i-1]], axis=1)
            distances = np.minimum(distances, dist)
            indices[i] = np.argmax(distances)
        return indices


    def chamfer(self, inputs, points):
        dis1, dis2, _, _ = self.chamfer_dist(inputs, points)
        loss = torch.maximum((1e-10+dis1).sqrt().mean(), (1e-10+dis2).sqrt().mean())
        return loss

    def chamfer_p(self, inputs, points):
        icoors, icolors = inputs[...,:3], inputs[...,3:]
        ocoors, ocolors = points[...,:3], points[...,3:]
        dis1, dis2, idx1, idx2 = self.chamfer_3d(icoors, ocoors)
        #dis1, dis2, idx1, idx2 = self.chamfer_3d(icolors, ocolors)
        b_idx = torch.arange(inputs.shape[0]).unsqueeze(1)
        cdis1 = (1e-10+(icolors-ocolors[b_idx, idx1]).square().sum(-1))#.sqrt()
        cdis2 = (1e-10+(icolors[b_idx, idx2]-ocolors).square().sum(-1))#.sqrt()
        loss1 = torch.maximum((1e-10+dis1).mean(), (1e-10+dis2).mean())
        #loss2 = (cdis1.mean() + cdis2.mean())/2
        loss2 = torch.maximum(cdis1.mean(), cdis2.mean())

        loss = (loss1 + loss2)/2.0
        return  loss

    def merge(self, outputs, points):

        b_idx = torch.arange(points.shape[0]).unsqueeze(1)
        outputs = outputs.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        ocoors, ocolors = outputs[...,:3], outputs[...,3:]
        pcoors, pcolors = points[...,:3], points[...,3:]
        cdis1, cdis2, cidx1, cidx2 = self.chamfer_3d(ocolors, pcolors)

        ocolors2 = pcolors[b_idx, cidx1]
        w = 1.0
        newcolors = (1.0-w) * ocolors + w*ocolors2
        results = torch.cat([ocoors, newcolors], dim=-1)
  
        return results.permute(0, 2, 1)

    def fidelity(self, inputs, points):
        dis1, dis2, _, _ = self.chamfer_dist(inputs, points)
        loss = (1e-10+dis2).sqrt().mean()
        return loss

    def sample_batch_progressive(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Iterator[torch.Tensor]:
        samples = None
        for (
            model,
            diffusion,
            stage_num_points,
            stage_guidance_scale,
            stage_use_karras,
            stage_karras_steps,
            stage_sigma_min,
            stage_sigma_max,
            stage_s_churn,
            stage_key_filter,
        ) in zip(
            self.models,
            self.diffusions,
            self.num_points,
            self.guidance_scale,
            self.use_karras,
            self.karras_steps,
            self.sigma_min,
            self.sigma_max,
            self.s_churn,
            self.model_kwargs_key_filter,
        ):
            stage_model_kwargs = model_kwargs.copy()
            if stage_key_filter != "*":
                use_keys = set(stage_key_filter.split(","))
                stage_model_kwargs = {k: v for k, v in stage_model_kwargs.items() if k in use_keys}
            
            if samples is not None:
                stage_model_kwargs["low_res"] = samples
            if self.x_s is not None:

                stage_model_kwargs["low_res"] = self.x_s#.repeat([int(batch_size/self.x_s.shape[0]), 1, 1]).detach()
            if hasattr(model, "cached_model_kwargs"):
                stage_model_kwargs = model.cached_model_kwargs(batch_size, stage_model_kwargs)
            if not self.embed is None:
                stage_model_kwargs['oembed']=torch.cat([self.embed, torch.zeros([self.embed.shape[0], 1024-self.dim0, 256]).cuda()], dim=1)#self.embed.detach()

            sample_shape = (batch_size, 3 + len(self.aux_channels), stage_num_points)

            if stage_guidance_scale != 1 and stage_guidance_scale != 0:
                for k, v in stage_model_kwargs.items():#copy
                    stage_model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0)

            if stage_use_karras:
                x_s = []
                for i in range(3):
                    x_s.append(diffusion.q_sample(diffusion.scale_channels(self.x_s), torch.tensor(self.startidx-1, device=self.device)))
                x_s = torch.cat(x_s, dim=-1)
                #x_s = diffusion.scale_channels(x_s)
                #assert False
                samples_it = karras_sample_progressive(
                    diffusion=diffusion,
                    model=model,
                    #noise=x_s.repeat(batch_size, 1, 1),
                    shape=sample_shape,
                    steps=stage_karras_steps,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=stage_model_kwargs,
                    device=self.device,
                    sigma_min=stage_sigma_min,
                    sigma_max=stage_sigma_max,
                    s_churn=stage_s_churn,
                    guidance_scale=stage_guidance_scale,
                )
            else:
                internal_batch_size = batch_size
                if stage_guidance_scale:
                    model = self._uncond_guide_model(model, stage_guidance_scale)
                    internal_batch_size *= 2

                x_s = []
                for i in range(3):
                    x_s.append(diffusion.q_sample(diffusion.scale_channels(self.x_s.repeat(batch_size, 1, 1)), torch.tensor(self.startidx-1, device=self.device)))
                x_s = torch.cat(x_s, dim=-1)
                self.noise = x_s

                samples_it = diffusion.p_sample_loop_inter(
                    model,
                    shape=(internal_batch_size, *sample_shape[1:]),
                    noise=x_s,#.repeat(internal_batch_size, 1, 1),
                    startidx=self.startidx,
                    model_kwargs=stage_model_kwargs,
                    device=self.device,
                    clip_denoised=self.clip_denoised,
                )

            for x in samples_it:
                samples = x["pred_xstart"][:batch_size]

            return samples#/self.mask.permute(0,2,1)


    @classmethod
    def combine(cls, *samplers: "PointCloudSampler") -> "PointCloudSampler":
        assert all(x.device == samplers[0].device for x in samplers[1:])
        assert all(x.aux_channels == samplers[0].aux_channels for x in samplers[1:])
        assert all(x.clip_denoised == samplers[0].clip_denoised for x in samplers[1:])
        return cls(
            device=samplers[0].device,
            models=[x for y in samplers for x in y.models],
            diffusions=[x for y in samplers for x in y.diffusions],
            num_points=[x for y in samplers for x in y.num_points],
            aux_channels=samplers[0].aux_channels,
            model_kwargs_key_filter=[x for y in samplers for x in y.model_kwargs_key_filter],
            guidance_scale=[x for y in samplers for x in y.guidance_scale],
            clip_denoised=samplers[0].clip_denoised,
            use_karras=[x for y in samplers for x in y.use_karras],
            karras_steps=[x for y in samplers for x in y.karras_steps],
            sigma_min=[x for y in samplers for x in y.sigma_min],
            sigma_max=[x for y in samplers for x in y.sigma_max],
            s_churn=[x for y in samplers for x in y.s_churn],
        )

    def _uncond_guide_model(
        self, model: Callable[..., torch.Tensor], scale: float
    ) -> Callable[..., torch.Tensor]:
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        return model_fn

    def split_model_output(
        self,
        output: torch.Tensor,
        rescale_colors: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (
            len(self.aux_channels) + 3 == output.shape[1]
        ), "there must be three spatial channels before aux"
        pos, joined_aux = output[:, :3], output[:, 3:]

        aux = {}
        for i, name in enumerate(self.aux_channels):
            v = joined_aux[:, i]
            if name in {"R", "G", "B", "A"}:
                v = v.clamp(0, 255).round()
                if rescale_colors:
                    v = v / 255.0
            aux[name] = v
        return pos, aux

    def output_to_point_clouds(self, output: torch.Tensor) -> List[PointCloud]:
        res = []
        for sample in output:
            xyz, aux = self.split_model_output(sample[None], rescale_colors=True)
            res.append(
                PointCloud(
                    coords=xyz[0].t().cpu().numpy(),
                    channels={k: v[0].cpu().numpy() for k, v in aux.items()},
                )
            )
        return res

    def with_options(
        self,
        guidance_scale: float,
        clip_denoised: bool,
        use_karras: Sequence[bool] = (True, True),
        karras_steps: Sequence[int] = (64, 64),
        sigma_min: Sequence[float] = (1e-3, 1e-3),
        sigma_max: Sequence[float] = (120, 160),
        s_churn: Sequence[float] = (3, 0),
    ) -> "PointCloudSampler":
        return PointCloudSampler(
            device=self.device,
            models=self.models,
            diffusions=self.diffusions,
            num_points=self.num_points,
            aux_channels=self.aux_channels,
            model_kwargs_key_filter=self.model_kwargs_key_filter,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=s_churn,
        )
