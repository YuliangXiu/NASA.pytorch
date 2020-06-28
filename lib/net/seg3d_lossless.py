import torch
import torch.nn as nn
import torch.nn.functional as F

def plot_mask3D(
    mask=None, title="", point_coords=None, figsize=1500, point_marker_size=8,
    interactive=True):
    '''
    Simple plotting tool to show intermediate mask predictions and points 
    where PointRend is applied.

    Args:
    mask (Tensor): mask prediction of shape DxHxW
    title (str): title for the plot
    point_coords ((Tensor, Tensor, Tensor)): x and y and z point coordinates
    figsize (int): size of the figure to plot
    point_marker_size (int): marker size for points
    '''
    import trimesh
    import vtkplotter
    from skimage import measure

    vp = vtkplotter.Plotter(title=title, size=(figsize, figsize))
    vis_list = []

    if mask is not None:
        mask = mask.detach().to("cpu").numpy()
        mask = mask.transpose(2, 1, 0)
        
        # marching cube to find surface
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            mask, 0.5, gradient_direction='ascent')

        # create a mesh
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        vis_list.append(mesh)
    
    if point_coords is not None:
        point_coords = torch.stack(point_coords, 1).to("cpu").numpy()
        pc = vtkplotter.Points(point_coords, r=point_marker_size, c='red')
        vis_list.append(pc)
        
    vp.show(*vis_list, bg="white", axes=1, interactive=interactive, azimuth=30, elevation=30)

def create_grid3D(min, max, steps):
    if type(min) is int:
        min = (min, min, min) # (x, y, z)
    if type(max) is int:
        max = (max, max, max) # (x, y)
    if type(steps) is int:
        steps = (steps, steps, steps) # (x, y, z)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).long()
    gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH, gridD]) # [2, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t() # [N, 3]
    return coords

class SmoothConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2 

        weight = torch.ones(
            (in_channels, out_channels, kernel_size, kernel_size, kernel_size), 
            dtype=torch.float32
        ) / (kernel_size**3)
        self.register_buffer('weight', weight)
        
    def forward(self, input):
        return F.conv3d(input, self.weight, padding=self.padding)

class Seg3dLossless(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions,
                 channels=1, balance_value=0.5, align_corners=False, 
                 visualize=False, debug=False, use_cuda_impl=False, faster=False, 
                 use_shadow=False, **kwargs):
        """
        align_corners: same with how you process gt. (grid_sample / interpolate) 
        """
        super().__init__()
        self.query_func = query_func
        self.register_buffer('b_min', torch.tensor(b_min).float().unsqueeze(1)) #[bz, 1, 3]
        self.register_buffer('b_max', torch.tensor(b_max).float().unsqueeze(1)) #[bz, 1, 3]
        if type(resolutions[0]) is int:
            resolutions = torch.tensor([(res, res, res) for res in resolutions])
        else:
            resolutions = torch.tensor(resolutions)
        self.register_buffer('resolutions', resolutions)
        self.batchsize = self.b_min.size(0); assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels; assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize
        self.debug = debug
        self.use_cuda_impl = use_cuda_impl
        self.faster = faster
        self.use_shadow = use_shadow
        self.nsamples_max = 200000

        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        init_coords = create_grid3D(
            0, resolutions[-1]-1, steps=resolutions[0]) #[N, 3]
        init_coords = init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 3]
        self.register_buffer('init_coords', init_coords)

        # some useful tensors
        calculated = torch.zeros(
            (self.resolutions[-1][2], self.resolutions[-1][1], self.resolutions[-1][0]), 
            dtype=torch.bool).int()
        self.register_buffer('calculated', calculated)

        gird8_offsets = torch.stack(torch.meshgrid([
            torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
        ])).int().view(3, -1).t() #[27, 3]
        self.register_buffer('gird8_offsets', gird8_offsets)

        # smooth convs
        self.smooth_conv3x3 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=3)
        self.smooth_conv5x5 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=5)
        self.smooth_conv7x7 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=7)
        self.smooth_conv9x9 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=9)

        # cuda impl
        if self.use_cuda_impl:
            from implicit_seg.functional import Interp2xBoundary3d
            self.upsampler = Interp2xBoundary3d()

    def batch_eval(self, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
        coords = coords.detach()
        # normalize coords to fit in [b_min, b_max]
        if self.align_corners:
            coords2D = coords.float() / (self.resolutions[-1] - 1)
        else:
            step = 1.0 / self.resolutions[-1].float()
            coords2D = coords.float() / self.resolutions[-1] + step // 2
        coords2D = coords2D * (self.b_max - self.b_min) + self.b_min
        
        # query function
        occupancys = []
        nsamples = coords2D.size(1)
        for i in range(0, nsamples, self.nsamples_max):
            occupancys_split = self.query_func(
                **kwargs, points=coords2D[:, i:i+self.nsamples_max, :])
            if type(occupancys_split) is list:
                occupancys_split = torch.stack(occupancys_split) #[bz, C, N]
            occupancys.append(occupancys_split)
        occupancys = torch.cat(occupancys, dim=2)
        assert len(occupancys.size()) == 3, \
            "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def forward(self, **kwargs):
        if self.faster:
            return self._forward_faster(**kwargs)
        else:
            return self._forward(**kwargs)

    def _forward_faster(self, **kwargs):
        """
        In faster mode, we make following changes to exchange accuracy for speed:
        1. no conflict checking: 4.88 fps -> 6.56 fps
        2. smooth_conv9x9 ~ smooth_conv3x3 for different resolution
        3. last step no examine
        """
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]
        
        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            
            # first step
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)

                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)
                
                with torch.no_grad():
                    coords_accum = coords / stride

            # last step
            elif torch.equal(resolution, self.resolutions[-1]):
                if self.use_cuda_impl:
                    occupancys, is_boundary = self.upsampler(occupancys.contiguous())

                else:
                    with torch.no_grad():
                        # here true is correct!
                        valid = F.interpolate(
                            (occupancys>self.balance_value).float(), 
                            size=(D, H, W), mode="trilinear", align_corners=True)

                    # here true is correct!
                    occupancys = F.interpolate(
                        occupancys.float(), 
                        size=(D, H, W), mode="trilinear", align_corners=True)

                    is_boundary = (valid > 0.0) & (valid < 1.0)

            # next steps
            else:
                coords_accum *= 2

                if self.use_cuda_impl:
                    occupancys, is_boundary = self.upsampler(occupancys)

                else:
                    with torch.no_grad():
                        # here true is correct!
                        valid = F.interpolate(
                            (occupancys>self.balance_value).float(), 
                            size=(D, H, W), mode="trilinear", align_corners=True)

                    # here true is correct!
                    occupancys = F.interpolate(
                        occupancys.float(), 
                        size=(D, H, W), mode="trilinear", align_corners=True)

                    is_boundary = (valid > 0.0) & (valid < 1.0)
                
                with torch.no_grad():
                    if torch.equal(resolution, self.resolutions[1]):
                        is_boundary = (self.smooth_conv9x9(is_boundary.float()) > 0)[0, 0]
                    elif torch.equal(resolution, self.resolutions[2]):
                        is_boundary = (self.smooth_conv7x7(is_boundary.float()) > 0)[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                    is_boundary[coords_accum[0, :, 2],
                                coords_accum[0, :, 1], 
                                coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero().unsqueeze(0)
                    point_indices = (
                        point_coords[:, :, 2] * H * W + 
                        point_coords[:, :, 1] * W + 
                        point_coords[:, :, 0])

                    R, C, D, H, W = occupancys.shape
                    
                    # inferred value
                    coords = point_coords * stride

                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, D * H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, D, H, W)
                )

                with torch.no_grad():
                    voxels = coords / stride
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=1).unique(dim=1)
        
        return occupancys

    
    def _forward(self, **kwargs):
        """
        output occupancy field would be:
        (bz, C, res, res)
        """
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]

        calculated = self.calculated.clone() > 0
        
        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            if self.visualize:
                this_stage_coords = []
            
            # first step
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)

                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)
                
                with torch.no_grad():
                    coords_accum = coords / stride
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True
            
            # next steps
            else:
                coords_accum *= 2

                if self.use_cuda_impl:
                    occupancys, is_boundary = self.upsampler(occupancys)
                    
                else:
                    with torch.no_grad():
                        # here true is correct!
                        valid = F.interpolate(
                            (occupancys>self.balance_value).float(), 
                            size=(D, H, W), mode="trilinear", align_corners=True)

                    # here true is correct!
                    occupancys = F.interpolate(
                        occupancys.float(), 
                        size=(D, H, W), mode="trilinear", align_corners=True)

                    is_boundary = (valid > 0.0) & (valid < 1.0)
                
                with torch.no_grad():
                    # TODO
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        # larger z means smaller depth here
                        depth_res = resolution[2].item()
                        depth_index = torch.linspace(0, depth_res-1, steps=depth_res).to(occupancys.device)
                        depth_index_max = torch.max( 
                            (occupancys > self.balance_value) * (depth_index + 1), dim=-1, keepdim=True)[0] - 1
                        shadow = depth_index < depth_index_max
                        is_boundary[shadow] = False
                        is_boundary = is_boundary[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                        # is_boundary = is_boundary[0, 0]

                    is_boundary[coords_accum[0, :, 2],
                                coords_accum[0, :, 1], 
                                coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero().unsqueeze(0)
                    point_indices = (
                        point_coords[:, :, 2] * H * W + 
                        point_coords[:, :, 1] * W + 
                        point_coords[:, :, 0])

                    R, C, D, H, W = occupancys.shape
                    # interpolated value
                    occupancys_interp = torch.gather(
                        occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))

                    # inferred value
                    coords = point_coords * stride

                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                if self.visualize:
                    this_stage_coords.append(coords)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, D * H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, D, H, W)
                )

                with torch.no_grad():
                    # conflicts
                    conflicts = (
                        (occupancys_interp - self.balance_value) *
                        (occupancys_topk - self.balance_value) < 0
                    )[0, 0]
                    
                    if self.visualize:
                        self.plot(occupancys, coords, final_D, final_H, final_W)
                    
                    voxels = coords / stride
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=1).unique(dim=1)
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

                while conflicts.sum() > 0:
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        break
                    
                    with torch.no_grad():
                        conflicts_coords = coords[0, conflicts, :]

                        if self.debug:
                            self.plot(occupancys, conflicts_coords.unsqueeze(0), 
                                      final_D, final_H, final_W, title='conflicts')
                          
                        conflicts_boundary = (
                            conflicts_coords.int() +
                            self.gird8_offsets.unsqueeze(1) * stride.int()
                        ).reshape(-1, 3).long().unique(dim=0)
                        conflicts_boundary[:, 0] = (
                            conflicts_boundary[:, 0].clamp(0, calculated.size(2) - 1))
                        conflicts_boundary[:, 1] = (
                            conflicts_boundary[:, 1].clamp(0, calculated.size(1) - 1))
                        conflicts_boundary[:, 2] = (
                            conflicts_boundary[:, 2].clamp(0, calculated.size(0) - 1))

                        coords = conflicts_boundary[
                            calculated[conflicts_boundary[:, 2], 
                                    conflicts_boundary[:, 1], 
                                    conflicts_boundary[:, 0]] == False
                        ]

                        if self.debug:
                            self.plot(occupancys, coords.unsqueeze(0), 
                                      final_D, final_H, final_W, title='coords')
                            
                        coords = coords.unsqueeze(0)
                        point_coords = coords / stride
                        point_indices = (
                            point_coords[:, :, 2] * H * W + 
                            point_coords[:, :, 1] * W + 
                            point_coords[:, :, 0])
                        
                        R, C, D, H, W = occupancys.shape
                        # interpolated value
                        occupancys_interp = torch.gather(
                            occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))

                        # inferred value
                        coords = point_coords * stride

                    if coords.size(1) == 0:
                        break
                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    if self.visualize:
                        this_stage_coords.append(coords)

                    with torch.no_grad():
                        # conflicts
                        conflicts = (
                            (occupancys_interp - self.balance_value) *
                            (occupancys_topk - self.balance_value) < 0
                        )[0, 0]
                    
                    # put mask point predictions to the right places on the upsampled grid.
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    occupancys = (
                        occupancys.reshape(R, C, D * H * W)
                        .scatter_(2, point_indices, occupancys_topk)
                        .view(R, C, D, H, W)
                    )

                    with torch.no_grad():
                        voxels = coords / stride
                        coords_accum = torch.cat([
                            voxels, 
                            coords_accum
                        ], dim=1).unique(dim=1)
                        calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True
                
                if self.visualize:
                    this_stage_coords = torch.cat(this_stage_coords, dim=1)
                    self.plot(occupancys, this_stage_coords, final_D, final_H, final_W)
                    
        return occupancys

    def plot(self, occupancys, coords, final_D, final_H, final_W, title='', **kwargs):
        final = F.interpolate(
            occupancys.float(), size=(final_D, final_H, final_W), 
            mode="trilinear", align_corners=True) # here true is correct!
        x = coords[0, :, 0].to("cpu")
        y = coords[0, :, 1].to("cpu")
        z = coords[0, :, 2].to("cpu")
        
        plot_mask3D(
            final[0, 0].to("cpu"), title, (x, y, z), **kwargs)