import torch
import torch.nn as nn
from .vfe_template import VFETemplate


def xy_to_idx(xy: torch.tensor, n_grids) -> torch.tensor:
    """
    xy: [points, (x,y)]
    n_grids: [x,y]
    """

    idx = torch.zeros(xy.shape[0], dtype=torch.int32, device=xy.device)
    idx = xy[:, 1] + xy[:, 0] * n_grids[1]
    return idx


def idx_to_xy(idx, n_grids):
    """
    Converts a linear index (idx) to 3D coordinates (xyz) using PyTorch.

    Args:
        idx (torch.Tensor): The linear index or tensor of linear indices.
        n_grids (torch.Tensor): A tensor representing the maximum dimensions (n_grids, ymax, zmax).

    Returns:
        torch.Tensor: A tensor containing the 3D coordinates (x, y, z).
    """

    x = idx // (n_grids[1])
    y = idx % (n_grids[1])

    return torch.stack((x, y), dim=-1)

def xyz_to_idx(xyz: torch.tensor, n_grids) -> torch.tensor:
    """
    xyz: [points, (x,y,z)]
    n_grids: [x,y,z]
    """
    idx = torch.zeros(xyz.shape[0], dtype=torch.int64, device=xyz.device)
    idx = xyz[:, 2] + n_grids[2] * (xyz[:, 1] + xyz[:, 0] * n_grids[1])
    return idx

def idx_to_xyz(idx, n_grids):
    """
    Converts a linear index (idx) to 3D coordinates (xyz) using PyTorch.

    Args:
        idx (torch.Tensor): The linear index or tensor of linear indices.
        n_grids (torch.Tensor): A tensor representing the maximum dimensions (n_grids, ymax, zmax).

    Returns:
        torch.Tensor: A tensor containing the 3D coordinates (x, y, z).
    """
    x = idx // (n_grids[2] * n_grids[1])
    y = (idx % (n_grids[2] * n_grids[1])) // n_grids[2]
    z = idx % n_grids[2]

    return torch.stack((x, y, z), dim=-1)

class PillarHist(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_filters = self.model_cfg.NUM_FILTERS
        self.point_cloud_range = torch.nn.Parameter(torch.tensor(point_cloud_range),requires_grad=False)
        self.min_range = torch.nn.Parameter(torch.tensor(point_cloud_range[:3]),requires_grad=False)
        self.voxel_size = torch.nn.Parameter(torch.tensor(voxel_size),requires_grad=False)
        n_grids = torch.round((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size)
        self.n_grids = torch.nn.Parameter(n_grids.int(), requires_grad=False)
        
        self.USE_INTENSITY = False

        num_pillar_feat = self.n_grids[2]
        if self.USE_INTENSITY:
            num_pillar_feat *= 2
        
        self.use_xy = False
        if self.use_xy:
            num_pillar_feat += 2

        self.linear1 = torch.nn.Linear(num_pillar_feat, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = nn.ReLU()


        """
        x(forward)
        y(left)
        z(up)
        """



    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def pillar_mlp(self, x, coords):
        x_lin = x.reshape(x.size(0), -1)

        if self.use_xy:
            x_lin = torch.cat([x_lin, coords], dim=1)
        x1 = self.linear1(x_lin)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        return x1
    
    def create_hist(self, points):
        pnts_coord = points[:, [0,1,2]] # zyx to xyz
        pnts_intensity = points[:, 3]

        NORMALIZE_INTENSITY = False
        if NORMALIZE_INTENSITY:
            pnts_intensity = pnts_intensity * 2.0 - 1.0


        if False:
            visualize_points(pnts_coord.cpu().numpy(), x_range=[-25,25], y_range=[-10,10], voxel_resolutions=[1.0, 1.0, 1.0], colors = pnts_intensity)

        voxel_coords = ((pnts_coord - self.min_range) // self.voxel_size).to(torch.int32)
        
        vox_idxs = xyz_to_idx(voxel_coords, self.n_grids)
        voxel_coords_rev = idx_to_xyz(vox_idxs, self.n_grids)
        diff = voxel_coords != voxel_coords_rev
        num_diff = torch.sum(diff)
        if num_diff !=0:
            print("num_diff=", num_diff)
            print("voxel_coords=", torch.max(voxel_coords, dim=0))

        uni_vox_idxs, uni_vox_inv_idxs, uni_vox_counts = torch.unique(
            vox_idxs, return_inverse=True, return_counts=True
        )

        USE_MAX_INTENSITY = False
        if USE_MAX_INTENSITY:
            ave_intensity = torch.zeros(
                len(uni_vox_idxs), device=pnts_intensity.device, dtype=pnts_intensity.dtype
            ).scatter_reduce_(0, uni_vox_inv_idxs, pnts_intensity, reduce="amax", include_self=False)
        else:
            total_intensity = torch.zeros(
                len(uni_vox_idxs), device=pnts_intensity.device, dtype=pnts_intensity.dtype
            ).scatter_reduce_(0, uni_vox_inv_idxs, pnts_intensity, reduce="sum", include_self=False)
            ave_intensity = total_intensity / uni_vox_counts


        bev_coords = uni_vox_idxs // self.n_grids[2]
        height_coords = uni_vox_idxs % self.n_grids[2]

        xyz_coords = idx_to_xyz(uni_vox_idxs, self.n_grids)
        bev_idxs = xy_to_idx(xyz_coords[:, :2], self.n_grids[:2])

        uni_bev_idxs, uni_bev_inv_idxs, _ = torch.unique(
            bev_idxs, sorted=True, return_inverse=True, return_counts=True
        )

        pillar_hist_idxs = self.n_grids[2] * uni_bev_inv_idxs + height_coords
        pillar_hist_idxs = pillar_hist_idxs.long()

        MAX_INTENSITY = 1.0
        if MAX_INTENSITY != 1.0:
            ave_intensity = torch.clamp(ave_intensity, max=MAX_INTENSITY)

        pillar_hist_intensity = torch.zeros(
            len(uni_bev_idxs) * self.n_grids[2], device=pillar_hist_idxs.device, dtype=ave_intensity.dtype
        ).scatter_(0, pillar_hist_idxs, ave_intensity)


        MAX_PNT_COUNTS = 1
        if MAX_PNT_COUNTS is None:
            vox_counts = uni_vox_counts.float()
        else:
            # vox_counts = 1 # for smaller memory footprint computation?
            vox_counts = torch.clamp(uni_vox_counts, max=MAX_PNT_COUNTS).float()
            vox_counts /= MAX_PNT_COUNTS

        pillar_hist_counts = torch.zeros(len(uni_bev_idxs) * self.n_grids[2], device=pillar_hist_idxs.device).scatter_(0,  pillar_hist_idxs, vox_counts)
        
        if self.USE_INTENSITY:
            pillar_hist = torch.stack(
                [
                    pillar_hist_counts.reshape(-1, self.n_grids[2]),
                    pillar_hist_intensity.reshape(-1, self.n_grids[2]),
                ],
                dim=1,
            )
        else:
            pillar_hist = torch.stack(
                [
                    pillar_hist_counts.reshape(-1, self.n_grids[2]),
                ],
                dim=1,
            )


        uni_xy_points = idx_to_xy(uni_bev_idxs, self.n_grids[:2]).float()
        NORMALIZE_XY_POINTS = True
        if NORMALIZE_XY_POINTS:
            uni_xy_points = uni_xy_points / self.n_grids[:2]
            uni_xy_points = (uni_xy_points  - 0.5) * 2.0

            NON_LINEAR_XY_POINTS = False
            if NON_LINEAR_XY_POINTS:
                transformed_coords = torch.tanh(uni_xy_points[:, :] * 2)
                transformed_coords = (transformed_coords + 1.0) / 2.0 * self.n_grids[:2]
                print(torch.max(transformed_coords, dim=0).values, torch.min(transformed_coords, dim=0).values)

                if False:
                    print(torch.max(uni_xy_points, dim=0).values, torch.min(uni_xy_points, dim=0).values)
                    print(torch.max(transformed_coords, dim=0).values, torch.min(transformed_coords, dim=0).values)
                    plt.figure(figsize=(10, 5))
                    plt.scatter(uni_xy_points[:, 1].cpu(), transformed_coords[:, 1].cpu(), label="Transformed")
                    plt.title("Logarithmic XY Coordinate Transformation")
                    plt.xlabel("original Coordinate")
                    plt.ylabel("compressed Coordinate")
                    plt.legend()
                    plt.grid(True)
                    plt.show()


        pillar_feat = self.pillar_mlp(pillar_hist, uni_xy_points)

        if False:
            max_vals = torch.max(pillar_feat, dim=1).values
            topk = torch.topk(max_vals, 3200, largest=True)
            indices = topk.indices[::100]

            cmap = plt.cm.get_cmap('coolwarm')

            import numpy as np

            pillar_feat_vis = pillar_feat[indices, :]
            X, Y = np.meshgrid(np.arange(len(indices)), np.arange(pillar_feat.size(1)))
            z = pillar_feat_vis.cpu().numpy().flatten()
            z = np.abs(z)
            colors = cmap(z / np.max(z))
            fig1 = plt.figure()
            ax = fig1.add_subplot(111, projection='3d')
            ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(z), 1, 1, z, color=colors)
            ax.set_xlabel('Pillars')
            ax.set_ylabel('Channels')
            ax.set_zlabel('Absolute Input Activation Value')


            pillar_hist_counts = pillar_hist_counts.reshape(-1, self.n_grids[2])
            pillar_feat_vis = pillar_hist_counts[indices, :]
            X, Y = np.meshgrid(np.arange(len(indices)), np.arange(pillar_feat_vis.size(1)))
            pillar_feat_vis = pillar_feat_vis.t()
            z = pillar_feat_vis.cpu().numpy().flatten()
            z = np.abs(z)
            colors = cmap(z / np.max(z))
            fig2 = plt.figure()
            ax = fig2.add_subplot(111, projection='3d')
            ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(z), 1, 1, z, color=colors)
            ax.set_xlabel('Pillars')
            ax.set_ylabel('Height Bins')
            ax.set_zlabel('Number of Points')
            plt.show()

        if False:
            valid_idx = torch.nonzero(pillar_hist_counts, as_tuple=False).squeeze(1)

            valid_pillar_idx = valid_idx // self.n_grids[2]
            valid_height_idx = valid_idx % self.n_grids[2]

            valid_bev_coords = idx_to_xy(uni_bev_idxs[valid_pillar_idx], self.n_grids[:2])

            # print(valid_bev_coords.shape, valid_height_idx.shape)

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            xyz = (
                torch.stack([valid_bev_coords[:, 0], valid_bev_coords[:, 1], valid_height_idx], dim=1)
                * self.voxel_size
                + self.min_range
            )

            xyz = (xyz).cpu().numpy()

            # visualize_points(xyz, x_range=[-75,75], y_range=[-75,75], voxel_resolutions=[0.1, 0.1, 0.1])

            colors = pillar_hist_intensity[valid_idx].cpu()
            # colors = pillar_hist_counts[valid_idx].cpu() * 10
            print(torch.max(colors), torch.min(colors))
            # colors = xyz[:, 2]

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, cmap="plasma")
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")

            ax.set_box_aspect([8, 8, 1])

            plt.show()

        canvas = torch.zeros(
            [self.n_grids[0] * self.n_grids[1], pillar_feat.size(1)],
            device=pillar_feat.device,
            dtype=pillar_feat.dtype,
        )

        if False:
            bev_coords = idx_to_xy(bev_idxs, self.n_grids[:2])
            print("-------------------")
            print("pnts_coord=", torch.max(pnts_coord, dim=0).values, torch.min(pnts_coord, dim=0).values)
            print("voxel_coords=", torch.max(voxel_coords, dim=0).values)
            print("xyz_coords=", torch.max(xyz_coords, dim=0).values)
            print("bev_coords=", torch.max(bev_coords, dim=0).values)

        canvas[uni_bev_idxs.long(), :] = pillar_feat
        canvas = canvas.t()
        canvas = canvas.reshape(pillar_feat.size(1), self.n_grids[0], self.n_grids[1])
        canvas = canvas.permute(0,2,1)
        

        return canvas


    def forward(self, batch_dict, **kwargs):
        points = batch_dict["points"]
        batch_spatial_features = []
        batch_size = points[:, 0].max().int().item() + 1


        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            batch_spatial_features.append(self.create_hist(points[batch_mask, 1:]))

        batch_dict['spatial_features'] = torch.stack(batch_spatial_features, 0)

        return batch_dict

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 2

X_RANGE = [-51.2, 51.2]
Y_RANGE = [-51.2, 51.2]
Z_RANGE = [-5, 3]
VOXEL_RESOLUTIONS = np.array([0.1, 0.1, 0.2])


def get_prallel_lines(x_range, y_range, reolution, is_vertical=False):
    intervals = np.arange(x_range[0], x_range[1] + reolution, reolution)
    intervals = np.repeat(intervals, NUM_DIMENSIONS)
    min_max = np.full_like(intervals, y_range[0])
    min_max[::NUM_DIMENSIONS] = y_range[1]
    heights = np.zeros_like(intervals) - 5
    parallel_lines = np.vstack((intervals, min_max, heights)).T

    parallel_lines = parallel_lines.astype(np.float64)
    if is_vertical:
        parallel_lines[:, [1, 0, 2]] = parallel_lines.copy()

    return parallel_lines


def gen_grid(x_range, y_range, raster_resolution: np.ndarray) -> o3d.geometry.LineSet:

    horizontal_lines = get_prallel_lines(
        x_range, y_range, raster_resolution[0], is_vertical=False
    )
    vertical_lines = get_prallel_lines(
        y_range, x_range, raster_resolution[1], is_vertical=True
    )

    line_set = o3d.geometry.LineSet()
    points = np.vstack((vertical_lines, horizontal_lines))
    # points = points.astype(np.float64)

    o3d.utility.Vector3dVector(points)
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = np.arange(points.shape[0]).reshape(-1, NUM_DIMENSIONS)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color((0.5, 0.5, 0.5))

    o3d.visualization.RenderOption.line_width = 0.01

    return line_set


def visualize_points(
    points: np.ndarray, x_range, y_range, voxel_resolutions, colors
) -> None:
    # if points.shape[1] == 2:
    #     z = np.zeros(points.shape[0])
    #     points = np.stack([points[:, 0], points[:, 1], z], axis=1)

    # colors = colors.cpu().numpy().repeat(-1, 3)
    colors /= torch.max(colors)
    colors = colors.cpu().numpy()
    cmap = plt.get_cmap("viridis")
    # colors = np.stack([0.5 - colors,  colors - 0.5, colors - 0.5], axis=1)
    colors = cmap(colors)[:, :3]

    points = points.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    line_set_grid = gen_grid(x_range, y_range, raster_resolution=voxel_resolutions[:2])
    o3d.visualization.draw_geometries(
        [pcd, line_set_grid],
        # [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )