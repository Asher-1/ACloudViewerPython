# CloudViewer: www.erow.cn
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import sys
import time
import cloudViewer as cv3d
import argparse

sys.path.append("../utility")
from file import *

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_path',
        type=str,
        help='path to the dataset.'
             'It should contain 16bit depth images in a folder named depth/'
             'and rgb images in a folder named color/ or rgb/')
    parser.add_argument('trajectory_path',
                        type=str,
                        help='path to the trajectory in open3d\'s .log format')
    parser.add_argument('--mesh_name',
                        type=str,
                        default='mesh.ply',
                        help='output mesh filename')
    parser.add_argument('--intrinsic_path',
                        type=str,
                        help='path to the intrinsic.json config file.'
                             'By default PrimeSense intrinsics is used.')
    parser.add_argument(
        '--block_count',
        type=int,
        default=100,
        help='estimated number of 16x16x16 voxel blocks to represent a scene.'
             'Typically with a 6mm resolution,'
             'a lounge scene requires around 30K blocks,'
             'while a large apartment requires 80K blocks.'
             'Open3D will dynamically increase the block count on demand,'
             'but a rough upper bound will be useful especially when memory is limited.'
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=3.0 / 512,
        help='voxel resolution.'
             'For small scenes, 6mm preserves fine details.'
             'For large indoor scenes, 1cm or larger will be reasonable for limited memory.'
    )
    parser.add_argument(
        '--depth_scale',
        type=float,
        default=1000.0,
        help='depth factor. Converting from a uint16 depth image to meter.')
    parser.add_argument('--max_depth',
                        type=float,
                        default=3.0,
                        help='max range in the scene to integrate.')
    parser.add_argument('--sdf_trunc',
                        type=float,
                        default=0.04,
                        help='SDF truncation threshold.')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)

    device = cv3d.core.Device(args.device)

    # Load RGBD
    [color_files, depth_files] = get_rgbd_file_lists(args.dataset_path)

    # Load intrinsics
    if args.intrinsic_path is None:
        intrinsic = cv3d.camera.PinholeCameraIntrinsic(
            cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = cv3d.io.read_pinhole_camera_intrinsic(args.intrinsic_path)

    intrinsic = cv3d.core.Tensor(intrinsic.intrinsic_matrix,
                                 cv3d.core.Dtype.Float32, device)

    # Load extrinsics
    trajectory = read_poses_from_log(args.trajectory_path)

    n_files = len(color_files)

    # Setup volume
    volume = cv3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': cv3d.core.Dtype.Float32,
            'weight': cv3d.core.Dtype.UInt16,
            'color': cv3d.core.Dtype.UInt16
        },
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        block_resolution=16,
        block_count=args.block_count,
        device=device)

    # For cblas-enabled Numpy, calling np.linalg.inv inside the for loop can
    # slow down subsequent computations. This needs to be further investigated.
    # Check `np.show_config()` and make sure MKL runtime `mkl_rt` is enabled.
    extrinsics = [np.linalg.inv(trajectory[i]) for i in range(n_files)]

    for i in range(n_files):
        rgb = cv3d.io.read_image(color_files[i])
        rgb = cv3d.t.geometry.Image.from_legacy_image(rgb, device=device)

        depth = cv3d.io.read_image(depth_files[i])
        depth = cv3d.t.geometry.Image.from_legacy_image(depth, device=device)

        extrinsic = cv3d.core.Tensor(extrinsics[i], cv3d.core.Dtype.Float32,
                                     device)

        start = time.time()
        volume.integrate(depth, rgb, intrinsic, extrinsic, args.depth_scale,
                         args.max_depth)
        end = time.time()
        print('Integration {:04d}/{:04d} takes {:.3f} ms'.format(
            i, n_files, (end - start) * 1000.0))

    mesh = volume.extract_surface_mesh().to_legacy_triangle_mesh()
    cv3d.io.write_triangle_mesh(args.mesh_name, mesh, False, True)
