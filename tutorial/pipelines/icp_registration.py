# CloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Basic/icp_registration.py

import cloudViewer as cv3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    cv3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    source = cv3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    target = cv3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = cv3d.pipelines.registration.evaluate_registration(source, target,
                                                                   threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = cv3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        cv3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = cv3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        cv3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
