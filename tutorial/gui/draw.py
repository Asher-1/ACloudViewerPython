#!/usr/bin/env python
import math
import numpy as np
import cloudViewer as cv3d
import cloudViewer.visualization as vis
import os
import random
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cloudViewer_tutorial as cv3dtut

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def normalize(v):
    a = 1.0 / math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (a * v[0], a * v[1], a * v[2])


def make_point_cloud(npts, center, radius, colorize):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = cv3d.geometry.ccPointCloud()
    cloud.set_points(cv3d.utility.Vector3dVector(pts))
    if colorize:
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.set_colors(cv3d.utility.Vector3dVector(colors))
    return cloud


def single_object():
    # No colors, no normals, should appear unlit black
    cube = cv3d.geometry.ccMesh.create_box(1, 2, 4)
    vis.draw(cube)


def multi_objects():
    pc_rad = 1.0
    pc_nocolor = make_point_cloud(100, (0, -2, 0), pc_rad, False)
    pc_color = make_point_cloud(100, (3, -2, 0), pc_rad, True)
    r = 0.4
    sphere_unlit = cv3d.geometry.ccMesh.create_sphere(r)
    sphere_unlit.translate((0, 1, 0))
    sphere_colored_unlit = cv3d.geometry.ccMesh.create_sphere(r)
    sphere_colored_unlit.paint_uniform_color((1.0, 0.0, 0.0))
    sphere_colored_unlit.translate((2, 1, 0))
    sphere_lit = cv3d.geometry.ccMesh.create_sphere(r)
    sphere_lit.compute_vertex_normals()
    sphere_lit.translate((4, 1, 0))
    sphere_colored_lit = cv3d.geometry.ccMesh.create_sphere(r)
    sphere_colored_lit.compute_vertex_normals()
    sphere_colored_lit.paint_uniform_color((0.0, 1.0, 0.0))
    sphere_colored_lit.translate((6, 1, 0))
    big_bbox = cv3d.geometry.ccBBox((-pc_rad, -3, -pc_rad), (6.0 + r, 1.0 + r, pc_rad))
    sphere_bbox = sphere_unlit.get_axis_aligned_bounding_box()
    sphere_bbox.set_color([1.0, 0.5, 0.0])
    lines = cv3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        sphere_lit.get_axis_aligned_bounding_box())
    lines_colored = cv3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        sphere_colored_lit.get_axis_aligned_bounding_box())
    lines_colored.paint_uniform_color((0.0, 0.0, 1.0))

    vis.draw([
        pc_nocolor, pc_color, sphere_unlit, sphere_colored_unlit, sphere_lit,
        sphere_colored_lit, big_bbox, sphere_bbox, lines, lines_colored
    ])


def actions():
    SOURCE_NAME = "Source"
    RESULT_NAME = "Result (Poisson reconstruction)"
    TRUTH_NAME = "Ground truth"
    bunny = cv3dtut.get_bunny_mesh()
    bunny.paint_uniform_color((1, 0.75, 0))
    bunny.compute_vertex_normals()
    cloud = cv3d.geometry.ccPointCloud()
    cloud.set_points(bunny.vertices)
    cloud.set_normals(bunny.vertex_normals)

    def make_mesh(o3dvis):
        # TODO: call o3dvis.get_geometry instead of using bunny
        mesh, _ = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(cloud)
        mesh.paint_uniform_color((1, 1, 1))
        mesh.compute_vertex_normals()
        o3dvis.add_geometry({"name": RESULT_NAME, "geometry": mesh})
        o3dvis.show_geometry(SOURCE_NAME, False)

    def toggle_result(o3dvis):
        truth_vis = o3dvis.get_geometry(TRUTH_NAME).is_visible
        o3dvis.show_geometry(TRUTH_NAME, not truth_vis)
        o3dvis.show_geometry(RESULT_NAME, truth_vis)

    vis.draw([{
        "name": SOURCE_NAME,
        "geometry": cloud
    }, {
        "name": TRUTH_NAME,
        "geometry": bunny,
        "is_visible": False
    }],
             actions=[("Create Mesh", make_mesh),
                      ("Toggle truth/result", toggle_result)])


def get_icp_transform(source, target, source_indices, target_indices):
    corr = np.zeros((len(source_indices), 2))
    corr[:, 0] = source_indices
    corr[:, 1] = target_indices

    # Estimate rough transformation using correspondences
    p2p = cv3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            cv3d.utility.Vector2iVector(corr))

    # Point-to-point ICP for refinement
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = cv3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        cv3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation


def selections():
    source = cv3d.io.read_point_cloud(CURRENT_DIR +
                                     "/../../TestData/ICP/cloud_bin_0.pcd")
    target = cv3d.io.read_point_cloud(CURRENT_DIR +
                                     "/../../TestData/ICP/cloud_bin_2.pcd")
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    source_name = "Source (yellow)"
    target_name = "Target (blue)"

    def do_icp_one_set(o3dvis):
        # sets: [name: [{ "index": int, "order": int, "point": (x, y, z)}, ...],
        #        ...]
        sets = o3dvis.get_selection_sets()
        source_picked = sorted(list(sets[0][source_name]),
                               key=lambda x: x.order)
        target_picked = sorted(list(sets[0][target_name]),
                               key=lambda x: x.order)
        source_indices = [idx.index for idx in source_picked]
        target_indices = [idx.index for idx in target_picked]

        t = get_icp_transform(source, target, source_indices, target_indices)
        source.transform(t)

        # Update the source geometry
        o3dvis.remove_geometry(source_name)
        o3dvis.add_geometry({"name": source_name, "geometry": source})

    def do_icp_two_sets(o3dvis):
        sets = o3dvis.get_selection_sets()
        source_set = sets[0][source_name]
        target_set = sets[1][target_name]
        source_picked = sorted(list(source_set), key=lambda x: x.order)
        target_picked = sorted(list(target_set), key=lambda x: x.order)
        source_indices = [idx.index for idx in source_picked]
        target_indices = [idx.index for idx in target_picked]

        t = get_icp_transform(source, target, source_indices, target_indices)
        source.transform(t)

        # Update the source geometry
        o3dvis.remove_geometry(source_name)
        o3dvis.add_geometry({"name": source_name, "geometry": source})

    vis.draw([{
        "name": source_name,
        "geometry": source
    }, {
        "name": target_name,
        "geometry": target
    }],
             actions=[("ICP Registration (one set)", do_icp_one_set),
                      ("ICP Registration (two sets)", do_icp_two_sets)],
             show_ui=True)


def time_animation():
    orig = make_point_cloud(200, (0, 0, 0), 1.0, True)
    clouds = [{"name": "t=0", "geometry": orig, "time": 0}]
    drift_dir = (1.0, 0.0, 0.0)
    expand = 1.0
    n = 20
    for i in range(1, n):
        amount = float(i) / float(n - 1)
        cloud = cv3d.geometry.ccPointCloud()
        pts = np.asarray(orig.points)
        pts = pts * (1.0 + amount * expand) + [amount * v for v in drift_dir]
        cloud.points = cv3d.utility.Vector3dVector(pts)
        cloud.colors = orig.colors
        clouds.append({
            "name": "points at t=" + str(i),
            "geometry": cloud,
            "time": i
        })

    vis.draw(clouds)


def groups():
    building_mat = vis.rendering.Material()
    building_mat.shader = "defaultLit"
    building_mat.base_color = (1.0, .90, .75, 1.0)
    building_mat.base_reflectance = 0.1
    midrise_mat = vis.rendering.Material()
    midrise_mat.shader = "defaultLit"
    midrise_mat.base_color = (.475, .450, .425, 1.0)
    midrise_mat.base_reflectance = 0.1
    skyscraper_mat = vis.rendering.Material()
    skyscraper_mat.shader = "defaultLit"
    skyscraper_mat.base_color = (.05, .20, .55, 1.0)
    skyscraper_mat.base_reflectance = 0.9
    skyscraper_mat.base_roughness = 0.01

    buildings = []
    size = 10.0
    half = size / 2.0
    min_height = 1.0
    max_height = 20.0
    for z in range(0, 10):
        for x in range(0, 10):
            max_h = max_height * (1.0 - abs(half - x) / half) * (
                1.0 - abs(half - z) / half)
            h = random.uniform(min_height, max(max_h, min_height + 1.0))
            box = cv3d.geometry.ccMesh.create_box(0.9, h, 0.9)
            box.compute_triangle_normals()
            box.translate((x + 0.05, 0.0, z + 0.05))
            if h > 0.333 * max_height:
                mat = skyscraper_mat
            elif h > 0.1 * max_height:
                mat = midrise_mat
            else:
                mat = building_mat
            buildings.append({
                "name": "building_" + str(x) + "_" + str(z),
                "geometry": box,
                "material": mat,
                "group": "buildings"
            })

    haze = make_point_cloud(5000, (half, 0.333 * max_height, half),
                            1.414 * half, False)
    haze.paint_uniform_color((0.8, 0.8, 0.8))

    smog = make_point_cloud(10000, (half, 0.25 * max_height, half), 1.2 * half,
                            False)
    smog.paint_uniform_color((0.95, 0.85, 0.75))

    vis.draw(buildings + [{
        "name": "haze",
        "geometry": haze,
        "group": "haze"
    }, {
        "name": "smog",
        "geometry": smog,
        "group": "smog"
    }])


def remove():

    def make_sphere(name, center, color, group, time):
        sphere = cv3d.geometry.ccMesh.create_sphere(0.5)
        sphere.compute_vertex_normals()
        sphere.translate(center)

        mat = vis.rendering.Material()
        mat.shader = "defaultLit"
        mat.base_color = color

        return {
            "name": name,
            "geometry": sphere,
            "material": mat,
            "group": group,
            "time": time
        }

    red = make_sphere("red", (0, 0, 0), (1.0, 0.0, 0.0, 1.0), "spheres", 0)
    green = make_sphere("green", (2, 0, 0), (0.0, 1.0, 0.0, 1.0), "spheres", 0)
    blue = make_sphere("blue", (4, 0, 0), (0.0, 0.0, 1.0, 1.0), "spheres", 0)
    yellow = make_sphere("yellow", (0, 0, 0), (1.0, 1.0, 0.0, 1.0), "spheres",
                         1)
    bbox = {
        "name": "bbox",
        "geometry": red["geometry"].get_axis_aligned_bounding_box()
    }

    def remove_green(visdraw):
        visdraw.remove_geometry("green")

    def remove_yellow(visdraw):
        visdraw.remove_geometry("yellow")

    def remove_bbox(visdraw):
        visdraw.remove_geometry("bbox")

    vis.draw([red, green, blue, yellow, bbox],
             actions=[("Remove Green", remove_green),
                      ("Remove Yellow", remove_yellow),
                      ("Remove Bounds", remove_bbox)])


def main():
    single_object()
    multi_objects()
    actions()
    selections()


if __name__ == "__main__":
    main()
