# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Misc/meshes.py

import numpy as np
import cloudViewer as cv3d
import os
import urllib.request
import gzip
import tarfile
import shutil
import time


def edges_to_lineset(mesh, edges, color):
    ls = cv3d.geometry.LineSet()
    ls.points = mesh.get_vertices()
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = cv3d.utility.Vector3dVector(colors)
    return ls


def apply_noise(mesh, noise):
    vertices = np.asarray(mesh.get_vertices())
    vertices += np.random.uniform(-noise, noise, size=vertices.shape)
    mesh.set_vertices(cv3d.utility.Vector3dVector(vertices))
    return mesh


def triangle():
    mesh = cv3d.geometry.ccMesh(
        vertices=cv3d.utility.Vector3dVector(
            np.array(
                [
                    (np.sqrt(8 / 9), 0, -1 / 3),
                    (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                    (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3),
                ],
                dtype=np.float32,
            )),
        triangles=cv3d.utility.Vector3iVector(np.array([[0, 1, 2]])),
    )
    mesh.compute_vertex_normals()
    return mesh


def plane(height=0.2, width=1):
    mesh = cv3d.geometry.ccMesh(
        vertices=cv3d.utility.Vector3dVector(
            np.array(
                [[0, 0, 0], [0, height, 0], [width, height, 0], [width, 0, 0]],
                dtype=np.float32,
            )),
        triangles=cv3d.utility.Vector3iVector(np.array([[0, 2, 1], [2, 0, 3]])),
    )
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_edge():
    verts = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
                     dtype=np.float64)
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = cv3d.geometry.ccMesh()
    mesh.create_internal_cloud()
    mesh.set_vertices(cv3d.utility.Vector3dVector(verts))
    mesh.set_triangles(cv3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_vertex():
    verts = np.array(
        [
            [-1, 0, -1],
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, 0],
            [-1, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    triangles = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
        [4, 5, 6],
        [4, 5, 3],
        [5, 6, 3],
        [4, 6, 3],
    ])
    mesh = cv3d.geometry.ccMesh()
    mesh.create_internal_cloud()
    mesh.set_vertices(cv3d.utility.Vector3dVector(verts))
    mesh.set_triangles(cv3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    return mesh


def open_box():
    mesh = cv3d.geometry.ccMesh.create_box()
    mesh.set_triangles(cv3d.utility.Vector3iVector(np.asarray(mesh.get_triangles())[:-2]))
    mesh.compute_vertex_normals()
    return mesh


def intersecting_boxes():
    mesh0 = cv3d.geometry.ccMesh.create_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = cv3d.geometry.ccMesh.create_box()
    mesh1.transform(T)
    mesh = mesh0 + mesh1
    mesh.compute_vertex_normals()
    return mesh


def _relative_path(path):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, path)


def knot():
    mesh = cv3d.io.read_triangle_mesh(_relative_path("../../test_data/knot.ply"))
    mesh.compute_vertex_normals()
    return mesh


def bathtub():
    mesh = cv3d.io.read_triangle_mesh(
        _relative_path("../../test_data/bathtub_0154.ply"))
    mesh.compute_vertex_normals()
    return mesh


def armadillo():
    armadillo_path = _relative_path("../../test_data/Armadillo.ply")
    if not os.path.exists(armadillo_path):
        print("downloading armadillo mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
        urllib.request.urlretrieve(url, armadillo_path + ".gz")
        print("extract armadillo mesh")
        with gzip.open(armadillo_path + ".gz", "rb") as fin:
            with open(armadillo_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + ".gz")
    mesh = cv3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh


def MonkeyPath():
    monkey_path = _relative_path("../../test_data/monkey")
    if not os.path.exists(monkey_path):
        print("downloading monkey mesh")
        url = "https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/MonkeyModel.zip"
        output_path_name = os.path.dirname(monkey_path)
        urllib.request.urlretrieve(url,  output_path_name + ".zip")
        print("extract monkey mesh")
        with gzip.open(output_path_name + ".zip", "rb") as fin:
            with open(monkey_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(output_path_name + ".zip")
    return monkey_path

def monkey():
    monkey_path = MonkeyPath()
    mesh = cv3d.io.read_triangle_mesh(os.path.join(monkey_path, "monkey.obj"))
    mesh.compute_vertex_normals()
    return mesh

def bunny():
    bunny_path = _relative_path("../../test_data/Bunny.ply")
    if not os.path.exists(bunny_path):
        print("downloading bunny mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        urllib.request.urlretrieve(url, bunny_path + ".tar.gz")
        print("extract bunny mesh")
        with tarfile.open(bunny_path + ".tar.gz") as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(os.path.dirname(bunny_path), "bunny", "reconstruction",
                         "bun_zipper.ply"),
            bunny_path,
        )
        os.remove(bunny_path + ".tar.gz")
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), "bunny"))
    mesh = cv3d.io.read_triangle_mesh(bunny_path)
    mesh.compute_vertex_normals()
    return mesh


def eagle():
    path = _relative_path("../../test_data/eagle.ply")
    if not os.path.exists(path):
        print("downloading eagle pcl")
        url = "http://www.cs.jhu.edu/~misha/Code/PoissonRecon/eagle.points.ply"
        urllib.request.urlretrieve(url, path)
    pcd = cv3d.io.read_point_cloud(path)
    return pcd


def center_and_scale(mesh):
    vertices = np.asarray(mesh.get_vertices())
    vertices = vertices / max(vertices.max(axis=0) - vertices.min(axis=0))
    vertices -= vertices.mean(axis=0)
    mesh.set_vertices(cv3d.utility.Vector3dVector(vertices))
    return mesh


def print_1D_array_for_cpp(prefix, array):
    if array.dtype == np.float32:
        dtype = "float"
    elif array.dtype == np.float64:
        dtype = "double"
    elif array.dtype == np.int32:
        dtype = "int"
    elif array.dtype == np.uint32:
        dtype = "size_t"
    elif array.dtype == np.bool:
        dtype = "bool"
    else:
        raise Exception("invalid dtype")
    print(f"std::vector<{dtype}> {prefix} = {{")
    print(", ".join(map(str, array)))
    print("};")


def print_2D_array_for_cpp(prefix, values, fmt):
    if values.shape[0] > 0:
        print(f"{prefix} = {{")
        print(",\n".join([
            f"  {{{v[0]:{fmt}}, {v[1]:{fmt}}, {v[2]:{fmt}}}}" for v in values
        ]))
        print(f"}};")


def print_mesh_for_cpp(mesh, prefix=""):
    print_2D_array_for_cpp(f"{prefix}vertices_", np.asarray(mesh.get_vertices()),
                           ".6f")
    print_2D_array_for_cpp(f"{prefix}vertex_normals_",
                           np.asarray(mesh.get_vertex_normals()), ".6f")
    print_2D_array_for_cpp(f"{prefix}vertex_colors_",
                           np.asarray(mesh.get_vertex_colors()), ".6f")
    print_2D_array_for_cpp(f"{prefix}triangles_", np.asarray(mesh.get_triangles()),
                           "d")
    print_2D_array_for_cpp(f"{prefix}triangle_normals_",
                           np.asarray(mesh.get_triangle_normals()), ".6f")
