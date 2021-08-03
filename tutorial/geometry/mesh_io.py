# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh_io.py

import numpy as np
import cloudViewer as cv3d
import os

import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import meshes

if __name__ == "__main__":
    mesh = meshes.bunny()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.get_triangles())
    vertices = np.asarray(mesh.get_vertices())
    vertex_normals = np.asarray(mesh.get_vertex_normals())
    n_vertices = vertices.shape[0]
    vertex_colors = np.random.uniform(0, 1, size=(n_vertices, 3))
    mesh.set_vertex_colors(cv3d.utility.Vector3dVector(vertex_colors))


    def test_float_array(array_a, array_b, eps=1e-6):
        diff = array_a - array_b
        dist = np.linalg.norm(diff, axis=1)
        return np.all(dist < eps)


    def test_int_array(array_a, array_b):
        diff = array_a - array_b
        return np.all(diff == 0)


    def compare_mesh(mesh):
        success = True
        if not test_float_array(vertices, np.asarray(mesh.get_vertices())):
            success = False
            print('[WARNING] vertices are not the same')
        if not test_float_array(vertex_normals, np.asarray(mesh.get_vertex_normals())):
            success = False
            print('[WARNING] vertex_normals are not the same')
        if not test_float_array(vertex_colors, np.asarray(mesh.get_vertex_colors()), eps=1e-2):
            success = False
            print('[WARNING] vertex_colors are not the same')
        if not test_int_array(triangles, np.asarray(mesh.get_triangles())):
            success = False
            print('[WARNING] triangles are not the same')
        if success:
            print('[INFO] written and read mesh are equal')


    print('Write ply file')
    cv3d.io.write_triangle_mesh('tmp.ply', mesh)
    print('Read ply file')
    mesh_test = cv3d.io.read_triangle_mesh('tmp.ply')
    compare_mesh(mesh_test)
    os.remove('tmp.ply')

    print('Write obj file')
    cv3d.io.write_triangle_mesh('tmp.obj', mesh)
    print('Read obj file')
    mesh_test = cv3d.io.read_triangle_mesh('tmp.obj')
    compare_mesh(mesh_test)
    os.remove('tmp.obj')
