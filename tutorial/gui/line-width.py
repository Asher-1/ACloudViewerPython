import cloudViewer as cv3d
import random

NUM_LINES = 10


def random_point():
    return [5 * random.random(), 5 * random.random(), 5 * random.random()]


def main():
    pts = [random_point() for _ in range(0, 2 * NUM_LINES)]
    line_indices = [[2 * i, 2 * i + 1] for i in range(0, NUM_LINES)]
    colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)]

    lines = cv3d.geometry.LineSet()
    lines.points = cv3d.utility.Vector3dVector(pts)
    lines.lines = cv3d.utility.Vector2iVector(line_indices)
    # The default color of the lines is white, which will be invisible on the
    # default white background. So we either need to set the color of the lines
    # or the base_color of the material.
    lines.colors = cv3d.utility.Vector3dVector(colors)

    # Some platforms do not require OpenGL implementations to support wide lines,
    # so the renderer requires a custom shader to implement this: "unlitLine".
    # The line_width field is only used by this shader; all other shaders ignore
    # it.
    mat = cv3d.visualization.rendering.Material()
    mat.shader = "unlitLine"
    mat.line_width = 10  # note that this is scaled with respect to pixels,
    # so will give different results depending on the
    # scaling values of your system
    cv3d.visualization.draw({
        "name": "lines",
        "geometry": lines,
        "material": mat
    })


if __name__ == "__main__":
    main()
