# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/customized_visualization_key_action.py

import cloudViewer as cv3d


def custom_key_action_without_kb_repeat_delay(pcd):
    rotating = False

    vis = cv3d.visualization.VisualizerWithKeyCallback()

    def key_action_callback(vis, action, mods):
        nonlocal rotating
        print(action)
        if action == 1:  # key down
            rotating = True
        elif action == 0:  # key up
            rotating = False
        elif action == 2:  # key repeat
            pass
        return True

    def animation_callback(vis):
        nonlocal rotating
        if rotating:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)

    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_action_callback(32, key_action_callback)  # space

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


if __name__ == "__main__":
    pcd = cv3d.io.read_point_cloud("../../test_data/fragment.ply")

    print(
        "Customized visualization with smooth key action (without keyboard repeat delay)"
    )
    custom_key_action_without_kb_repeat_delay(pcd)
