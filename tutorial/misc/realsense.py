# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/misc/realsense.py

# Simple example to show RealSense camera discovery and frame capture

import cloudViewer as cv3d

if __name__ == "__main__":

    cv3d.t.io.RealSenseSensor.list_devices()
    rscam = cv3d.t.io.RealSenseSensor()
    rscam.start_capture()
    print(rscam.get_metadata())
    for fid in range(5):
        rgbd_frame = rscam.capture_frame()
        cv3d.io.write_image(f"color{fid:05d}.jpg",
                           rgbd_frame.color.to_legacy_image())
        cv3d.io.write_image(f"depth{fid:05d}.png",
                           rgbd_frame.depth.to_legacy_image())
        print("Frame: {}, time: {}s".format(fid, rscam.get_timestamp() * 1e-6))

    rscam.stop_capture()
