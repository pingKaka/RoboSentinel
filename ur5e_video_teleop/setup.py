from setuptools import find_packages, setup


setup(
    name="ur5e_video_teleop",
    version="0.1.0",
    description="UR5e teleoperation with hold-to-record camera video only",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "pynput",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "ur5e-video-teleop = ur5e_video_teleop.run_video_teleop:main",
        ]
    },
)
