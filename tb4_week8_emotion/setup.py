from setuptools import setup
from glob import glob

package_name = "tb4_week8_emotion"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/models", glob("models/*.task")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ngô Thiện Đắc",
    maintainer_email="student@example.com",
    description="Facial emotion recognition and TurtleBot 4 behavior control.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "emotion_recognition_node = tb4_week8_emotion.emotion_recognition_node:main",
            "emotion_behavior_node = tb4_week8_emotion.emotion_behavior_node:main",
        ],
    },
)
