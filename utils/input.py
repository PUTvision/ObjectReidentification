import cv2
import os

from enum import Enum
from collections import namedtuple
from typing import NamedTuple

Frame = namedtuple('Frame', ['index', 'image'])


class ImageSourceType(Enum):
    cameras = 1
    video_files = 2
    images = 3


class InputHandler:
    def __init__(self, image_source_type: ImageSourceType, image_source):
        self.__image_source = image_source
        if image_source_type == ImageSourceType.cameras:
            self.get_frame = self.__get_camera_frame
        elif image_source_type == ImageSourceType.video_files:
            self.get_frame = self.__get_video_frame
        elif image_source_type == ImageSourceType.images:
            self.get_frame = self.__get_image_frame
        else:
            raise ValueError('Unknown image source type')

    def __get_camera_frame(self, camera_name: str):
        pass

    def __get_video_frame(self, camera_name: str):
        pass

    def __get_image_frame(self, camera_name: str):
        images_paths = self.__image_source[camera_name]

        path = images_paths.pop(0)
        file_name = os.path.split(path)[1]
        frame_index = file_name.split('_')[0]
        return Frame(frame_index, cv2.imread(path))
