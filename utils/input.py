import cv2

from enum import Enum


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

    def __get_camera_frame(self, camera_index: int):
        pass

    def __get_video_frame(self, camera_index: int):
        pass

    def __get_image_frame(self, camera_index: int):
        camera_name, images_paths = list(self.__image_source.items())[camera_index]

        return cv2.imread(images_paths.pop(0))
