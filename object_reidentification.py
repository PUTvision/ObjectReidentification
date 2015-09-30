#!/usr/bin/env python
import cv2
import numpy
import os
import argparse
import copy

from typing import Optional
from collections import OrderedDict
from utils.input import InputHandler, ImageSourceType
from processor.tracking import CMT
from processor import util
from processor.recognition import SubjectIdentifier
from utils.database import Database

parser = argparse.ArgumentParser(description='Object reidentification.')
parser.add_argument('-i', '--input-dir', dest='input_dir', help='Directory with cameras directories.')
parser.add_argument('-ic', '--input-comparison-dirs', dest='input_comparison_dirs', nargs='+',
                    help='Directories with pictures to compare')
parser.add_argument('-s', '--single-images', action='store_true', default=False,
                    dest='single_images', help='Use single image for each subject')
parser.add_argument('-c', '--camera-name', type=str, dest='camera_name', help='Camera name.')
parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to save results to.')
parser.add_argument('-a', '--add-subject', dest='add_subject', help='Add subject to database.')


def main(args):
    def check_extension(entry):
        if entry.name.endswith('.jpg') or entry.name.endswith('.jpeg') or entry.name.endswith('.png'):
            return True
        else:
            return False

    input_dir = args.input_dir
    input_comparison_dirs = args.input_comparison_dirs

    if input_dir:
        cameras_with_images_paths = OrderedDict()
        dir_content = os.scandir(input_dir)
        for camera_entry in dir_content:
            if not camera_entry.name.startswith('.') and camera_entry.is_dir():
                cameras_with_images_paths[camera_entry.name] = []

                for image_entry in sorted(os.scandir(camera_entry.path), key=lambda t: t.name):
                    if not image_entry.name.startswith('.') and image_entry.is_file() and check_extension(image_entry):
                        cameras_with_images_paths[camera_entry.name].append(image_entry.path)

        input_handler = InputHandler(ImageSourceType.images, cameras_with_images_paths)
    elif input_comparison_dirs:
        cameras_with_images_paths = OrderedDict()
        for i in range(len(input_comparison_dirs)):
            cameras_with_images_paths[i] = []

            dir_content = sorted(os.scandir(input_comparison_dirs[i]), key=lambda t: t.name)
            for camera_entry in dir_content:
                cameras_with_images_paths[i].append(camera_entry.path)

        input_handler = InputHandler(ImageSourceType.images, cameras_with_images_paths)
    else:
        input_handler = None

    if not args.single_images:
        run(input_handler, args.camera_name, args.add_subject)
    else:
        handle_single_images(input_handler, len(input_comparison_dirs))


def run(input_handler: InputHandler, camera_name: str, add_to_db: Optional[str]):
    Database.initialize()
    im0 = input_handler.get_frame(camera_name).image
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_draw = numpy.copy(im0)
    tl, br = util.get_rect(im_draw)

    cmt = CMT(im_gray0, tl, br, estimate_rotation=False)
    identifier = SubjectIdentifier(add_to_db)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        try:
            # Read image
            im = input_handler.get_frame(camera_name).image

            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_draw = numpy.copy(im)

            cmt.process_frame(im_gray)

            # Display results
            # Draw updated estimate
            if cmt.has_result:
                cropped_image = im[cmt.tl[1]:cmt.bl[1], cmt.tl[0]:cmt.tr[0]]

                difference = cv2.absdiff(im_gray0, im_gray)

                blurred = cv2.medianBlur(difference, 3)
                display = cv2.compare(blurred, 6, cv2.CMP_GT)

                eroded = cv2.erode(display, structuring_element)
                dilated = cv2.dilate(eroded, structuring_element)

                cropped_mask = dilated[cmt.tl[1]:cmt.bl[1], cmt.tl[0]:cmt.tr[0]]
                cropped_mask[cropped_mask == 255] = 1

                horizontal_center = cropped_mask.shape[1] // 2
                vertical_center = cropped_mask.shape[0] // 2

                cv2.ellipse(cropped_mask, (horizontal_center, vertical_center),
                            (horizontal_center, vertical_center), 0, 0, 360, 3, -1)

                cv2.rectangle(im_draw, cmt.tl, cmt.br, (255, 0, 0), 4)
                subject = identifier.identify(cropped_image, cropped_mask)
                cv2.putText(im_draw, subject.name, cmt.tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            util.draw_keypoints(cmt.tracked_keypoints, im_draw, (255, 255, 255))
            util.draw_keypoints(cmt.votes[:, :2], im_draw)
            util.draw_keypoints(cmt.outliers[:, :2], im_draw, (0, 0, 255))
            cv2.imshow('main', im_draw)
            cv2.waitKey(1)

            im_gray0 = im_gray
        except IndexError:
            Database.save_db()
            exit(0)


def handle_single_images(input_handler: InputHandler, number_of_cameras: int):
    Database.initialize()

    input_handler_copy = copy.deepcopy(input_handler)
    while True:
        try:
            image_index, image = input_handler_copy.get_frame(0)

            identifier = SubjectIdentifier(image_index)
            identifier.identify(image)
        except IndexError:
            break

    for i in range(1, number_of_cameras):
        input_handler_copy = copy.deepcopy(input_handler)
        successes = 0
        fails = 0
        while True:
            try:
                image_index, image = input_handler_copy.get_frame(i)

                identifier = SubjectIdentifier()
                subject = identifier.identify(image)

                if subject.name == image_index:
                    successes += 1
                else:
                    fails += 1
                    print(image_index, 'identified as', subject.name)
            except IndexError:
                print(successes * 100 / (successes + fails))
                break


if __name__ == '__main__':
    main(parser.parse_args())
