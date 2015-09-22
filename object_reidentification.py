#!/usr/bin/env python
import cv2
import numpy as np
import os
import argparse
import copy

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
parser.add_argument('-c', '--camera-number', type=int, dest='camera_number', help='Camera number.')
parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to save results to.')
parser.add_argument('-a', '--add-subject', dest='add_subject', help='Add subject to database.')


def main(args):
    def check_extension(file_name: str):
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            return True
        else:
            return False

    input_dir = args.input_dir
    input_comparison_dirs = args.input_comparison_dirs

    if input_dir:
        cameras = [file for file in os.listdir(input_dir)
                   if not file.startswith('.') and os.path.isdir(os.path.join(input_dir, file))]

        cameras_with_images_paths = OrderedDict()
        for camera in cameras:
            camera_path = os.path.join(input_dir, camera)
            cameras_with_images_paths[camera] = []
            for file in sorted(os.listdir(camera_path)):
                file_path = os.path.join(camera_path, file)
                if not file.startswith('.') and os.path.isfile(file_path) and check_extension(file):
                    cameras_with_images_paths[camera].append(file_path)

        input_handler = InputHandler(ImageSourceType.images, cameras_with_images_paths)
    elif input_comparison_dirs:
        print(input_comparison_dirs)
        cameras_with_images_paths = OrderedDict()
        for i in range(len(input_comparison_dirs)):
            cameras_with_images_paths[i] = []
            files = sorted(os.listdir(input_comparison_dirs[i]))
            for file in files:
                cameras_with_images_paths[i].append(os.path.join(input_comparison_dirs[i], file))

        input_handler = InputHandler(ImageSourceType.images, cameras_with_images_paths)
    else:
        input_handler = None

    if not args.single_images:
        run(input_handler, args.camera_number, args.add_subject)
    else:
        handle_single_images(input_handler, len(input_comparison_dirs))


def run(input_handler: InputHandler, camera_number, add_to_db):
    Database.initialize()
    im0 = input_handler.get_frame(camera_number).image
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_draw = np.copy(im0)
    tl, br = util.get_rect(im_draw)

    cmt = CMT(im_gray0, tl, br, estimate_rotation=False)
    identifier = SubjectIdentifier(add_to_db)

    while True:
        try:
            # Read image
            im = input_handler.get_frame(camera_number).image

            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_draw = np.copy(im)

            cmt.process_frame(im_gray)

            # Display results
            # Draw updated estimate
            if cmt.has_result:
                cropped_image = im[cmt.tl[1]:cmt.bl[1], cmt.tl[0]:cmt.tr[0]]

                ellipse_mask = np.zeros(cropped_image.shape, np.uint8)
                horizontal_center = cropped_image.shape[1] // 2
                vertical_center = cropped_image.shape[0] // 2
                cv2.ellipse(ellipse_mask, (horizontal_center, vertical_center),
                            (horizontal_center, vertical_center), 0, 0, 360, (255, 255, 255), -1)
                masked_image = cropped_image & ellipse_mask

                cv2.rectangle(im_draw, cmt.tl, cmt.br, (255, 0, 0), 4)
                subject = identifier.identify(masked_image)
                cv2.putText(im_draw, subject.name, cmt.tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            util.draw_keypoints(cmt.tracked_keypoints, im_draw, (255, 255, 255))
            util.draw_keypoints(cmt.votes[:, :2], im_draw)
            util.draw_keypoints(cmt.outliers[:, :2], im_draw, (0, 0, 255))
            cv2.imshow('main', im_draw)
            cv2.waitKey(1)
        except IndexError:
            Database.save_db()
            exit(0)


def handle_single_images(input_handler: InputHandler, number_of_cameras):
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
