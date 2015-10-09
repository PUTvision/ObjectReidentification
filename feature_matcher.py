#!/usr/bin/env python
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Object reidentification.')
parser.add_argument('-ic', '--input-comparison-dirs', dest='input_comparison_dirs', nargs='+',
                    help='Directories with pictures to compare')


def do_feature_matching(img1, img2, detector, matcher):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    keypoints_img1, descriptors_img1 = detector.detectAndCompute(gray_img1, None)
    keypoints_img2, descriptors_img2 = detector.detectAndCompute(gray_img2, None)

    cv2.imshow('img1', cv2.drawKeypoints(img1, keypoints_img1, None))
    cv2.imshow('img2', cv2.drawKeypoints(img2, keypoints_img2, None))

    matches = matcher.knnMatch(descriptors_img1, descriptors_img2, 2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    out_img = cv2.drawMatchesKnn(gray_img1, keypoints_img1, gray_img2, keypoints_img2, good, None, flags=2)
    cv2.imshow('img', out_img)
    cv2.waitKey()


def main(args):
    files_in_first_dir = sorted(os.scandir(args.input_comparison_dirs[0]), key=lambda t: t.name)
    files_in_second_dir = sorted(os.scandir(args.input_comparison_dirs[1]), key=lambda t: t.name)

    sift = cv2.xfeatures2d.SURF_create()
    bf_matcher = cv2.BFMatcher()

    for file_index in range(len(files_in_first_dir)):
        img1 = cv2.imread(files_in_first_dir[file_index].path)
        img2 = cv2.imread(files_in_second_dir[file_index].path)
        do_feature_matching(img1, img2, sift, bf_matcher)

if __name__ == '__main__':
    main(parser.parse_args())
