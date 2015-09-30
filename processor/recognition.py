import cv2
import numpy
from collections import namedtuple, OrderedDict
from utils.database import Database
from typing import Tuple, Optional

SplitImage = Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

Similarities = namedtuple('Similarities', ['head', 'trunk', 'legs'])
SubjectHistograms = namedtuple('SubjectHistograms', ['hue', 'sat', 'blue', 'green', 'red', 'cr', 'cb'])
SubjectBodyHistograms = namedtuple('SubjectBodyHistograms', ['first', 'second', 'third', 'fourth', 'fifth', 'sixth'])


class Subject:
    def __init__(self, name: str, histograms: SubjectHistograms, descriptors: numpy.ndarray):
        self.__name = name
        self.__histograms = [histograms]
        self.__descriptors = [descriptors]
        self.__unmatched_histograms_counter = 0

    @staticmethod
    def __compare_body_parts_histograms(histograms: SubjectBodyHistograms,
                                        new_histograms: SubjectBodyHistograms) -> float:
        first_similarity = cv2.compareHist(histograms.first, new_histograms.first, cv2.HISTCMP_CORREL)
        second_similarity = cv2.compareHist(histograms.second, new_histograms.second, cv2.HISTCMP_CORREL)
        third_similarity = cv2.compareHist(histograms.third, new_histograms.third, cv2.HISTCMP_CORREL)
        fourth_similarity = cv2.compareHist(histograms.fourth, new_histograms.fourth, cv2.HISTCMP_CORREL)
        fifth_similarity = cv2.compareHist(histograms.fifth, new_histograms.fifth, cv2.HISTCMP_CORREL)
        sixth_similarity = cv2.compareHist(histograms.sixth, new_histograms.sixth, cv2.HISTCMP_CORREL)

        return (first_similarity + second_similarity + third_similarity + fourth_similarity + fifth_similarity +
                sixth_similarity) / 6

    def __check_histograms_similarity(self, histograms: SubjectHistograms, new_histograms: SubjectHistograms) -> float:
        hue_sim = self.__compare_body_parts_histograms(histograms.hue, new_histograms.hue)
        sat_sim = self.__compare_body_parts_histograms(histograms.sat, new_histograms.sat)
        blue_sim = self.__compare_body_parts_histograms(histograms.blue, new_histograms.blue)
        green_sim = self.__compare_body_parts_histograms(histograms.green, new_histograms.green)
        red_sim = self.__compare_body_parts_histograms(histograms.red, new_histograms.red)
        cr_sim = self.__compare_body_parts_histograms(histograms.cr, new_histograms.cr)
        cb_sim = self.__compare_body_parts_histograms(histograms.cb, new_histograms.cb)

        return (hue_sim + sat_sim + blue_sim + green_sim + red_sim + cr_sim + cb_sim) / 7

    def __check_if_approximately_matches(self, new_histograms: SubjectHistograms):
        matched = False
        for histograms in self.__histograms:
            if self.__check_histograms_similarity(histograms, new_histograms) > 0.9:
                matched = True
                break

        if not matched:
            self.__unmatched_histograms_counter += 1

        return matched

    def add_histograms(self, histograms: SubjectHistograms):
        if not self.__check_if_approximately_matches(histograms):
            self.__histograms.append(histograms)

    def add_descriptors(self, descriptors: numpy.ndarray):
        self.__descriptors.append(descriptors)

    def check_similarities(self, new_histograms: SubjectHistograms) -> float:
        score = 0.0
        number_of_histograms = len(self.__histograms)

        for histograms in self.__histograms:
            score += self.__check_histograms_similarity(histograms, new_histograms)

        return score / number_of_histograms

    @property
    def name(self):
        return self.__name

    @property
    def number_of_histograms(self):
        return len(self.__histograms)

    @property
    def histograms(self):
        return self.__histograms

    @property
    def descriptors(self):
        return self.__descriptors


class SubjectIdentifier:
    def __init__(self, name_to_save_in_db: Optional[str]=None, enable_feature_matching=False):
        self.__add_to_db = name_to_save_in_db
        self.__feature_matching_enabled = enable_feature_matching
        self.__subjects = Database.get_subjects()
        self.__subject = Database.get_subject(name_to_save_in_db)
        self.__first_run = True

        self.__scores = {}
        self.__best_match = None
        self.__new_histograms = None
        self.__sift = cv2.xfeatures2d.SIFT_create()

    @staticmethod
    def __split_image(image: numpy.ndarray) -> SplitImage:
        image_height, image_width = image.shape
        one_sixth_image_height = image_height / 6
        two_sixths_image_height = one_sixth_image_height * 2
        three_sixths_image_height = one_sixth_image_height * 3
        four_sixth_image_height = image_height * 4
        five_sixth_image_height = image_height * 5
        one_sixth_image_height = image_height / 6

        first = image[0:one_sixth_image_height, 0:image_width]
        second = image[one_sixth_image_height:two_sixths_image_height, 0:image_width]
        third = image[two_sixths_image_height:three_sixths_image_height, 0:image_width]
        fourth = image[three_sixths_image_height:four_sixth_image_height, 0:image_width]
        fifth = image[four_sixth_image_height:five_sixth_image_height, 0:image_width]
        sixth = image[five_sixth_image_height:image_height, 0:image_width]

        return first, second, third, fourth, fifth, sixth

    @staticmethod
    def __calculate_histograms(masks: SplitImage, buckets: int, value_range: Tuple[int, int], images: SplitImage):
        histograms = []

        for image, mask in zip(images, masks):
            histogram = cv2.calcHist([image], [0], mask, [buckets], value_range)
            cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
            histograms.append(histogram)

        return SubjectBodyHistograms(*histograms)

    def __get_descriptors(self, image_with_mask: numpy.ndarray):
        image, mask = image_with_mask
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.__sift.detectAndCompute(gray_image, mask)

        return descriptors

    @staticmethod
    def __do_feature_match(source_descriptors: numpy.ndarray, new_descriptors: numpy.ndarray):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(source_descriptors, new_descriptors, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        return len(good)

    @staticmethod
    def __extract_person(image: numpy.ndarray, mask: numpy.ndarray):
        bgd_model = numpy.zeros((1, 65), numpy.float64)
        fgd_model = numpy.zeros((1, 65), numpy.float64)

        if mask is None:
            mask = numpy.zeros(image.shape[:2], numpy.uint8)
            horizontal_center = image.shape[1] // 2
            vertical_center = image.shape[0] // 2

            cv2.ellipse(mask, (horizontal_center, vertical_center),
                        (horizontal_center, vertical_center), 0, 0, 360, 3, -1)

            cv2.line(mask, (horizontal_center, 0), (horizontal_center, image.shape[0]), 1, 1)

        mask, bgd_model, fgd_model = cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        mask = numpy.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask[:, :, numpy.newaxis]

        cv2.imshow('grabcut', image)

        return image, mask

    def identify(self, bounding_box_content: numpy.ndarray, mask: numpy.ndarray=None) -> Subject:
        bounding_box_content, mask = self.__extract_person(bounding_box_content, mask)

        image_hsv = cv2.cvtColor(bounding_box_content, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, _ = cv2.split(image_hsv)
        hue_images = self.__split_image(hue_channel)
        sat_images = self.__split_image(saturation_channel)

        image_bgr = bounding_box_content
        blue_channel, green_channel, red_channel = cv2.split(image_bgr)
        blue_images = self.__split_image(blue_channel)
        green_images = self.__split_image(green_channel)
        red_images = self.__split_image(red_channel)

        image_ycrcb = cv2.cvtColor(bounding_box_content, cv2.COLOR_BGR2YCrCb)
        _, cr_channel, cb_channel = cv2.split(image_ycrcb)
        cr_images = self.__split_image(cr_channel)
        cb_images = self.__split_image(cb_channel)

        masks = self.__split_image(mask)

        hue_histograms = self.__calculate_histograms(masks, 19, (0, 180), hue_images)
        sat_histograms = self.__calculate_histograms(masks, 16, (0, 256), sat_images)
        blue_histograms = self.__calculate_histograms(masks, 16, (0, 256), blue_images)
        green_histograms = self.__calculate_histograms(masks, 16, (0, 256), green_images)
        red_histograms = self.__calculate_histograms(masks, 16, (0, 256), red_images)
        cr_histograms = self.__calculate_histograms(masks, 16, (0, 256), cr_images)
        cb_histograms = self.__calculate_histograms(masks, 16, (0, 256), cb_images)

        new_histograms = SubjectHistograms(hue_histograms, sat_histograms, blue_histograms, green_histograms,
                                           red_histograms, cr_histograms, cb_histograms)
        self.__new_histograms = new_histograms

        if self.__feature_matching_enabled:
            new_descriptors = self.__get_descriptors((image_bgr, mask))
        else:
            new_descriptors = None

        if self.__add_to_db:
            if self.__subject is None:
                self.__subject = Subject(self.__add_to_db, new_histograms, new_descriptors)
                Database.add_subject(self.__subject)
            else:
                self.__subject.add_histograms(new_histograms)

                if self.__feature_matching_enabled:
                    self.__subject.add_descriptors(new_descriptors)

        for subject in self.__subjects:
            overall_score = subject.check_similarities(new_histograms)

            if self.__feature_matching_enabled:
                self.__do_feature_match(subject.descriptors[0], new_descriptors)

            try:
                self.__scores[subject.name][1] += overall_score
            except KeyError:
                self.__scores[subject.name] = {}
                self.__scores[subject.name][0], self.__scores[subject.name][1] = subject, overall_score

            if self.__best_match is None or self.__scores[subject.name][1] > self.__scores[self.__best_match.name][1]:
                self.__best_match = subject

        if self.__feature_matching_enabled:
            ordered_scores = OrderedDict(sorted(self.__scores.items(), key=lambda t: t[1][1], reverse=True))

            counter = 0
            highest_feature_match = 0
            for subject_with_score in ordered_scores.values():
                feature_match = self.__do_feature_match(subject_with_score[0].descriptors[0], new_descriptors)
                print(subject_with_score[0].name, feature_match)

                if feature_match > highest_feature_match:
                    self.__best_match = subject_with_score[0]
                    highest_feature_match = feature_match

                if counter > 8:
                    break
                counter += 1

        return self.__best_match
