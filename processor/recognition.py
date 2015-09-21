import cv2
import numpy
from collections import namedtuple
from utils.database import Database

Similarities = namedtuple('Similarities', ['head', 'trunk', 'legs'])
SubjectHistograms = namedtuple('SubjectHistograms', ['hue', 'sat'])
SubjectHueHistograms = namedtuple('SubjectHueHistograms', ['head', 'trunk', 'legs'])
SubjectSatHistograms = namedtuple('SubjectSatHistograms', ['head', 'trunk', 'legs'])


class Subject:
    def __init__(self, name: str, hue_histograms: SubjectHueHistograms,
                 saturation_histograms: SubjectSatHistograms):
        self.__name = name
        self.__histograms = [SubjectHistograms(hue_histograms, saturation_histograms)]
        self.__unmatched_histograms_counter = 0

    @staticmethod
    def __compare_histograms(histograms: SubjectHistograms, new_histograms: SubjectHistograms) -> float:
        head_hue_similarity = cv2.compareHist(histograms.hue.head, new_histograms.hue.head, cv2.HISTCMP_CORREL)
        head_sat_similarity = cv2.compareHist(histograms.sat.head, new_histograms.sat.head, cv2.HISTCMP_CORREL)
        trunk_hue_similarity = cv2.compareHist(histograms.hue.trunk, new_histograms.hue.trunk, cv2.HISTCMP_CORREL)
        trunk_sat_similarity = cv2.compareHist(histograms.sat.trunk, new_histograms.sat.trunk, cv2.HISTCMP_CORREL)
        legs_hue_similarity = cv2.compareHist(histograms.hue.legs, new_histograms.hue.legs, cv2.HISTCMP_CORREL)
        legs_sat_similarity = cv2.compareHist(histograms.sat.legs, new_histograms.sat.legs, cv2.HISTCMP_CORREL)

        return (head_hue_similarity * head_sat_similarity * trunk_hue_similarity * trunk_sat_similarity *
                legs_hue_similarity * legs_sat_similarity)

    def __check_if_approximately_matches(self, new_histograms: SubjectHistograms):
        matched = False
        for histograms in self.__histograms:
            if self.__compare_histograms(histograms, new_histograms) > 0.8:
                matched = True
                break

        if not matched:
            self.__unmatched_histograms_counter += 1

        return matched

    def add_histograms(self, hue_histograms: SubjectHueHistograms,
                       saturation_histograms: SubjectSatHistograms):
        subject_histograms = SubjectHistograms(hue_histograms, saturation_histograms)

        if not self.__check_if_approximately_matches(subject_histograms):
            self.__histograms.append(subject_histograms)

    def check_similarities(self, new_histograms: SubjectHistograms) -> float:
        score = 0.0
        number_of_histograms = len(self.__histograms)

        for histograms in self.__histograms:
            score += self.__compare_histograms(histograms, new_histograms)

        return score / number_of_histograms

    @property
    def name(self):
        return self.__name

    @property
    def number_of_histograms(self):
        return len(self.__histograms)


class SubjectIdentifier:
    def __init__(self, add_to_db):
        self.__add_to_db = add_to_db
        self.__subjects = Database.get_subjects()
        self.__subject = Database.get_subject(add_to_db)
        self.__first_run = True

        self.__scores = {}
        self.__best_match = None

    @staticmethod
    def __split_image(image) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray):
        image_height, image_width = image.shape
        one_fifth_image_height = image_height // 5
        three_fifths_image_height = one_fifth_image_height * 3
        top_margin = image_height - (one_fifth_image_height * 5)

        head = image[top_margin:one_fifth_image_height, 0:image_width]
        trunk = image[one_fifth_image_height:three_fifths_image_height, 0:image_width]
        legs = image[three_fifths_image_height:image_height, 0:image_width]

        return head, trunk, legs

    def identify(self, bounding_box_content: numpy.ndarray) -> Subject:
        image = cv2.cvtColor(bounding_box_content, cv2.COLOR_RGB2HSV)
        hsv_planes = cv2.split(image)
        hue_channel, saturation_channel = hsv_planes[0], hsv_planes[1]

        head_hue, trunk_hue, legs_hue = self.__split_image(hue_channel)
        head_sat, trunk_sat, legs_sat = self.__split_image(saturation_channel)

        hue_histograms = SubjectHueHistograms(cv2.calcHist([head_hue], [0], None, [19], [0, 180]),
                                              cv2.calcHist([trunk_hue], [0], None, [19], [0, 180]),
                                              cv2.calcHist([legs_hue], [0], None, [19], [0, 180]))

        sat_histograms = SubjectSatHistograms(cv2.calcHist([head_sat], [0], None, [16], [0, 256]),
                                              cv2.calcHist([trunk_sat], [0], None, [16], [0, 256]),
                                              cv2.calcHist([legs_sat], [0], None, [16], [0, 256]))

        if self.__add_to_db:
            if self.__subject is None:
                self.__subject = Subject(self.__add_to_db, hue_histograms, sat_histograms)
                Database.add_subject(self.__subject)
            else:
                self.__subject.add_histograms(hue_histograms, sat_histograms)

        for subject in self.__subjects:
            overall_score = subject.check_similarities(SubjectHistograms(hue_histograms, sat_histograms))

            try:
                self.__scores[subject.name] += overall_score
            except KeyError:
                self.__scores[subject.name] = overall_score

            if self.__best_match is None or self.__scores[subject.name] > self.__scores[self.__best_match.name]:
                self.__best_match = subject

        return self.__best_match
