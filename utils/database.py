import os
import pickle


class Database:
    __subjects = None

    @classmethod
    def initialize(cls):
        os.makedirs(os.path.dirname('db/subjects'), exist_ok=True)
        try:
            with open('db/subjects', 'rb+') as subjects_db:
                cls.__subjects = pickle.load(subjects_db)
        except FileNotFoundError:
            cls.__subjects = []

        print(len(cls.__subjects), 'subjects in database:')

        for subject in cls.__subjects:
            print(subject.name, 'with', subject.number_of_histograms, 'histograms')

    @classmethod
    def get_subjects(cls):
        return cls.__subjects

    @classmethod
    def get_subject(cls, name: str):
        for subject in cls.__subjects:
            if subject.name == name:
                return subject

        return None

    @classmethod
    def save_db(cls):
        with open('db/subjects', 'wb+') as subjects_db:
            pickle.dump(cls.__subjects, subjects_db)

    @classmethod
    def add_subject(cls, subject_to_add: 'Subject'):
        for subject in cls.__subjects:
            if subject.name == subject_to_add.name:
                return

        cls.__subjects.append(subject_to_add)
        cls.save_db()
