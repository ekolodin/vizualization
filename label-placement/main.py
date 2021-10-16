# pip install z3-solver
import z3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from matplotlib import patches

from utils import *


class LabelPlacement:

    def __init__(self):
        pass

    @staticmethod
    def __draw(answer):

        labels = []
        for t in answer.decls():
            if z3.is_true(answer[t]):
                labels.append(t)

        fig, ax = plt.subplots(figsize=(8, 8))

        for entry in labels:
            box_index, label_index = map(int, str(entry).split('$'))
            box = boxes[box_index]
            label = [x for x in box][label_index]

            ax.scatter(*tuple(box.pos), lw=2, color='black')
            ax.add_patch(
                patches.Rectangle(
                    xy=label.xy, height=label.height, width=label.width, edgecolor=np.random.rand(3), facecolor='none', lw=3
                )
            )

        plt.show()

    @staticmethod
    def __check_intersection(a, b):
        first, second = Point(*a.xy), Point(*b.xy)

        l1, r1 = Point(first.x, first.y + a.height), Point(first.x + a.width, first.y)
        l2, r2 = Point(second.x, second.y + b.height), Point(second.x + b.width, second.y)

        return Point.overlap(l1, r1, l2, r2)

    @staticmethod
    def __add_at_least_one(box_iter, labels):

        """ Add conditions for at least one label to be selected """

        template = [z3.Not(z3.Bool(f'{box_iter}${i}')) for i in range(len(labels))]
        conjunctions = []

        for i in range(len(labels)):
            current = deepcopy(template)
            current[i] = z3.Not(current[i])
            conjunctions.append(z3.And(current))

        return z3.simplify(z3.Or(conjunctions))

    @staticmethod
    def __add_no_intersect(first_box_iter, first_labels, second_box_iter, second_labels):

        """ Add the conditions for the prohibition of intersection """

        disjunctions = []

        for i, label_first in enumerate(first_labels):
            for j, label_second in enumerate(second_labels):

                if LabelPlacement.__check_intersection(label_first, label_second):

                    x, y = z3.Bool(f'{first_box_iter}${i}'), z3.Bool(f'{second_box_iter}${j}')
                    a, b, c = z3.And([z3.Not(x), y]), z3.And([x, z3.Not(y)]), z3.And([z3.Not(x), z3.Not(y)])

                    disjunctions.append(z3.Or([a, b, c]))

        return disjunctions

    def __call__(self, boxes):

        disjunctions = []

        for first_box_it, box_first in enumerate(boxes):
            for second_box_it, box_second in enumerate(boxes):

                if box_first == box_second:
                    continue

                first_labels = [x for x in box_first]
                second_labels = [x for x in box_second]

                disjunctions.append(self.__add_at_least_one(first_box_it, first_labels))
                disjunctions.append(self.__add_at_least_one(second_box_it, second_labels))
                disjunctions.extend(self.__add_no_intersect(first_box_it, first_labels, second_box_it, second_labels))

        result = z3.And(disjunctions)

        solver = z3.Solver()
        solver.add(result)
        solver.check()

        answer = solver.model()

        self.__draw(answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to file')

    args = parser.parse_args()

    with open(args.path, 'r') as f:
        boxes = [Box(line) for line in f.readlines()]

    LabelPlacement()(boxes)
