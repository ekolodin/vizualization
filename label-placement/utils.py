class Pos:
    def __init__(self, *arr):
        self.xy = arr

    def __neg__(self):
        return [-a for a in self.xy]

    def __add__(self, that):
        return [a + b for a, b in zip(self.xy, that.xy)]

    def __sub__(self, that):
        return [a - b for a, b in zip(self.xy, that.xy)]

    def x(self):
        return self.xy[0]

    def y(self):
        return self.xy[1]

    def __iter__(self):
        for i in self.xy:
            yield i

    @staticmethod
    def parse(string):
        return Pos(*[int(a) for a in string.split(',')])


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def overlap(l1, r1, l2, r2):

        if l1.x >= r2.x or l2.x >= r1.x:
            return False

        if l1.y <= r2.y or l2.y <= r1.y:
            return False

        if l1.x == r1.x or l1.y == r2.y or l2.x == r2.x or l2.y == r2.y:
            return False

        return True


class Rectangle:
    def __init__(self, xy, height, width):
        self.xy = xy
        self.height = height
        self.width = width


class Box:
    def __init__(self, line):
        arr = line.split('\t')

        self.pos = Pos.parse(arr[0])
        self.size = Pos.parse(arr[1])
        self.offsets = [Pos.parse(pos) for pos in arr[2].split(' ')]

    def __iter__(self):
        for offset in self.offsets:
            yield Rectangle(xy=tuple(self.pos - offset), height=self.size.y(), width=self.size.x())
