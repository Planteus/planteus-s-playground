
import math
import stddraw


class Turtle:
    def __init__(self, x0, y0, a0):
        self._x = x0
        self._y = y0
        self._angle = a0

    def turnLeft(self, delta):
        self._angle += delta

    def goForward(self, step):
        oldx = self._x
        oldy = self._y
        self._x += step * math.cos(math.radians(self._angle))
        self._y += step * math.sin(math.radians(self._angle))
        stddraw.line(oldx, oldy, self._x, self._y)
