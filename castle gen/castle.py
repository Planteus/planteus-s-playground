import sys
import stddraw
from turtle import Turtle


def koch(n, step, turtle):
    if n == 0:
        turtle.goForward(step)
        return
    koch(n-1, step, turtle)
    turtle.turnLeft(60.0)
    koch(n-1, step, turtle)
    turtle.turnLeft(-120.0)
    koch(n-1, step, turtle)
    turtle.turnLeft(60.0)
    koch(n-1, step, turtle)


def castle(n, step, turtle):
    turtle.turnLeft(120.0)
    koch(n, step, turtle)
    turtle.turnLeft(-120.0)
    koch(n, step, turtle)
    turtle.turnLeft(-120.0)
    koch(n, step, turtle)


def main():
    n = int(sys.argv[1])
    step = 3.0 ** -n
    stddraw.setPenRadius(0.0)
    turtle = Turtle(0.5, (1-(3**0.5)/4)/4, 0.0)
    castle(n, step / (3**0.5), turtle)
    stddraw.show()


if __name__ == '__main__':
    main()
