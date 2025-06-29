import turtle
import math

def draw_tree(length, level):
    if level == 0:
        return
    turtle.forward(length)
    turtle.left(45)
    draw_tree(length * math.sqrt(2) / 2, level - 1)
    turtle.right(90)
    draw_tree(length * math.sqrt(2) / 2, level - 1)
    turtle.left(45)
    turtle.backward(length)

def pythagoras_tree(level):
    turtle.speed(0)
    turtle.left(90)
    draw_tree(100, level)
    turtle.done()

print("\n=== Pythagoras Tree (Fractal) ===")
try:
    pythagoras_tree(6)
except turtle.Terminator:
    print("Pythagoras Tree window closed.")