import pygame.draw
import math

# https://stackoverflow.com/a/73855696
def draw_rectangle(screen, x, y, width, height, color, rx, ry):
    """Draw a rectangle, centered at x, y.

    Arguments:
    x (int/float):
        The x coordinate of the center of the shape.
    y (int/float):
        The y coordinate of the center of the shape.
    width (int/float):
        The width of the rectangle.
    height (int/float):
        The height of the rectangle.
    color (str):
        Name of the fill color, in HTML format.
    """

    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = math.sqrt((height / 2) ** 2 + (width / 2) ** 2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = math.atan2(height / 2, width / 2)

    # Transform that angle to reach each corner of the rectangle.
    angles = [angle, -angle + math.pi, angle + math.pi, -angle]

    # Calculate rotation.
    rot_radians = math.atan2(ry, rx)

    # Calculate the coordinates of each point.
    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rot_radians)
        x_offset = radius * math.cos(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))

    pygame.draw.polygon(screen, color, points)