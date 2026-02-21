def in_bounds(p, size):
    """Returns True if (x,y) is inside board."""
    x, y = p
    return 0 <= x < size and 0 <= y < size


def all_in_bounds(points, size):
    """Returns True if all points lie inside board."""
    return all(in_bounds(p, size) for p in points)


def no_self_overlap(points):
    """Returns True if there are no duplicate positions."""
    return len(points) == len(set(points))


def no_overlap(points_a, points_b):
    """Returns True if two sets of points do not intersect."""
    return set(points_a).isdisjoint(set(points_b))

def pixel_equal(pixel, rgb):
    """Returns True if the pixel at (x, y) has the exact RGB color."""
    return tuple(pixel) == rgb