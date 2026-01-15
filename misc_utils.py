import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.animation import FuncAnimation, PillowWriter

def greedy_distance(list1, list2, squared_distances: bool = False):
    total_distance = 0.0
    used1, used2 = set(), set()
    sq_distances = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2, a, b) for a in list1 for b in list2]
    sq_distances.sort()
    for dist, a, b in sq_distances:
        if (a not in used1) and (b not in used2):
            used1.add(a)
            used2.add(b)
            if squared_distances:
                total_distance += dist
            else:
                total_distance += math.sqrt(dist)
    
    return total_distance

def prod_distance(list1, list2):
    total_distance = 0.0
    a_x = np.array([i[0] for i in list2], dtype = np.float64)
    a_y = np.array([i[1] for i in list2], dtype = np.float64)
    for point in list1:
        total_product = ((a_x - point[0]) ** 2 + (a_y - point[1]) ** 2).prod()
        total_distance += total_product ** (1 / len(list1)) # geometric average
    return total_distance

def calculate_variety_score(data, range_values): # Normalized Shannon entropy
    counts, _ = np.histogram(data, bins = len(range_values), range = (min(range_values), max(range_values) + 1))
    probabilities = counts / counts.sum()
    ent = entropy(probabilities, base = 2)
    max_entropy = np.log2(len(range_values))
    variety_score = ent / max_entropy if max_entropy > 0 else 0.0
    return variety_score

def find_latest_version(folder_path: str, file_name: str, separator: str, extension: str):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None, 0

    pattern = re.compile(re.escape(file_name) + re.escape(separator) + r"(\d+)" + re.escape(extension))

    latest_version = -1
    latest_file_path = None

    for root, _, files in os.walk(folder_path):
        for file in files:
            match = pattern.match(file)
            if match:
                version = int(match.group(1))
                if version > latest_version:
                    latest_version = version
                    latest_file_path = os.path.join(root, file)

    if latest_file_path:
        return latest_file_path, latest_version
    else:
        return None, 0

def create_animation(image_list, title: str | None = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    animations_dir = os.path.join(script_dir, 'animations')
    os.makedirs(animations_dir, exist_ok=True)
    _, version = find_latest_version(animations_dir, "animation", "_", ".gif")
    version += 1

    fig, ax = plt.subplots()
    img = ax.imshow(image_list[0], interpolation = 'nearest')
    ax.set_title(f'Animation {version}' if title is None else title)
    ax.axis('off')

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = FuncAnimation(fig, update, frames=image_list, repeat=False)

    output_path = os.path.join(animations_dir, f'animation_{version}.gif')
    ani.save(output_path, writer=PillowWriter(fps = 30))
    print(f'Animation saved to: {output_path}')

def smooth_list(values, n):
    smoothed_values = list()
    length = len(values)
    
    for i in range(length):
        start = max(0, i - n)
        end = min(length, i + n + 1)
        window = values[start:end]
        smoothed_values.append(sum(window) / len(window))
        
    return smoothed_values
