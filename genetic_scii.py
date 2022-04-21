#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import random
import argparse
import time
import copy
import string
import pickle
from collections import namedtuple
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


# Evolution parameters
STEPS = 800
POPULATION_NUM = 100
BEST_NUM = 3
MUTATION_FACTOR = 1/8
CROSS_NUM = 2
SNAPSHOT_STEP = 5


# Different unicode chars sets used to generate final image
CHAR_POOL_SPACE     = " "
CHAR_POOL_ASCII     = string.digits + string.ascii_letters + string.punctuation

# https://en.wikipedia.org/wiki/Box_Drawing_(Unicode_block)
CHAR_POOL_BOX       = "".join([chr(ch) for ch in range(0x2500, 0x257F+1)])

# https://en.wikipedia.org/wiki/Block_Elements
CHAR_POOL_BLOCK     = "".join([chr(ch) for ch in range(0x2580, 0x259F+1)])

# https://en.wikipedia.org/wiki/Geometric_Shapes_(Unicode_block)
CHAR_POOL_GEOMETRIC = "".join([chr(ch) for ch in range(0x25A0, 0x25FF+1)])

CHAR_POOL_NOT_ALPHA = string.punctuation + CHAR_POOL_BOX + CHAR_POOL_BLOCK
CHAR_POOL_PUNCT_BOX = string.punctuation + CHAR_POOL_BOX

# Image and font configuration
BLACK = 0
WHITE = 255

# FONT_NAME = "DejaVuSansMono"
FONT_NAME = os.path.expanduser("~/.fonts/truetype/fixed-sys-excelsior/FSEX300.ttf")
FONT_SIZE = 16
FONT_SPACING = 0
FONT = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)


Individual = namedtuple("Individual", ["dna", "img", "char_pool", "fg_pool", "bg_pool"])


def singe_char_shape():
    """
    Character size estimation. Pillow doesn't have good method to get chars/font
    metrics, especially Unicode box-drawing characters are drown out of border
    (top/left).
    """
    img = Image.new("L", color=BLACK, size=(50, 50))
    draw = ImageDraw.Draw(img)
    width, height = draw.textsize(text="╂", font=FONT, spacing=FONT_SPACING)
    return height, width

# Pre-calculated single char shape
CHAR_SHAPE = singe_char_shape()


class DnaChar:
    """
    Single char property (symbol, foreground and background). DNA is array
    Chars.
    """
    def __init__(self, symbol=CHAR_POOL_SPACE, foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    """
    Three steps of genetic algorithm:
    - mutate last population
    - select (score) best individuals
    - cross best individuals

    Tested on image with resoultion: 371x480
    """
    parser = argparse.ArgumentParser(description="Genetic ASCII (glitch) generator.\n\n"
            "Mateusz Janda (c) <mateusz janda at gmail com>\n"
            "genetic-scii project github.com/MateuszJanda/genetic-scii\n",
        formatter_class=CustomFormatter)
    parser.add_argument("path", help="Path to image.")
    args = parser.parse_args()

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)

    input_arr = get_input_array(args.path)
    edge_arr = get_edge_array(input_arr)
    population = basic_population(input_arr.shape)

    counter = 0
    for step in range(STEPS + 1):
        tic = time.time()

        population = mutate(population)
        best_indices, scores = select(population, input_arr, edge_arr, score_fun=score_pixels)
        population = cross(population, best_indices)

        if step % SNAPSHOT_STEP == 0:
            save_dna_as_img(population, best_indices[0], counter)
            counter += 1

        print("Generation: {step}, time: {t:.12f}, best: {best}, diff: {diff}"
            .format(step=step, t=time.time() - tic, best=scores[best_indices[0]],
                diff=scores[best_indices[-1]] - scores[best_indices[0]]))

    save_dna_as_img(population, best_indices[0], counter)
    print("End")


def get_input_array(path):
    """
    Get input image (gray scale) as numpy array.

    In ImageMagic this could look like this:
    convert input.png  -fx 'intensity/8' output.png
    """
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return gray_img


def get_edge_array(gray_arr):
    """
    Get gray scale (numpy array) image of detected edges.

    Reference:
    https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html
    """
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(gray_arr, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_arr, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    THRESHOLD = 50
    MAX_VALUE = 255
    _, grad = cv2.threshold(grad, THRESHOLD, MAX_VALUE, cv2.THRESH_BINARY)

    return grad


def basic_population(img_shape):
    """
    Create basic population - list of individuals. Each individual is
    represented as DNA and image (redundant because image can be generated from
    DNA, but this operation is expensive). At the beginning each individual is
    black image.
    """
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//CHAR_SHAPE[0], img_shape[1]//CHAR_SHAPE[1]),
                  fill_value=DnaChar(background=BLACK))
    char_pool = create_char_pool(dna)
    fg_pool, bg_pool = create_color_pools(dna)

    population = [Individual(np.copy(dna), copy.copy(img), copy.copy(char_pool), \
        copy.copy(fg_pool), copy.copy(bg_pool)) for _ in range(POPULATION_NUM)]

    individual = population[0]
    print(f"Input image resolution: {img_shape[1]}x{img_shape[0]}")
    surface_size = individual.dna.shape[1] * individual.dna.shape[0]
    print(f"ASCII resolution: {individual.dna.shape[1]}x{individual.dna.shape[0]}")
    print(f"Needed chars: {surface_size}")
    print(f"Available chars: {surface_size + sum([count for _, count in char_pool.items()])}")
    print(f"Available foreground colors: {surface_size + sum([count for _, count in fg_pool.items()])}")
    print(f"Available background colors: {surface_size + sum([count for _, count in bg_pool.items()])}\n")

    return population


def create_char_pool(dna):
    """
    Create char pool.
    """
    surface_size = dna.shape[1] * dna.shape[0]
    factor = int(surface_size/676)

    char_pool = Counter()
    char_pool.update(CHAR_POOL_BLOCK * factor)
    char_pool.update(CHAR_POOL_BOX * factor)
    char_pool.update(CHAR_POOL_GEOMETRIC * factor)

    current_num = len(list(char_pool.elements()))
    print(f"Pure chars in pool: {current_num}")

    # Include spaces in empty (init) image
    char_pool[CHAR_POOL_SPACE] -= surface_size
    # If there is not enough chars fill with spaces
    char_pool[CHAR_POOL_SPACE] += max(0, surface_size - current_num)

    return char_pool


def create_color_pools(dna):
    """
    Create color palette.
    """
    surface_size = dna.shape[1] * dna.shape[0]

    fg_pool = Counter([color for color in range(0, 256, 16)] * int(surface_size/27))
    fg_pool_num = len(list(fg_pool.elements()))
    # Include white foreground in empty (init) image
    fg_pool[WHITE] -= surface_size
    # If there is not enough foreground colors fill with white
    fg_pool[WHITE] += max(0, surface_size - fg_pool_num)

    bg_pool = Counter([color for color in range(0, 256, 16)] * int(surface_size/54))
    bg_pool_num = len(list(bg_pool.elements()))
    # Include black background in empty (init) image
    bg_pool[BLACK] -= surface_size
    # If there is not enough background colors fill with black
    bg_pool[BLACK] += max(0, surface_size - bg_pool_num)

    return fg_pool, bg_pool


def mutate(population):
    """
    Mutate - add random "rectangle" to each individual in population. Could be
    tuned by MUTATION_FACTOR.
    """
    new_population = []

    for individual in population:
        width = individual.img.size[0]//CHAR_SHAPE[1]
        height = individual.img.size[1]//CHAR_SHAPE[0]

        # Randomize the location
        begin_x = random.randint(0, width - 1)
        begin_y = random.randint(0, height - 1)
        size_x = int(random.randint(1, width) * MUTATION_FACTOR)
        size_y = int(random.randint(1, height) * MUTATION_FACTOR)

        if begin_x + size_x > width:
            size_x = width - begin_x
        if begin_y + size_y > height:
            size_y = height - begin_y

        surface_size = size_x * size_y

        # Return symbols to char_pool
        for x in range(begin_x, begin_x + size_x):
            for y in range(begin_y, begin_y + size_y):
                individual.char_pool[individual.dna[y, x].symbol] += 1
                individual.fg_pool[individual.dna[y, x].foreground] += 1
                individual.bg_pool[individual.dna[y, x].background] += 1

        # Choice random symbol, foreground and background color
        new_symbols = random.choices(list(individual.char_pool.elements()), k=surface_size)
        new_foreground = random.choices(list(individual.fg_pool.elements()), k=surface_size)
        new_background = random.choices(list(individual.bg_pool.elements()), k=surface_size)

        # Draw image for individual
        draw = ImageDraw.Draw(individual.img)
        for x in range(begin_x, begin_x + size_x):
            for y in range(begin_y, begin_y + size_y):
                idx = (y - begin_y) * size_x + (x - begin_x)
                individual.dna[y, x] = DnaChar(new_symbols[idx], new_foreground[idx], new_background[idx])
                individual.char_pool[new_symbols[idx]] -= 1
                individual.fg_pool[new_foreground[idx]] -= 1
                individual.bg_pool[new_background[idx]] -= 1

                draw_char(draw, x, y, individual.dna[y, x])

        new_population.append(individual)

    return new_population


def select(population, input_arr, edge_arr, score_fun):
    """
    Score all individuals in population and choose BEST_NUM of them (from best
    to worst).
    """
    scores = []
    for individual in population:
        result = score_fun(individual, input_arr, edge_arr)
        scores.append(result)

    # input_arrs = itertools.repeat(input_arr, len(population))
    # edge_arrs = itertools.repeat(edge_arr, len(population))
    # with ThreadPoolExecutor() as executor:
    #     scores = executor.map(score_fun, population, input_arrs, edge_arrs)
    #     scores = list(scores)

    best_indices = sorted(range(len(scores)), key=lambda k: scores[k])[:BEST_NUM]

    return best_indices, scores


def score_pixels(individual, input_arr, edge_arr):
    """
    Score pixels differences between two images.
    """
    output_arr = np.array(individual.img)
    return np.sum(np.subtract(input_arr, output_arr, dtype=np.int64)**2)


def score_shape(individual, input_arr, edge_arr):
    """
    Score characters shape differences between two different images (currently
    Hausdorff distance).

    Reference:
    https://docs.opencv.org/3.4.9/d1/d85/group__shape.html
    https://answers.opencv.org/question/60974/matching-shapes-with-hausdorff-and-shape-context-distance/
    """
    PENALTY = 1000

    output_arr = np.array(individual.img)

    hd = cv2.createHausdorffDistanceExtractor()

    width = output_arr.shape[1]//CHAR_SHAPE[1]
    height = output_arr.shape[1]//CHAR_SHAPE[0]
    result = 0

    for x in range(width):
        for y in range(height):
            begin_x = x*CHAR_SHAPE[1]
            begin_y = y*CHAR_SHAPE[0]
            end_x = begin_x + CHAR_SHAPE[1]
            end_y = begin_y + CHAR_SHAPE[0]

            # When there is no shape (black only background) continue
            if not np.any(input_arr[begin_y:end_y, begin_x:end_x]):
                if np.any(output_arr[begin_y:end_y, begin_x:end_x]):
                    result += PENALTY
                continue

            contours1, _ = cv2.findContours(input_arr[begin_y:end_y, begin_x:end_x], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            if not contours1:
                continue

            contours2, _ = cv2.findContours(output_arr[begin_y:end_y, begin_x:end_x], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            # Penalty when missing shape proposal
            if not contours2:
                result += PENALTY
                continue

            r = hd.computeDistance(contours1[0], contours2[0])

            # Filter low quality distance calculation
            if r > 1.0:
                result += r
            else:
                result += PENALTY

            # Calculate average color
            color = np.average(input_arr[begin_y:end_y, begin_x:end_x])
            result += abs(color - individual.dna[y, x].foreground)

    return result


def score_edge_and_pixels(dna, input_arr, edge_arr, output_arr):
    """
    Score edge in first place, if there is no edges score pixels.
    """
    PENALTY = 1000

    hd = cv2.createHausdorffDistanceExtractor()

    width = output_arr.shape[1]//CHAR_SHAPE[1]
    height = output_arr.shape[1]//CHAR_SHAPE[0]
    result = 0

    for x in range(width):
        for y in range(height):
            begin_x = x*CHAR_SHAPE[1]
            begin_y = y*CHAR_SHAPE[0]
            end_x = begin_x + CHAR_SHAPE[1]
            end_y = begin_y + CHAR_SHAPE[0]

            # When there is no shape (black only background) continue
            if not np.any(input_arr[begin_y:end_y, begin_x:end_x]):
                if np.any(output_arr[begin_y:end_y, begin_x:end_x]):
                    result += PENALTY
                continue

            if np.any(edge_arr[begin_y:end_y, begin_x:end_x]):
                contours1, _ = cv2.findContours(input_arr[begin_y:end_y, begin_x:end_x], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                if not contours1:
                    continue

                contours2, _ = cv2.findContours(output_arr[begin_y:end_y, begin_x:end_x], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                # Penalty when missing shape proposal
                if not contours2:
                    result += PENALTY
                    continue

                r = hd.computeDistance(contours1[0], contours2[0])

                # Filter low quality distance calculation
                if r > 1.0:
                    result += r
                else:
                    result += PENALTY

                # Calculate average color
                color = np.average(input_arr[begin_y:end_y, begin_x:end_x])
                result += abs(color - dna[y, x].foreground)
            else:
                result += np.sum(np.abs(np.subtract(input_arr[begin_y:end_y, begin_x:end_x], output_arr[begin_y:end_y, begin_x:end_x])))

    return result


def cross(population, best_indices):
    """
    Cross individuals in population (preview called select method should narrow
    this to BEST_NUM individuals). Copy random rectangle/matrix from one
    individual and merge it with second individual.
    """
    best_individuals = [copy.deepcopy(population[idx]) for idx in best_indices]

    result = []
    for _ in range(len(population)):
        individual1, individual2 = random.sample(best_individuals, 2)
        dna = np.copy(individual1.dna)
        img = copy.copy(individual1.img)
        char_pool = copy.copy(individual1.char_pool)
        fg_pool = copy.copy(individual1.fg_pool)
        bg_pool = copy.copy(individual1.bg_pool)

        draw = ImageDraw.Draw(img)
        for _ in range(CROSS_NUM):
            begin_x = np.random.randint(dna.shape[1] - 1)
            begin_y = np.random.randint(dna.shape[0] - 1)
            end_x = np.random.randint(begin_x + 1, dna.shape[1])
            end_y = np.random.randint(begin_y + 1, dna.shape[0])

            # Copy characters from individual2 if possible
            for x in range(begin_x, end_x):
                for y in range(begin_y, end_y):
                    # Return symbol to char pool
                    char_pool[individual1.dna[y, x].symbol] += 1
                    fg_pool[individual1.dna[y, x].foreground] += 1
                    bg_pool[individual1.dna[y, x].background] += 1

                    # If char can't be copied choice avaliable one
                    dna_char = copy.copy(individual2.dna[y, x])
                    if char_pool[dna_char.symbol] <= 0:
                        dna_char.symbol = random.choice(list(char_pool.elements()))

                    if fg_pool[dna_char.foreground] <= 0:
                        dna_char.foreground = random.choice(list(fg_pool.elements()))

                    if bg_pool[dna_char.background] <= 0:
                        dna_char.background = random.choice(list(bg_pool.elements()))

                    char_pool[dna_char.symbol] -= 1
                    fg_pool[dna_char.foreground] -= 1
                    bg_pool[dna_char.background] -= 1
                    dna[y, x] = dna_char

                    draw_char(draw, x, y, dna_char)

        result.append(Individual(dna, img, char_pool, fg_pool, bg_pool))

    return result


def print_dna(dna):
    """
    For debug only - draw raw character representation of individual DNA.
    """
    for line in dna:
        print("".join([ch.symbol for ch in line]))


def load_dna_from_pickle(file_name):
    """
    Load DNA from pickle.
    """
    with open(file_name, "rb") as f:
        dna = pickle.load(f)

    return dna


def save_dna_as_pickle(population, idx, step):
    """
    Serialize DNA to pickle.
    """
    dna, _ = population[idx]
    with open("p%04d.dat" % step, "wb") as f:
        pickle.dump(dna, f)


def save_dna_as_img(population, idx, step):
    """
    Save DNA as image.
    """
    individual = population[idx]
    img_shape = (individual.img.size[1], individual.img.size[0])
    out_img = dna_to_img(individual.dna, img_shape)
    out_img.save("snapshot_%04d.png" % step)

    # Problem with Pillow not deterministic char shape calculation, so in worst
    # case this two images could differ
    # assert np.all(np.array(out_img) == np.array(img))


def dna_to_img(dna, img_shape):
    """
    For some characters Pillow doesn't draw them in correct position, so output
    image could contain some glitches, and differ from input image.
    """
    img = Image.new("L", color=BLACK, size=(img_shape[1], img_shape[0]))
    draw = ImageDraw.Draw(img)

    for y, line in enumerate(dna):
        for x, dna_char in enumerate(line):
            draw_char(draw, x, y, dna_char)

    return img


def draw_char(draw, x, y, dna_char):
    """
    Draw character on the image.
    """
    pos_x, pos_y = x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]
    # Borders are part of rectangle so we must subtract 1 from bottom and right
    draw.rectangle(xy=[(pos_x, pos_y), (pos_x + CHAR_SHAPE[1] - 1, pos_y + CHAR_SHAPE[0] - 1)], fill=dna_char.background)
    draw.text(xy=(pos_x, pos_y), text=dna_char.symbol, fill=dna_char.foreground, font=FONT, spacing=FONT_SPACING)


if __name__ == "__main__":
    main()
