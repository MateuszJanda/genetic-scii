#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import random
import time
import copy
import string
import pickle
import itertools
from collections import namedtuple
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np


# Evolution parameters
STEPS = 800
POPULATION_NUM = 100
BEST_NUM = 3
MUTATION_FACTOR = 1/8
CROSS_NUM = 2
SNAPSHOT_STEP = 10


# Different unicode chars sets used to generate final image
CHAR_BASE_SPACE     = " "
CHAR_BASE_ASCII     = string.digits + string.ascii_letters + string.punctuation

# https://en.wikipedia.org/wiki/Box_Drawing_(Unicode_block)
CHAR_BASE_BOX       = "".join([chr(ch) for ch in range(0x2500, 0x257F+1)])

# https://en.wikipedia.org/wiki/Block_Elements
CHAR_BASE_BLOCK     = "".join([chr(ch) for ch in range(0x2580, 0x259F+1)])

# https://en.wikipedia.org/wiki/Geometric_Shapes_(Unicode_block)
CHAR_BASE_GEOMETRIC = "".join([chr(ch) for ch in range(0x25A0, 0x25FF+1)])

CHAR_BASE_NOT_ALPHA = string.punctuation + CHAR_BASE_BOX + CHAR_BASE_BLOCK
CHAR_BASE_PUNCT_BOX = string.punctuation + CHAR_BASE_BOX

# Image and font configuration
BLACK = 0
WHITE = 255

# FONT_NAME = "DejaVuSansMono"
FONT_NAME = os.path.expanduser("~/.fonts/truetype/fixed-sys-excelsior/FSEX300.ttf")
FONT_SIZE = 16
FONT_SPACING = 0
FONT = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)


Individual = namedtuple("Individual", ["dna", "img", "char_base", "foreground", "background"])


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
    def __init__(self, symbol=" ", foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


def main():
    """
    Three steps of genetic algorithm:
    - mutate last population
    - select (score) best individuals
    - cross best individuals

    Tested on image with resoultion: 371x480
    """
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)

    input_arr = get_input_array("sincity1.jpg")
    edge_arr = get_edge_array(input_arr)
    population = basic_population(input_arr.shape)

    counter = 0
    for step in range(STEPS + 1):
        tic = time.time()

        mutate(population, mutate_fg_color=True, mutate_bg_color=True)
        best_idx, scores = select(population, input_arr, edge_arr, score_fun=score_pixels)
        population = cross(population, best_idx)

        if step % SNAPSHOT_STEP == 0:
            save_dna_as_img(population, best_idx[0], counter)
            counter += 1

        print("Generation: {step}, time: {t:.12f}, best: {best}, diff: {diff}"
            .format(step=step, t=time.time() - tic, best=scores[best_idx[0]],
                diff=scores[best_idx[-1]] - scores[best_idx[0]]))

    save_dna_as_img(population, best_idx[0], counter)
    print("End")


def create_char_base(dna):
    """
    Create char base.
    """
    FACTOR = 2
    char_base = Counter()
    char_base.update(CHAR_BASE_BLOCK * FACTOR)
    char_base.update(CHAR_BASE_BOX * FACTOR)
    char_base.update(CHAR_BASE_GEOMETRIC * FACTOR)

    current_num = len(list(char_base.elements()))
    print(f"Char in base: {current_num}")

    min_num = dna.shape[1] * dna.shape[0]
    char_base.update(CHAR_BASE_SPACE * (min_num - current_num))

    return char_base


def create_color_palette(dna):
    """
    Create color palette.
    """
    FACTOR = 2

    foregrounds = Counter([color for color in range(128, 256)] * FACTOR)
    foregrounds_num = len(list(foregrounds.elements()))
    print(f"Foregrounds colors in base: {foregrounds_num}")
    min_num = dna.shape[1] * dna.shape[0]
    foregrounds.update([0] * (min_num - foregrounds_num))

    backgrounds = Counter([color for color in range(128, 256)] * FACTOR)
    backgrounds_num = len(list(backgrounds.elements()))
    print(f"Backgrounds colors in base: {backgrounds_num}")
    min_num = dna.shape[1] * dna.shape[0]
    backgrounds.update([0] * (min_num - backgrounds_num))

    return foregrounds, backgrounds


def basic_population(img_shape):
    """
    Create basic population - list of individuals. Each individual is
    represented as DNA and image (redundant because image can be generated from
    DNA, but this operation is expensive). At the beginning each individual is
    black image.
    """
    bk_color = BLACK
    img = Image.new("L", color=bk_color, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//CHAR_SHAPE[0], img_shape[1]//CHAR_SHAPE[1]),
                  fill_value=DnaChar(background=bk_color))
    char_base = create_char_base(dna)
    foregrounds, backgrounds = create_color_palette(dna)

    population = [Individual(np.copy(dna), copy.copy(img), copy.copy(char_base), \
        copy.copy(foregrounds), copy.copy(backgrounds)) for _ in range(POPULATION_NUM)]

    individual = population[0]
    print(f"Input image resolution: {img_shape[1]}x{img_shape[0]}")
    print(f"ASCII resolution: {individual.dna.shape[1]}x{individual.dna.shape[0]}")
    print(f"Needed chars: {individual.dna.shape[1] * individual.dna.shape[0]}")
    print(f"Available chars: {len(''.join([ch * count for (ch, count) in char_base.items()]))}\n")

    return population


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


def mutate(population, mutate_fg_color=True, mutate_bg_color=True):
    """
    Mutate - add random "rectangle" to each individual in population. Could be
    tuned by MUTATION_FACTOR.
    """

    out = []

    # with ProcessPoolExecutor() as executor:
    # with ThreadPoolExecutor() as executor:
    for individual in population:
        out.append(fff1(individual))
        # out = executor.map(fff1, population)

    population = out



def fff1(individual, mutate_fg_color=True, mutate_bg_color=True):
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

    # Return symbols to char_base
    for x in range(begin_x, begin_x + size_x):
        for y in range(begin_y, begin_y + size_y):
            individual.char_base += individual.dna[y, x].symbol

    # Choice random symbol, foreground and background color
    new_symbols = random.choices(list(individual.char_base.elements()), k=surface_size)
    new_background = 0
    new_foreground = 0
    if mutate_fg_color:
        new_foreground = random.randint(0, 255)

    if mutate_bg_color:
        new_background = random.randint(0, 255)

    # Draw image for individual
    draw = ImageDraw.Draw(individual.img)
    for x in range(begin_x, begin_x + size_x):
        for y in range(begin_y, begin_y + size_y):
            dna_char = individual.dna[y, x]
            foreground = (dna_char.foreground + new_foreground)//2
            background = (dna_char.background + new_background)//2

            idx = (y - begin_y) * size_x + (x - begin_x)
            individual.dna[y, x] = DnaChar(new_symbols[idx], foreground, background)
            individual.char_base.substrac(new_symbols[idx])

            draw_char(draw, x, y, individual.dna[y, x])

    return individual


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

    best_idx = sorted(range(len(scores)), key=lambda k: scores[k])[:BEST_NUM]

    return best_idx, scores


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


def cross(population, best_idx):
    """
    Cross individuals in population (preview called select method should narrow
    this to BEST_NUM individuals). Copy random rectangle/matrix from one
    individual and merge it with second individual.
    """
    best_specimens = [copy.deepcopy(population[idx]) for idx in best_idx]

    result = []
    for _ in range(len(population)):
        individual1, individual2 = random.sample(best_specimens, 2)
        dna = np.copy(individual1.dna)
        img = copy.copy(individual1.img)
        char_base = copy.copy(individual1.char_base)

        for _ in range(CROSS_NUM):
            y = np.random.randint(dna.shape[0] - 1)
            x = np.random.randint(dna.shape[1] - 1)
            end_y = np.random.randint(y + 1, dna.shape[0])
            end_x = np.random.randint(x + 1, dna.shape[1])

            dna[y:end_y, x:end_x] = individual2.dna[y:end_y, x:end_x]
            cut = individual2.img.crop(box=(x*CHAR_SHAPE[1], y*CHAR_SHAPE[0], end_x*CHAR_SHAPE[1], end_y*CHAR_SHAPE[0]))
            img.paste(cut, box=(x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]))

        result.append(Individual(dna, img, individual1.char_base, individual1.foreground, individual1.background))

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
        for x, char in enumerate(line):
            draw_char(draw, x, y, char)

    return img


def draw_char(draw, x, y, char):
    """
    Draw character on the image.
    """
    pos_x, pos_y = x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]
    # Borders are part of rectangle so we must subtract 1 from bottom and right
    draw.rectangle(xy=[(pos_x, pos_y), (pos_x + CHAR_SHAPE[1] - 1, pos_y + CHAR_SHAPE[0] - 1)], fill=char.background)
    draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=FONT, spacing=FONT_SPACING)


if __name__ == "__main__":
    main()
