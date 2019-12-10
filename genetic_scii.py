#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import random
import time
import copy
import string
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np


STEPS = 5001
POPULATION_NUM = 200
BEST_NUM = 3
MUTATION_FACTOR = 1/8
CROSS_NUM = 2


CHAR_BASE_SPACE     = " "
CHAR_BASE_ASCII     = string.digits + string.ascii_letters + string.punctuation
# https://en.wikipedia.org/wiki/Box_Drawing_(Unicode_block)
CHAR_BASE_BOX       = "".join([chr(ch) for ch in range(0x2500, 0x257F+1)])
# https://en.wikipedia.org/wiki/Block_Elements
CHAR_BASE_BLOCK     = "".join([chr(ch) for ch in range(0x2580, 0x259F+1)])
CHAR_BASE_NOT_ALPHA = string.punctuation + CHAR_BASE_BOX + CHAR_BASE_BLOCK
CHAR_BASE_PUNCT_BOX = string.punctuation + CHAR_BASE_BOX

CHAR_BASE = CHAR_BASE_ASCII

BLACK = 0
WHITE = 255

FONT_NAME = 'DejaVuSansMono'
FONT_SIZE = 16
FONT_SPACING = 0
FONT = ImageFont.truetype(FONT_NAME, size=FONT_SIZE)


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

CHAR_SHAPE = singe_char_shape()



class Char:
    def __init__(self, symbol=" ", foreground=WHITE, background=BLACK):
        self.symbol = symbol
        self.foreground = foreground
        self.background = background


def main():
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    orig_arr = get_orig_array()
    # orig_arr = convert_to_mosaic(orig_arr)
    # orig_arr = invert_colors(orig_arr)
    population = basic_population(orig_arr.shape)

    for step in range(STEPS):
        tic = time.time()

        mutate(population, CHAR_BASE, random_background=False)
        best_idx, scores = select(population, orig_arr)
        population = cross(population, best_idx)

        if step % 10 == 0:
            save_dna_as_img(population, best_idx[0], step)

        print("Generation: {step}, time: {t}, best: {best}, diff: {diff}"
            .format(step=step, t=time.time() - tic, best=scores[best_idx[0]],
                diff=scores[best_idx[-1]] - scores[best_idx[0]]))

    save_dna_as_img(population, best_idx[0], step)
    print("End")


def basic_population(img_shape):
    bk_color = BLACK
    img = Image.new("L", color=bk_color, size=(img_shape[1], img_shape[0]))
    dna = np.full(shape=(img_shape[0]//CHAR_SHAPE[0], img_shape[1]//CHAR_SHAPE[1]),
                  fill_value=Char(background=bk_color))
    population = [(np.copy(dna), copy.copy(img)) for _ in range(POPULATION_NUM)]
    return population


def get_orig_array(path="orig.png"):
    img = Image.open(path)

    assert img.mode == "L", "Gray scale (8-bit) image required"

    return np.array(img)


def convert_to_mosaic(arr):
    for x in range(0, arr.shape[1], CHAR_SHAPE[1]):
        for y in range(0, arr.shape[0], CHAR_SHAPE[0]):
            arr[y:y+CHAR_SHAPE[0], x:x+CHAR_SHAPE[1]] = np.average(arr[y:y+CHAR_SHAPE[0], x:x+CHAR_SHAPE[1]])

    return arr


def convert_to_contours(arr):
    """
    https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html
    """
    pass


def invert_colors(arr):
    arr = np.invert(arr)
    return arr


def mutate(population, char_base, random_background=True):
    for dna, img in population:
        width = img.size[0]//CHAR_SHAPE[1]
        height = img.size[1]//CHAR_SHAPE[0]
        begin_x = random.randint(0, width - 1)
        begin_y = random.randint(0, height - 1)
        size_x = int(random.randint(1, width) * MUTATION_FACTOR)
        size_y = int(random.randint(1, height) * MUTATION_FACTOR)

        symbol = random.choice(char_base)
        new_foreground = random.randint(0, 255)
        if random_background:
            new_background = random.randint(0, 255)
        else:
            new_background = 0

        draw = ImageDraw.Draw(img)

        for x in range(begin_x, min(begin_x + size_x, width)):
            for y in range(begin_y, min(begin_y + size_y, height)):
                char = dna[y, x]
                foreground = (char.foreground + new_foreground)//2
                background = (char.background + new_background)//2
                dna[y, x] = Char(symbol, foreground, background)

                draw_char(draw, x, y, dna[y, x])


def select(population, orig_arr):
    scores = {}
    for idx, (_, img) in enumerate(population):
        result = np.sum(np.subtract(orig_arr, img, dtype=np.int64)**2)
        scores[idx] = result

    best_idx = sorted(scores, key=scores.get)[:BEST_NUM]

    return best_idx, scores


def cross(population, best_idx):
    best_specimens = [copy.deepcopy(population[idx]) for idx in best_idx]

    result = []
    for _ in range(len(population)):
        (dna1, img1), (dna2, img2) = random.sample(best_specimens, 2)
        dna = np.copy(dna1)
        img = copy.copy(img1)

        for _ in range(CROSS_NUM):
            y = np.random.randint(dna.shape[0] - 1)
            x = np.random.randint(dna.shape[1] - 1)
            end_y = np.random.randint(y + 1, dna.shape[0])
            end_x = np.random.randint(x + 1, dna.shape[1])

            dna[y:end_y, x:end_x] = dna2[y:end_y, x:end_x]
            c = img2.crop(box=(x*CHAR_SHAPE[1], y*CHAR_SHAPE[0], end_x*CHAR_SHAPE[1], end_y*CHAR_SHAPE[0]))
            img.paste(c, box=(x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]))

        result.append((dna, img))

    return result


def print_dna(dna):
    for line in dna:
        print("".join([ch.symbol for ch in line]))


def load_dna_from_pickle(file_name):
    with open(file_name, "rb") as f:
        dna = pickle.load(f)

    return dna


def save_dna_as_pickle(population, idx, step):
    dna, _ = population[idx]
    with open("p%04d.dat" % step, "wb") as f:
        pickle.dump(dna, f)


def save_dna_as_img(population, idx, step):
    dna, img = population[idx]
    img_shape = (img.size[1], img.size[0])
    out_img = dna_to_img(dna, img_shape)
    out_img.save("a%04d.png" % step)

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
    pos_x, pos_y = x*CHAR_SHAPE[1], y*CHAR_SHAPE[0]
    # Borders are part of rectangle so we must subtract 1 from bottom and right
    draw.rectangle(xy=[(pos_x, pos_y), (pos_x + CHAR_SHAPE[1] - 1, pos_y + CHAR_SHAPE[0] - 1)], fill=char.background)
    draw.text(xy=(pos_x, pos_y), text=char.symbol, fill=char.foreground, font=FONT, spacing=FONT_SPACING)


if __name__ == '__main__':
    main()
