#!/usr/bin/env python3

import time
import ctypes
from ctypes import cdll

#
# Definitions from curses.h
#

NCURSES_ATTR_SHIFT = 8


def NCURSES_BITS(mask, shift):
    return mask << (shift + NCURSES_ATTR_SHIFT)


A_COLOR = NCURSES_BITS((1 << 8) - 1, 0)


def COLOR_PAIR(n):
    return NCURSES_BITS(n, 0) & A_COLOR


# https://docs.python.org/2.5/lib/ctypes-loading-dynamic-link-libraries.html
curses = ctypes.CDLL('libncursesw.so.6.1')
# curses = ctypes.CDLL('libncurses.so.5')

# http://www.tldp.org/HOWTO/NCURSES-Programming-HOWTO/
scr = curses.initscr()
curses.start_color()

curses.init_extended_color(2, 999, 0, 0)
curses.init_extended_color(3, 0, 999, 0)

curses.init_extended_pair(2, 2, 3)

# curses.printw("asdf")
curses.printw("asdf".encode('utf-8'))
curses.printw("asdf")

curses.attron(COLOR_PAIR(2))
curses.mvprintw(1, 1, "jkl".encode('utf-8'))
curses.attroff(COLOR_PAIR(2))

curses.refresh()

curses.getch()

curses.endwin()
