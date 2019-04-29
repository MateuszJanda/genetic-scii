#!/usr/bin/env python3

import time
import ctypes
from ctypes import cdll

# https://docs.python.org/2.5/lib/ctypes-loading-dynamic-link-libraries.html
curses = ctypes.CDLL('libncursesw.so.6.1')
# curses = ctypes.CDLL('libncurses.so.5')

# http://www.tldp.org/HOWTO/NCURSES-Programming-HOWTO/
scr = curses.initscr()
curses.start_color()

# curses.printw("asdf")
curses.printw("asdf".encode('utf-8'))
curses.printw("asdf")
curses.mvprintw(1, 1, "jkl".encode('utf-8'))
curses.refresh()

curses.getch()

curses.endwin()
