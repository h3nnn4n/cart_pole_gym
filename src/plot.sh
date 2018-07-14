#!/bin/sh

./convergence.gnu
rsvg-convert -b white -h 800 convergence.svg > convergence.png
