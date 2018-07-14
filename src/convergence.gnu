#!/usr/local/bin/gnuplot

reset

set terminal svg size 800,600 enhanced font 'Verdana,12'
set output 'convergence.svg'

set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
set style line 12 lc rgb '#808080' lt 0 lw 1

set grid back ls 12

set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 1 # --- red
set style line 2 lc rgb '#5e9c36' pt 2 ps 1 lt 1 lw 1 # --- green
set style line 3 lc rgb '#65393d' pt 3 ps 1 lt 1 lw 1 # --- brown
set style line 4 lc rgb '#3db7c2' pt 4 ps 1 lt 1 lw 1 # --- blue
set style line 5 lc rgb '#f9c386' pt 5 ps 1 lt 1 lw 1 # --- blue
set style line 6 lc rgb '#98cdc5' pt 6 ps 1 lt 1 lw 1 # --- grey-cyan-thing

#set xrange [0:10000]
set yrange [0:200]

set key top left

set xlabel 'Episodes'
set ylabel 'Mean reward over 100 episodes'

plot 'log' using 1:2 title ''   with lines ls 1, \
     'log' using 1:2 notitle    with lines ls 2 smooth bezier, \
