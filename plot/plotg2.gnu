set terminal fig color
set output 'out/plotg2.fig'
set size 1,1
set key right bottom
set xlabel 'Registration error threshold [px]'
set ylabel 'Prob. of correct registration [-]'
set ytics 0.2
set grid


set title "g2"
stats '../results/nFMNN03w2.srt' prefix "A"
plot [0:20] [0.3:]\
'../results/nPNN03w2.srt' using 1:($0/A_records) with lines title "PD output spring->winter w2" lw 2 lc 1,\
'../results/nFMNN03w2.srt' using 1:($0/A_records) with lines title "Augmented FM spring->winter w2" lw 2 lc 2,\
'../results/nPFM03.srt' using 1:($0/A_records) with lines title "Pure FM spring->winter" lw 2 lc 3,

set terminal x11
set output
replot