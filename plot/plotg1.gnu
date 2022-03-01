set terminal fig color
set output 'out/plotg1.fig'
set size 1,1
set key right bottom
set xlabel 'Registration error threshold [px]'
set ylabel 'Prob. of correct registration [-]'
set ytics 0.1
set grid

set title "g1"
stats '../results/nFMNN02w1.srt' prefix "A"
plot [0:20] [0.3:]\
'../results/nFMNN02w1.srt' using 1:($0/A_records) with lines title "Augmented FM spring->fall w1" lw 2 lc 1,\
'../results/nFMNN03w2.srt' using 1:($0/A_records) with lines title "Augmented FM spring->winter w2" lw 2 lc 2,\
'../results/nPFM02.srt' using 1:($0/A_records) with lines title "Pure FM spring->fall" lw 2 lc 3,\
'../results/nPFM03.srt' using 1:($0/A_records) with lines title "Pure FM spring->winter" lw 2 lc 4,\
'../results/nPFM01.srt' using 1:($0/A_records) with lines title "Pure FM spring->summer" lw 2 lc 5,

set terminal x11
set output
replot
