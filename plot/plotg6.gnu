set terminal fig color
set output 'out/plotg6.fig'
set size 1,1
set key right bottom
set xlabel 'Registration error threshold [px]'
set ylabel 'Prob. of correct registration [-]'
set ytics 0.1
set grid

set tics font "Halvetica Bold,20"
set title "g6"
stats '../results/nFMNN02w1.srt' prefix "A"
plot [0:10] [0.5:]\
'../results/nFMNN02w1.srt' using 1:($0/A_records) with lines title "Augmented FM spring->fall w1" lw 2 lc 1,\
'../results/nFMNN03w2.srt' using 1:($0/A_records) with lines title "Augmented FM spring->winter w2" lw 2 lc 2,\
'../results/nFMNN02w1gt.srt' using 1:($0/A_records) with lines title "Augmented from Ground truth spring-fall" lw 2 lc 3,\
'../results/nFMNN03w2gt.srt' using 1:($0/A_records) with lines title "augmented From GT spring-wiunter" lw 2 lc 4,

set terminal x11
set output
replot
