set terminal fig color
set output 'out/plotg5.fig'
set size 1,1
set key right bottom
set xlabel 'Registration error threshold [px]'
set ylabel 'Prob. of correct registration [-]'
set ytics 0.1
set grid

set title "g5"
stats '../results/sFMNNw3.srt' prefix "A"
plot [0:100] [0.3:]\
'../results/sFMNNw3.srt' using 1:($0/A_records) with lines title "stromovka full" lw 2 lc 1,\
'../results/sFMNNw3gt.srt' using 1:($0/A_records) with lines title "stromovka GT" lw 2 lc 2,\
'../results/cFMNNw3.srt' using 1:($0/A_records) with lines title "carlevaris full" lw 2 lc 3,\
'../results/cFMNNw3gt.srt' using 1:($0/A_records) with lines title "carlevaris gt" lw 2 lc 4,

set terminal x11
set output
replot
