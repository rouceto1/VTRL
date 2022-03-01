set terminal fig color
set output 'out/plotg3.fig'
set size 1,1
set key right bottom
set key font ",400"
set xlabel 'Registration error threshold [px]'
set ylabel 'Prob. of correct registration [-]'
set ytics 0.1
set grid
set title "Stromovka"
    
stats '../results/sPFM.srt' prefix "A"
plot [0:100] [0.3:]\
'../results/sPFM.srt' using 1:($0/A_records) with lines title "stromovka Pure FM" lw 2 lc 1,\
'../results/sFMNNw1.srt' using 1:($0/A_records) with lines title "Augmented FM w1" lw 2 lc 2,\
'../results/sFMNNw2.srt' using 1:($0/A_records) with lines title "Augmented FM w2" lw 2 lc 3,\
'../results/sFMNNw3.srt' using 1:($0/A_records) with lines title "Augmented FM w3" lw 2 lc 4,


set terminal x11
set output
replot
