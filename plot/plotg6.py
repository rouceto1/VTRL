import matplotlib.pyplot as plt
import numpy as np
file_list=['../results/nFMNN02w1.srt','../results/nFMNN03w2.srt','../results/nFMNN02w1gt.srt', '../results/nFMNN03w2gt.srt']
names  = [ "FM training spring-fall $e(w_1)$","FM training spring-winter $e(w_2)$", "GT training spring-fall $e(w_1)$","GT training spring-winter $e(w_2)$"]

fig, ax = plt.subplots()
lengt = 0
for i in range(len(file_list)):
    print ("reading: " + file_list[i])
    with open(file_list[i]) as file:
        lines = [float(line.rstrip()) for line in file]
        
    length = len(lines)
    lines = np.array(lines)
    print ("plotting: " + file_list[i])
    print(lines)
    print(length)
    ax.plot(lines,np.array(range(length))/length, label=names[i])
plt.xlim(0,10)
plt.grid()
plt.ylim(0.5,1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.title("Accuracies of NN+FM taught on FM or ground truth of Nordland")
plt.ylabel('Prob. of correct registration [-]',fontsize = 13)
plt.xlabel('Registration error threshold [pixels]',fontsize = 13)
plt.savefig('out/plotg6.eps', format='eps')
plt.show()

#set terminal fig color
#set output 'out/plotg1.fig'
# set size 1,1
# set key right bottom
# set xlabel 'Registration error threshold [px]'
# set ylabel 'Prob. of correct registration [-]'
# set ytics 0.1
# set grid

# set title "g1"
# stats '../results/nFMNN02w1.srt' prefix "A"
# plot [0:20] [0.3:]\
# '../results/nFMNN02w1.srt' using 1:($0/A_records) with lines title "Augmented FM spring->fall w1" lw 2 lc 1,\
# '../results/nFMNN03w2.srt' using 1:($0/A_records) with lines title "Augmented FM spring->winter w2" lw 2 lc 2,\
# '../results/nPFM02.srt' using 1:($0/A_records) with lines title "Pure FM spring->fall" lw 2 lc 3,\
# '../results/nPFM03.srt' using 1:($0/A_records) with lines title "Pure FM spring->winter" lw 2 lc 4,\
# '../results/nPFM01.srt' using 1:($0/A_records) with lines title "Pure FM spring->summer" lw 2 lc 5,

# set terminal x11
# set output
# replot
