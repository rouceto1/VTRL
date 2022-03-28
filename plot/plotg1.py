import matplotlib.pyplot as plt
import numpy as np
file_list=['../results/nFMNN02w1.srt','../results/nFMNN03w2.srt','../results/nPFM02.srt', '../results/nPFM03.srt','../results/nPFM01.srt']
names  = [ "NN+FM spring-fall $e(02,w_1)$","NN+FM spring-winter $e(03,w_2)$", "Pure FM spring-summer $e(01,w_0)$","Pure FM spring-fall $e(02,w_0)$","Pure FM spring-winter $e(03,w_0)$"]

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
plt.xlim(0,20)
plt.grid()
plt.ylim(0.3,1)
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.title("Accuracies over different seasons of Nordland")
plt.ylabel('Prob. of correct registration [-]',fontsize = 13)
plt.xlabel('Registration error threshold [pixels]',fontsize = 13)
plt.savefig('out/plotg1.eps', format='eps')
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
