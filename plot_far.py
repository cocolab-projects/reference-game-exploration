import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

counter = 0
counterLine = 0
far_biList = np.array([])
far_nonbiList = np.array([])
close_biList = np.array([])
close_nonbiList = np.array([])

objects = ('Bi-LSTM', 'LSTM')
y_pos = np.arange(len(objects))

file = open('plot_data/plot_far_bi.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    far_biList = np.append(far_biList, float(line))

bi_far = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/plot_far_nonbi.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    far_nonbiList = np.append(far_nonbiList, float(line))

nonbi_far = counter/counterLine

counter = 0
counterLine = 0

performance = [bi_far,nonbi_far]

best = 0
low = 100

for x in np.nditer(far_biList):
    if low > x:
        low = x
    elif best < x:
        best = x

for x in np.nditer(far_nonbiList):
    if low > x:
        low = x
    elif best < x:
        best = x


print (far_biList)
print (far_nonbiList)

far_biList = np.std(far_biList)
far_nonbiList = np.std(far_nonbiList)

error = [far_biList,far_nonbiList]

plt.bar(y_pos, performance, yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Far')
plt.ylim(low - abs(best-low),best + abs(best-low))
plt.savefig('plot_data_png/far.png')
plt.show()
