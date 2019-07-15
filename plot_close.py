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

file = open('plot_data/plot_close_bi.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    close_biList = np.append(close_biList, float(line))


bi_close = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/plot_close_nonbi.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    close_nonbiList = np.append(close_nonbiList, float(line))

nonbi_close = counter/counterLine

counter = 0
counterLine = 0

performance = [bi_close,nonbi_close]

best = 0
low = 100

for x in np.nditer(close_biList):
    if low > x:
        low = x
    elif best < x:
        best = x

for x in np.nditer(close_nonbiList):
    if low > x:
        low = x
    elif best < x:
        best = x

print (close_biList)
print (close_nonbiList)

close_biList = np.std(close_biList)
close_nonbiList = np.std(close_nonbiList)

error = [close_biList,close_nonbiList]

plt.bar(y_pos, performance, align='center', yerr=error,
       alpha=0.5,
       ecolor='black',
       capsize=10)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Close')
plt.ylim(low - abs(best-low),best + abs(best-low))
plt.savefig('plot_data_png/close.png')
plt.show()