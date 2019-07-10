import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

counter = 0
counterLine = 0
fat_oneList = np.array([])
fat_twoList = np.array([])
fat_threeList = np.array([])
medium_oneList = np.array([])
medium_twoList = np.array([])
medium_threeList = np.array([])
skinny_oneList = np.array([])
skinny_twoList = np.array([])
skinny_threeList = np.array([])

objects = ('F-1','F-2','F-3','M-1','M-2','M-3','S-1','S-2','S-3',)
y_pos = np.arange(len(objects))

file = open('plot_data/Rework_Sup/plot_far_False_Fat_1.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    fat_oneList = np.append(fat_oneList, float(line))

fat_one = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Fat_2.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    fat_twoList = np.append(fat_twoList, float(line))


fat_two = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Fat_3.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    fat_threeList = np.append(fat_threeList, float(line))

fat_three = counter/counterLine






counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Medium_1.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    medium_oneList = np.append(medium_oneList, float(line))

medium_one = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Medium_2.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    medium_twoList = np.append(medium_twoList, float(line))


medium_two = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Medium_3.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    medium_threeList = np.append(medium_threeList, float(line))

medium_three = counter/counterLine






counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Skinny_1.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    skinny_oneList = np.append(skinny_oneList, float(line))

skinny_one = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Skinny_2.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    skinny_twoList = np.append(skinny_twoList, float(line))


skinny_two = counter/counterLine

counter = 0
counterLine = 0

file = open('plot_data/Rework_Sup/plot_far_False_Skinny_3.txt', 'r') 
for line in file: 
    counter = counter + float(line)
    counterLine += 1
    skinny_threeList = np.append(skinny_threeList, float(line))

skinny_three = counter/counterLine





performance = [fat_one,fat_two,fat_three,medium_one,medium_two,medium_three,skinny_one,skinny_two,skinny_three]

best = 0
low = 100

for x in np.nditer(fat_oneList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(fat_twoList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(fat_threeList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(medium_oneList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(medium_twoList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(medium_threeList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(skinny_oneList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(skinny_twoList):
    if low > x:
        low = x
    elif best < x:
        best = x
for x in np.nditer(skinny_threeList):
    if low > x:
        low = x
    elif best < x:
        best = x



fat_oneList = np.std(fat_oneList)
fat_twoList = np.std(fat_twoList)
fat_threeList = np.std(fat_threeList)
medium_oneList = np.std(medium_oneList)
medium_twoList = np.std(medium_twoList)
medium_threeList = np.std(medium_threeList)
skinny_oneList = np.std(skinny_oneList)
skinny_twoList = np.std(skinny_twoList)
skinny_threeList = np.std(skinny_threeList)

error = [fat_oneList,fat_twoList,fat_threeList,medium_oneList,medium_twoList,medium_threeList,skinny_oneList,skinny_twoList,skinny_threeList]

plt.bar(y_pos, performance, align='center', yerr=error,
       alpha=0.5,
       ecolor='black',
       capsize=10)
plt.xticks(y_pos, objects)
plt.ylabel('Accurcy')
plt.title('Far/Non-Bi')
plt.ylim(low - abs(best-low),best + abs(best-low))
plt.savefig('plot_data_png/Rework_Sup/rework-far-nonbi.png')
plt.show()