# libraries
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


with open('Attention_Score/pickle_Att_Quran_Bukhari_1.pickle','rb') as f:
    y0,y1=pickle.load(f)


# use the plot function
plt.plot(y0,color='g', label='Quran')
plt.plot(y1,color='b', label="Prophet's Statement")

plt.xlabel('Feature Vector Position',fontweight='bold',size=15)
plt.ylabel('Normalized Attention Score',fontweight='bold',size=15)

plt.legend(loc='best', prop={'size': 15, 'weight': 'bold'})
#plt.show()
plt.savefig("Graphs\pic_Att_Quran_Prophet_Statement.png")
