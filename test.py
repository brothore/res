import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# 实际值
acts = [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
# 预测值
pres2 = [0.5187,0.1514,0.2547,0.0455,0.6197,0.4798,0.2487,0.283,0.1549,0.1347,0.2103,0.1034,0.3752,0.3845,0.4326]
# pres2 = [1 if x >= 0.5 else 0 for x in pres]
act = np.array(acts)
pre = np.array(pres2)
FPR, TPR, thresholds = metrics.roc_curve(act, pre)
AUC = auc(FPR, TPR)
print('AUC:',AUC)
plt.rc('font', family='Arial Unicode MS', size=14)
plt.plot(FPR,TPR,label="AUC={:.2f}" .format(AUC),marker = 'o',color='b',linestyle='--')
plt.legend(loc=4, fontsize=10)
plt.title('ROC曲线',fontsize=20)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.show()