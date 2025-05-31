from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
def draw_roc(real_label, test_outputs_label,model,path):
    plt.figure(figsize=(8, 7), dpi=600, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
    # 假设有真实标签和模型的预测得分

    # 计算真阳性率和假阳性率
    fpr, tpr, thresholds = roc_curve(real_label, test_outputs_label)
    roc_auc = auc(fpr, tpr)
    print('test auc:',roc_auc)

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, color='darkorange',linestyle='-', lw=2.5, label='%s'% model+'-AUC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.01, 1.02])
    plt.ylim([-0.01, 1.02])
    plt.xticks(np.arange(0, 1.1, 0.1), fontproperties='Times New Roman', size=15)  # 绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1), fontproperties='Times New Roman', size=15)
    plt.xlabel('1-Specificity', fontsize=20, fontproperties='Times New Roman')
    plt.ylabel('Sensitivity', fontsize=20, fontproperties='Times New Roman')  # 绘制x,y 坐标轴对应的标签
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  # 设置顶端坐标轴的粗细
    plt.grid(ls=':', alpha=1, linewidth=1.4)
    plt.title(model, fontsize=20, fontdict={'family': 'Times New Roman', 'size': 20})  # 打印标题
    plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 18})
    plt.savefig(path,dpi=600)

