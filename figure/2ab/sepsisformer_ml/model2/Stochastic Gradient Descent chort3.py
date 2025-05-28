import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.linear_model import SGDClassifier
from comparison_utilis_data import *
import sklearn.calibration


def SGD_classify(train_features, train_labels, test_features, test_labels, loss, penalty):
    start_time = time.time()
    # loss: "hinge':线性SVM（默认）;  "modified_huber":带平滑的hinge loss; "log":logistic回归
    # penalty: "l1":L1正则化; "l2"：L2正则化(默认）;  "elasticnet":L2和L1的组合
    model = SGDClassifier(loss=loss, penalty=penalty,)
    #model = joblib.load('I:\mimic_transformer\model\SGD.model')
    model = sklearn.calibration.CalibratedClassifierCV(model)
    model.fit(train_features, train_labels)
    # path = "I:\mimic_transformer\model\SGD.model"
    # joblib.dump(model, path)
    print("model training is done in %0.3fs" % (time.time() - start_time))

    train_results = model.predict_proba(train_features)  # 测试集的预测结果
    print('test_results[:10]\n', train_results[:10])  # (2128,2)

    test_results = model.predict_proba(test_features)  # 测试集的预测结果
    print('test_results[:10]\n', test_results[:10])  # (2128,2)
    test_pre_label = model.predict(test_features)  # 测试集的预测标签，相当于将test_results经过argmax
    print('test_pre_lable[:10]\n', test_pre_label[:10])           # (2128,1)

    f1 = f1_score(test_labels, test_pre_label, average='binary')
    mcc = matthews_corrcoef(test_labels, test_pre_label)

    return train_results, test_results, accuracy_score(test_labels, test_pre_label), \
           confusion_matrix(test_labels, test_pre_label), f1, mcc


 # mimic3_eICU_smote_8
 # mimic4_smote_8
 # local_smote_8

 # mimic3_smote_36
 # mimic4_smote_36
def main():

    loss = 'hinge'
    penalty = 'l2'
    dir_name = ml_get_dir_name(model_name=f'Stochastic Gradient Descent_{loss}_{penalty}3')
    # 创建Logs、model文件夹，并以运行时间（年月日）+'LR' 命名
    logs_name, model_name = ml_mkdir(dir_name=dir_name)
    standard_pyh_train_inputs, pyh_train_labels, \
    standard_pyh_test_inputs, pyh_test_labels = ml_get_data(path=r'F:\sepsisformer\data\8\local_smote_8.csv',
                                                            logs=logs_name,
                                                            stage='pre')#I:\mimic_transformer\data\mimic3811_amend2.csv
    print('standard_pyh_train_inputs.shape', standard_pyh_train_inputs.shape)
    print('standard_pyh_test_inputs.shape', standard_pyh_test_inputs.shape)
    print('pyh_train_labels.shape', pyh_train_labels.shape)
    print('pyh_test_labels.shape', pyh_test_labels.shape)
    # dir_name = ml_get_dir_name(model_name=f'Stochastic Gradient Descent_{loss}_{penalty}')
    # # 创建Logs、model文件夹，并以运行时间（年月日）+'LR' 命名
    # logs_name, model_name = ml_mkdir(dir_name=dir_name)

    real_labels = pd.concat([pd.DataFrame(pyh_test_labels), pd.DataFrame(pyh_train_labels)], axis=0)
    real_labels.to_csv(os.path.join(logs_name, 'real_labels.csv'), index=False, header=True)


    train_results, test_results, test_acc, test_confusion, f1, mcc = SGD_classify(standard_pyh_train_inputs, pyh_train_labels,
                                                          standard_pyh_test_inputs, pyh_test_labels,
                                                          loss=loss, penalty=penalty)
    # print(type(test_results))  <class 'numpy.ndarray'>

    tn, fp, fn, tp = test_confusion.ravel()
    print('*********************', tp)
    print('********************', fp)
    print('*********************', tn)
    print('********************', fn)
    ppr = tp / (tp + fp)
    npr = tn / (tn + fn)
    ta = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)  # 查全率
    tr = (tp + tn) / pyh_train_labels.shape[0]

    test_auc = roc_auc_score(pyh_test_labels, test_results[:, -1])
    print("test_acc:\t", test_acc)
    print("test_auc:\t", test_auc)
    test_results_labels = pd.concat([pd.DataFrame(test_results), pd.DataFrame(pyh_test_labels)], axis=1)
    test_results_labels.columns = ['results[0]', 'results[1]', 'labels']
    test_results_labels.to_csv(os.path.join(logs_name, 'test_results_labels.csv'), index=False, header=True)
    #
    # train_results_labels = pd.concat([pd.DataFrame(train_results), pd.DataFrame(pyh_train_labels)], axis=1)
    # train_results_labels.columns = ['results[0]', 'results[1]', 'labels']
    # train_results_labels.to_csv(os.path.join(logs_name, 'train_results_labels.csv'), index=False, header=True)

    # test_results_labels = pd.concat([pd.DataFrame(test_results), pd.DataFrame(train_results)], axis=0)
    # test_results_labels.to_csv(os.path.join(logs_name, 'test_results_labels.csv'), index=False, header=True)

    s = open(os.path.join(logs_name, "confusion_matrix.txt"), mode="a")
    s.write(f"test_acc:{'%.4f' % test_acc}\n"
            f"test_auc:{'%.4f' % test_auc}\n"
            f"tp:{'%.4f' % tp}\nfp:{'%.4f' % fp}\ntn:{'%.4f' % tn}\nfn:{'%.4f' % fn}\n"
            f"ppr:{'%.4f' % ppr}\nnpr:{'%.4f' % npr}\nrecall:{'%.4f' % recall}\ntr:{'%.4f' % tr}\nta:{'%.4f' % ta}\n"
            f"f1:{'%.6f' % f1}\nmcc:{'%.6f' % mcc}\n"
            )
    s.close()


if __name__ == '__main__':
    main()
