from sklearn.metrics import accuracy_score
import sys
import numpy as np

def calculate_accuracy(gt_pred_file):
    fp =open(gt_pred_file)
    gt_pred = fp.readlines()
    fp.close()
    for i in range(len(gt_pred)):
        gt_pred[i] = gt_pred[i].strip().split("\t")
    gt_pred = np.asarray(gt_pred,dtype='int')
    acc = accuracy_score(gt_pred[:, 0],gt_pred[:,1])
    print('Accuracy %.4f\n\n\n'%(acc))

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        gt_pred_file = 'gt_pred'
    else:
        gt_pred_file = sys.argv[1]
    calculate_accuracy(gt_pred_file)