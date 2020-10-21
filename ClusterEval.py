from mnist_data_helper import *

def find_accuracy(predictions, truth, one_hot=True):
    if(one_hot):
        truth_labels = tf.argmax(truth,axis=-1)
        pred_labels = tf.argmax(predictions,axis=-1)
    else:
        truth_labels = truth
        pred_labels = predictions
    corrects = tf.cast(tf.equal(pred_labels,truth_labels),tf.float32)
    acc = tf.reduce_mean(corrects)
    return acc.numpy()

def write_selected_words(y_age=None, z_gen=None, labels_sent=None, selected_words_text=None,
                  one_hot=True, out_file='predictions.txt'):
    if(one_hot):
        truth_labels = tf.argmax(labels_sent,axis=-1)
    else:
        truth_labels = labels_sent
    truth_labels = truth_labels.numpy()#N,1
    f=open(out_file,'a')
    for tl in range(len(truth_labels)):
        if len(selected_words_text) > 0:
            if len(selected_words_text[tl]) > 0:
                f.write("%d\t%d\t%s\t%d\t%d\n"%(int(tf.argmax(y_age[tl])),int(tf.argmax(z_gen[tl])),' '.join(selected_words_text[tl]),
                                                int(tf.argmax(labels_sent[tl])),len(selected_words_text[tl])))
    f.close()

def write_gt_pred(gt, pred, file_name='gt_pred'):
    gt = gt.numpy()  # N,1
    f = open(file_name, 'w')
    for tl in range(len(gt)):
        f.write("%d\t%d\n" % (gt[tl],pred[tl]))
    f.close()


