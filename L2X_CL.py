from Model_Core import *
from ClusterEval import *
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])


class l2x_cl:
    def __init__(self, SERE_Path, CL_Path, num_class, learning_rate=0.01):
        self.num_class = num_class
        self.SERE = L2X()
        self.CL = LR(self.num_class)
        self.Checkpoint_SERE = tf.train.Checkpoint(SERE=self.SERE)
        self.Checkpoint_CL = tf.train.Checkpoint(CL=self.CL)
        self.WeightManager_SERE = tf.train.CheckpointManager(self.Checkpoint_SERE, SERE_Path, max_to_keep=2)
        self.WeightManager_CL = tf.train.CheckpointManager(self.Checkpoint_CL, CL_Path, max_to_keep=2)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.TRAIN = False

    def compute_loss(self, label_true, label_pred):
        if self.num_class > 2:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label_true, label_pred))
        else:
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label_true, label_pred))
        return loss

    def train_step(self, data_x, label_true, imp_row_count, row_count):
        with tf.GradientTape() as tape:
            l2x_encoding, _, _ = self.SERE(data_x, imp_row_count, row_count, self.TRAIN)
            label_pred = self.CL(l2x_encoding)
            variable_list = self.CL.trainable_variables + self.SERE.trainable_variables
            loss_class = self.compute_loss(label_true, label_pred)
            avg_acc = find_accuracy(label_pred, label_true)
            grads = tape.gradient(loss_class, variable_list)
            self.optimizer.apply_gradients(zip(grads, variable_list))
        return loss_class, avg_acc

    def validation(self, data_x, label_true, imp_row_count, row_count):
        l2x_encoding, _, _ = self.SERE(data_x, imp_row_count, row_count, True)
        label_pred = self.CL(l2x_encoding)
        avg_acc = find_accuracy(label_pred, label_true)
        return avg_acc

    def train(self, dataset_train, dataset_test, epochs, tao):
        print("Training started. Train size %d, Test size %d\n\n"%(len(dataset_train[0]),len(dataset_test[0])))
        BATCHSIZE = Config.BATCHSIZE
        train_batch_gen = get_batches(dataset_train, BATCHSIZE, tao)
        test_batch_gen = get_batches(dataset_test, BATCHSIZE, tao)
        train_len = len(dataset_train[0])
        nbbatches = int(np.ceil(float(train_len) / BATCHSIZE))
        nbbatches_test = int(np.ceil(float(len(dataset_test[0])) / BATCHSIZE))
        best_val_epoch_acc = 0
        for j in range(nbbatches_test):
            batch_vec, batch_p, batch_mask, start, end, imp_row_count, row_count = next(test_batch_gen)
            b_acc = self.validation(batch_vec, batch_p, imp_row_count, row_count)
            best_val_epoch_acc += b_acc
        best_val_epoch_acc /= float(nbbatches_test)
        for i in range(1, epochs):
            epoch_loss = 0
            epoch_acc = 0
            for j in range(nbbatches):
                batch_x_vec, batch_p, batch_mask, start, end, imp_row_count, row_count = next(train_batch_gen)
                b_x = batch_x_vec
                b_loss, b_acc = self.train_step(b_x, batch_p, imp_row_count, row_count)
                epoch_loss += b_loss
                epoch_acc += b_acc
            epoch_loss /= float(nbbatches)
            epoch_acc /= float(nbbatches)
            val_epoch_acc = 0
            for j in range(nbbatches_test):
                batch_vec, batch_p, batch_mask, start, end, imp_row_count, row_count = next(test_batch_gen)
                b_acc = self.validation(batch_vec, batch_p, imp_row_count, row_count)
                val_epoch_acc += b_acc
            val_epoch_acc /= float(nbbatches_test)
            if val_epoch_acc > best_val_epoch_acc:
                best_val_epoch_acc = val_epoch_acc
                print("Weight saved in %s %s" % (
                self.WeightManager_SERE.latest_checkpoint, self.WeightManager_CL.latest_checkpoint))
                self.WeightManager_SERE.save()
                self.WeightManager_CL.save()
            msg = "\tEpoch %d/%d L2X defence Loss %0.4f Primary task Training Acc %0.3f Val Acc %0.4f Best Val Acc %0.3f"%\
                  (i, epochs, epoch_loss,epoch_acc,val_epoch_acc,best_val_epoch_acc)
            print(msg)
            gc.collect()

    def load_model(self, SERE_Path, CL_Path):
        sere_path = tf.train.latest_checkpoint(SERE_Path)
        cl_path = tf.train.latest_checkpoint(CL_Path)
        self.Checkpoint_SERE.restore(sere_path)
        self.Checkpoint_CL.restore(cl_path)

    def normalize(self,array, mask):
        lb = 0.1
        ub = 1
        inf = 1000000
        ninf = -1000000
        row_min = tf.reduce_min(tf.where(tf.equal(mask, 1), array, inf), axis=-1, keepdims=True)
        row_max = tf.reduce_max(tf.where(tf.equal(mask, 1), array, ninf), axis=-1, keepdims=True)
        denom = row_min - row_max
        denom = tf.where(tf.equal(denom, 0), 1, denom)
        numr = tf.math.subtract(array, row_min)
        array = lb + (lb - ub) * (numr / denom)
        array = tf.where(tf.equal(mask, True), array, 0)
        return array

    def predict(self, batchid, x, labels_p, imp_row_count, row_count):
        l2x_encoding, T, priority_weight = self.SERE(x, imp_row_count, row_count, True)
        class_prediction = self.CL(l2x_encoding)
        return class_prediction, l2x_encoding

    def evaluate(self, dataset, tao=None, vec_file='data/vecs.txt'):
        batch_size = Config.BATCHSIZE
        batch_gen = get_batches(dataset, batch_size, tao)
        total = len(dataset[0])
        nbbatches = int(np.ceil(float(total) / batch_size))
        total_acc = 0
        vecs = None
        for j in range(nbbatches):
            batch_x_vec, batch_p, batch_mask, start, end, imp_row_count, row_count = next(batch_gen)
            prediction, l2x_encoding = self.predict(j, batch_x_vec, batch_p, imp_row_count, row_count)
            b_acc = find_accuracy(prediction, batch_p)
            total_acc += b_acc
            if j == 0:
                vecs = l2x_encoding
            else:
                vecs = np.concatenate((vecs, np.asarray(l2x_encoding)), axis=0)
        writeVectors(vecs, vec_file=vec_file)
        total_acc /= float(nbbatches)
        print("Accuracy %0.3f"%(total_acc))
        return total_acc
