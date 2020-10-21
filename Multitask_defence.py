from Model_Core import *
from ClusterEval import *
from mnist_data_helper import *

tf.random.set_seed(1236)
np.random.seed(1234)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3096)])


def batch_gen(dataset, p_labels, s_labels, s1_labels, batch_size=32):
    total = len(dataset)
    start = 0
    while 1:
        end = min(start + batch_size, total)
        batch_x = dataset[start:end]
        batch_p_labels = p_labels[start:end]
        batch_s_labels = s_labels[start:end]
        batch_s1_labels = s1_labels[start:end]
        yield batch_x, batch_p_labels, batch_s_labels, batch_s1_labels
        start = end
        if start >= total:
            start = 0


class multitask_defence:
    def __init__(self, num_class_p, num_class_s, num_class_s1):
        self.MID_DIM = 300
        self.num_class_p = num_class_p
        self.num_class_s = num_class_s
        self.num_class_s1 = num_class_s1
        self.CL_p = LR(self.num_class_p)
        self.CL_s = LR(self.num_class_s)
        self.CL_s1 = LR(self.num_class_s1)
        self.shared_layer = tf.keras.layers.Dense(self.MID_DIM, activation='tanh', name='dense_1')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    def compute_loss(self, label_true, label_pred, num_class):
        if num_class > 2:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label_true, label_pred))
        else:
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label_true, label_pred))
        return loss

    def train_step(self, data, label_p, label_s, label_s1, gamma, beta):
        with tf.GradientTape() as tape:
            shared_out = self.shared_layer(data)
            pred_p = self.CL_p(shared_out)
            pred_s = self.CL_s(shared_out)
            pred_s1 = self.CL_s1(shared_out)
            variable_list = self.shared_layer.trainable_variables + self.CL_p.trainable_variables + \
                            self.CL_s.trainable_variables + self.CL_s1.trainable_variables
            loss_p = self.compute_loss(label_p, pred_p, self.num_class_p)
            loss_s = self.compute_loss(label_s, pred_s, self.num_class_s)
            loss_s1 = self.compute_loss(label_s1, pred_s1, self.num_class_s1)
            joint_loss = (1 - gamma - beta) * loss_p + gamma * loss_s + beta * loss_s1
            avg_acc_p = find_accuracy(pred_p, label_p)
            grads = tape.gradient(joint_loss, variable_list)
            self.optimizer.apply_gradients(zip(grads, variable_list))
        return loss_p, avg_acc_p, shared_out

    def train(self, dataset, label_p, label_s, label_s1, epochs, batch_size, gamma, beta):
        dataset = np.asarray(dataset, dtype="float32")
        label_p = np.asarray(label_p, dtype="float32")
        label_s = np.asarray(label_s, dtype="float32")
        label_s1 = np.asarray(label_s1, dtype="float32")
        train_data = dataset[0:SPLIT_INDEX]
        train_label_p = label_p[0:SPLIT_INDEX]
        train_label_s = label_s[0:SPLIT_INDEX]
        train_label_s1 = label_s1[0:SPLIT_INDEX]
        test_data = dataset[SPLIT_INDEX:DATABASE_SIZE]
        p_test_labels = label_p[SPLIT_INDEX:DATABASE_SIZE]
        train_batch_gen = batch_gen(train_data, train_label_p, train_label_s, train_label_s1, batch_size)
        train_len = len(train_data)
        nbbatches = int(np.ceil(float(train_len) / batch_size))
        best_test_acc = 0
        best_shared_out_train = self.shared_layer(train_data)
        best_shared_out_test = self.shared_layer(test_data)
        print("\n\n\nTrain database size %d Test database size %d" % (SPLIT_INDEX, DATABASE_SIZE - SPLIT_INDEX))
        for i in range(1, epochs):
            epoch_loss = 0
            epoch_acc = 0
            for j in range(nbbatches):
                batch_vec, batch_p, batch_s, batch_s1 = next(train_batch_gen)
                b_loss, b_acc, shared_out_train = self.train_step(batch_vec, batch_p, batch_s, batch_s1, gamma, beta)
                epoch_loss += b_loss
                epoch_acc += b_acc
            epoch_loss /= float(nbbatches)
            epoch_acc /= float(nbbatches)

            shared_out_test = self.shared_layer(test_data)
            test_prediction = self.CL_p(shared_out_test)
            test_acc = find_accuracy(test_prediction, p_test_labels)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_shared_out_train = shared_out_train
                best_shared_out_test = shared_out_test

            if i % 100 == 0:
                msg = "\tEpoch %d/%d Loss %0.4f Multitask defence train Acc on primary task %0.4f test Acc on primary task %0.4f" % (i, epochs, epoch_loss, epoch_acc, test_acc)
                print(msg)

            gc.collect()
        vecs = np.concatenate((np.asarray(best_shared_out_train), np.asarray(best_shared_out_test)), axis=0)
        writeVectors(vecs)


if __name__ == "__main__":
    global DATABASE_SIZE, SPLIT_INDEX
    path_vec = "data/vecs.txt"
    path_p = "data/p.txt"
    path_s1 = "data/s1.txt"
    path_s2 = "data/s2.txt"

    if len(sys.argv) == 1:
        GAMMA = 0.2
        BETA = 0.2
    else:
        GAMMA = float(sys.argv[1])
        BETA = float(sys.argv[2])
    vec = loadInputVec(path_vec)
    DATABASE_SIZE = len(vec)
    SPLIT_INDEX = math.ceil(DATABASE_SIZE * Config.TRAIN_TEST_DIV)
    label_p, num_class_p = load_labels(path_p)
    target_p = make_onehot(label_p, num_class_p)
    label_s, num_class_s = load_labels(path_s1)
    label_s = np.random.randint(num_class_s, size=DATABASE_SIZE)
    target_s = make_onehot(label_s, num_class_s)
    label_s1, num_class_s1 = load_labels(path_s2)
    label_s1 = np.random.randint(num_class_s1, size=DATABASE_SIZE)
    target_s1 = make_onehot(label_s1, num_class_s1)
    client_model = multitask_defence(num_class_p, num_class_s, num_class_s1)
    client_model.train(vec, target_p, target_s, target_s1, epochs=1000, batch_size=SPLIT_INDEX, gamma=GAMMA, beta=BETA)
