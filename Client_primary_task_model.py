from Model_Core import *
from ClusterEval import *

tf.random.set_seed(1236)
np.random.seed(1234)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3096)])


def batch_gen(dataset, labels, batch_size=32):
    total = len(dataset)
    start = 0
    while 1:
        end = min(start + batch_size, total)
        batch_x = dataset[start:end]
        batch_labels = labels[start:end]
        yield batch_x, batch_labels
        start = end
        if (start >= total):
            start = 0


class Client:
    def __init__(self, num_class):
        self.num_class = num_class
        self.CL = LR(self.num_class)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            class_prediction = self.CL(data)
            variable_list = self.CL.trainable_variables
            if self.num_class > 2:
                loss_class = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, class_prediction))
            else:
                loss_class = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, class_prediction))
            avg_acc = find_accuracy(class_prediction, labels)
            grads = tape.gradient(loss_class, variable_list)
            self.optimizer.apply_gradients(zip(grads, variable_list))
        return loss_class, avg_acc

    def train(self, dataset, labels, epochs, batch_size, path_gt_pred='gt_pred'):
        dataset = np.asarray(dataset, dtype="float32")
        train_data = dataset[0:SPLIT_INDEX]
        train_labels = labels[0:SPLIT_INDEX]
        test_data = dataset[SPLIT_INDEX:DATABASE_SIZE]
        test_labels = labels[SPLIT_INDEX:DATABASE_SIZE]
        train_batch_gen = batch_gen(train_data, train_labels, batch_size)
        train_len = len(train_data)
        nbbatches = int(np.ceil(float(train_len) / batch_size))
        best_test_acc = 0
        no_improvement_count = 0
        print("\n\n\nTrain database size %d Test database size %d" % (SPLIT_INDEX, DATABASE_SIZE - SPLIT_INDEX))
        for i in range(1, epochs):
            epoch_loss = 0
            epoch_acc = 0
            for j in range(nbbatches):
                batch_x_vec, batch_y_age = next(train_batch_gen)
                b_x = batch_x_vec
                b_y = batch_y_age
                b_loss, b_acc = self.train_step(b_x, b_y)
                epoch_loss += b_loss
                epoch_acc += b_acc
            epoch_loss /= float(nbbatches)
            epoch_acc /= float(nbbatches)

            test_prediction = self.CL(test_data)
            test_acc = find_accuracy(test_prediction, test_labels)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_prediction = test_prediction
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if i % 100 == 0:
                msg = "\tEpoch %d/%d Loss %0.4f primary task p train Acc %0.4f test Acc %0.4f" % (
                i, epochs, epoch_loss, epoch_acc, test_acc)
                print(msg)
        write_gt_pred(tf.argmax(test_labels, axis=-1), tf.argmax(best_test_prediction, axis=-1), file_name=path_gt_pred)


if __name__ == "__main__":
    global DATABASE_SIZE, SPLIT_INDEX
    path_vec = "data/vecs.txt"
    path_p = "data/p.txt"
    path_gt_pred = sys.argv[1]
    vec = loadInputVec(path_vec)
    DATABASE_SIZE = len(vec)
    SPLIT_INDEX = math.ceil(DATABASE_SIZE * Config.TRAIN_TEST_DIV)
    label, num_class = load_labels(path_p)
    target = make_onehot(label, num_class)
    client_model = Client(num_class)
    client_model.train(vec, target, epochs=1000, batch_size=SPLIT_INDEX, path_gt_pred=path_gt_pred)
