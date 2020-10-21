from L2X_CL import *

tf.random.set_seed(1236)
np.random.seed(1234)


def load_data():
    global num_class, train_data, train_label, test_data, test_label
    path_vec = "data/vecs.txt"
    path_p = "data/p.txt"
    vec = loadInputVec(path_vec)
    DATABASE_SIZE = len(vec)
    SPLIT_INDEX = math.ceil(DATABASE_SIZE * TRAIN_TEST_DIV)
    dataset = np.asarray(vec, dtype="float32")
    dataset = dataset.reshape((DATABASE_SIZE, -1, MAX_ROW_LEN))
    label, num_class = load_labels(path_p)
    target = np.asarray(make_onehot(label, num_class), dtype="float32")
    train_data = dataset[0:SPLIT_INDEX]
    train_label = target[0:SPLIT_INDEX]
    test_data = dataset[SPLIT_INDEX:DATABASE_SIZE]
    test_label = target[SPLIT_INDEX:DATABASE_SIZE]


def train(SERE_PATH, CL_PATH, load=False, epochs=10, learning_rate=0.01):
    l2x_model = l2x_cl(SERE_PATH, CL_PATH, num_class, learning_rate=learning_rate)
    l2x_model.TRAIN = True
    if load:  # load pretrained weights
        l2x_model.load_model(SERE_PATH, CL_PATH)
    l2x_model.train([train_data, train_label], [test_data, test_label], epochs=epochs, tao=TAO)


def test(SERE_PATH, CL_PATH, vec_file='data/vecs.txt'):
    l2x_model = l2x_cl(SERE_PATH, CL_PATH, num_class)
    l2x_model.TRAIN = False
    l2x_model.load_model(SERE_PATH, CL_PATH)
    l2x_model.evaluate([test_data, test_label], tao=TAO, vec_file=vec_file)


if __name__ == "__main__":
    global TRAIN_TEST_DIV
    global BITA
    global TAO
    global MAX_ROW_LEN
    if len(sys.argv) == 1:
        TRAIN_TEST_DIV = 0.857142857
        TAO = 0.6
        MAX_ROW_LEN = 1
        Config.SE_DENSE_DIM = 8 * MAX_ROW_LEN
        Config.SE_CNN_FILTER = 8 * MAX_ROW_LEN
        copyfile('data/mnist_slant_frac_vecs.txt', 'data/vecs.txt')
        copyfile('data/mnist_slant_frac_p.txt', 'data/p.txt')
        load_data()
        sere_path = "Weights/SERE/"
        cl_path = "Weights/CL/"
        train(sere_path, cl_path, load=True)
        test(sere_path, cl_path)
    else:
        TRAIN_TEST_DIV = float(sys.argv[2])
        TAO = float(sys.argv[3])                           # Fraction of words to retain
        vec_file = sys.argv[4]
        MAX_ROW_LEN = int(sys.argv[5])
        Config.SE_DENSE_DIM = 8 * MAX_ROW_LEN
        Config.SE_CNN_FILTER = 8 * MAX_ROW_LEN
        sere_path = "Weights/SERE/"
        cl_path = "Weights/CL/"
        load_data()
        if sys.argv[1] == 'train':
            print(sere_path, cl_path)
            epochs = int(sys.argv[6])
            learning_rate = float(sys.argv[7])
            train(sere_path, cl_path, load=True, epochs=epochs, learning_rate=learning_rate)
        else:
            print(sere_path, cl_path)
            test(sere_path, cl_path, vec_file=vec_file)
