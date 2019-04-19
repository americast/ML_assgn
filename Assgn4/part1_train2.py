import numpy as np
import part1_data
from copy import copy
import pudb
import frame


nn_architecture = [
    {"input_dim": 500, "output_dim": 100, "activation": "relu"},
    {"input_dim": 100, "output_dim": 1, "activation": "sigmoid"},
]


if __name__ == "__main__":
    # global nn_architecture
    train_batch, test_batch = part1_data.prep_data()
    pv, ch, ah = frame.train(train_batch, nn_architecture, 10, 0.1)
    ch_test, ah_test = frame.test(test_batch, nn_architecture, pv)
    pu.db
    # infer(train_batch)
    # pu.db
