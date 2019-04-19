import numpy as np
import part2_data
from copy import copy
import pudb
import frame


nn_architecture = [
    {"input_dim": 500, "output_dim": 100, "activation": "sigmoid"},
    {"input_dim": 100, "output_dim": 100, "activation": "sigmoid"},
    {"input_dim": 100, "output_dim": 2, "activation": "softmax"},
]


if __name__ == "__main__":
    # global nn_architecture
    train_batch, test_batch = part2_data.prep_data()
    pv, ch, ah = frame.train_2(train_batch, nn_architecture, 10, 0.1)
    ch_test, ah_test = frame.test_2(test_batch, nn_architecture, pv)
    pu.db
    # infer(train_batch)
    # pu.db
