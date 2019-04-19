import numpy as np
import part2_data
from copy import copy
import frame
import sys
import pickle

if __name__ == "__main__":

    nn_architecture = [
        {"input_dim": 500, "output_dim": int(sys.argv[1]), "activation": "sigmoid"},
        {"input_dim": int(sys.argv[1]), "output_dim": int(sys.argv[2]), "activation": "sigmoid"},
        {"input_dim": int(sys.argv[2]), "output_dim": 2, "activation": "softmax"},
    ]
    # global nn_architecture
    print("Would you like to load the previous model?\n1) Yes\n2) No.")
    a = input("Enter choice: ")
    print("Enter no of epochs (give less if old model loaded): ")
    ep = int(input())
    train_batch, test_batch = part2_data.prep_data()
    if (a == '1'):
        f = open("model1.pickle", "rb")
        param_values = pickle.load(f)
        f.close()
        pv, ch, ah = frame.train_2(train_batch, nn_architecture, ep, 0.1, param_values)
    else:
        pv, ch, ah = frame.train_2(train_batch, nn_architecture, ep, 0.1)

    f = open("model1.pickle", "wb")
    pickle.dump(pv, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    ch_test, ah_test = frame.test_2(test_batch, nn_architecture, pv)
    print("Train acc: ", np.sum(ah)/len(ah))
    print("Test acc: ", np.sum(ah_test)/len(ah_test))
    # print(ch_test)
    # pu.db
    # infer(train_batch)