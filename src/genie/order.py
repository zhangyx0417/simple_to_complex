import json
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--dataset", type=str, default="KAIROS")
    args = parser.parse_args() 

    ex_confs = []
    doc_confs = []
    curr_doc_id = str()
    gen_file = "checkpoints/{}/predictions.jsonl".format(args.ckpt_name)

    with open(gen_file, "r") as f:
        for line in f:
            pred = json.loads(line.strip())
            doc_id = pred["doc_key"]
            args_confs = pred["confs"]

            if curr_doc_id != doc_id:
                curr_doc_id = doc_id
                if len(ex_confs) > 0:
                    doc_confs.append(ex_confs)
                    ex_confs = []

            first_confs = [arg_confs[0] for arg_confs in args_confs if len(arg_confs) > 0]
            if len(first_confs) > 0:
                ex_confs.append(sum(first_confs) / len(first_confs))
            else:
                ex_confs.append(-1)

        if len(ex_confs) > 0:
            doc_confs.append(ex_confs)

    # simple to complex ordering
    orderings = []
    curr_len = 0
    for ex_confs in doc_confs:
        orders = [order + curr_len for order in np.argsort(ex_confs).tolist()[::-1]]
        curr_len += len(orders)
        orderings.extend(orders)
    order_dict = {}
    for i, r in enumerate(orderings):
        order_dict[r] = i

    if os.path.exists("preprocessed_simtr_{}/ordered_test.jsonl".format(args.dataset)):
        os.remove("preprocessed_simtr_{}/ordered_test.jsonl".format(args.dataset))
    if os.path.exists("preprocessed_simtr_{}/order.jsonl".format(args.dataset)):
        os.remove("preprocessed_simtr_{}/order.jsonl".format(args.dataset))
    reader = open("preprocessed_simtr_{}/test.jsonl".format(args.dataset), "r")
    writer = open("preprocessed_simtr_{}/ordered_test.jsonl".format(args.dataset), "a")
    f = open("preprocessed_simtr_{}/order.jsonl".format(args.dataset), "a")
    test_examples = []
    for line in reader:
        test_example = json.loads(line.strip())
        test_examples.append(test_example)
    for i in range(len(order_dict)):
        writer.write(json.dumps(test_examples[order_dict[i]]) + "\n")
    f.write(json.dumps(order_dict))

    print("------Events are successfully ordered from simple to complex!------")
