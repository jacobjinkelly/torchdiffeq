"""
parse logged results to see how often ODEFunc was called
"""
import parse


def parse_line(line):
    """
    Parse a line of the form: "Steps: %d Fn Evals: %d\n"
    """
    format_string = "Steps: {:d} Fn Evals: {:d}"
    return tuple(parse.parse(format_string, line))


def parse_batch(batch_size, type):
    """
    Parse the results from a batch. Type is FORWARD or BACKWARD.
    """
    results = {"BATCHED": [],
               "SEPARATE": []}
    assert lines.pop(0) == "BATCHED %s\n" % type
    results["BATCHED"].append(parse_line(lines.pop(0)))
    assert lines.pop(0) == "SEPARATE %s\n" % type
    for i in range(batch_size):
        results["SEPARATE"].append(parse_line(lines.pop(0)))
    return results


if __name__ == "__main__":
    # SET THESE
    n_epochs = 1
    batch_size = 128
    test_batch_size = 1000
    train_size = 1000
    test_size = 1000

    # read in the file
    f = open("results.txt", "r")
    lines = f.readlines()

    # clean the file
    ind = -1
    for i in range(len(lines)):
        if lines[i].startswith("Number of parameters"):
            assert ind == -1
            ind = i + 1
    # remove logging of code.
    lines = lines[ind:]

    # remove training reports
    c = 0
    for line in lines:
        if line.startswith("Epoch"):
            lines.remove(line)
            c += 1
    assert c == n_epochs

    results = []

    # simulate the code
    batches_per_epoch = train_size // batch_size
    for i in range(n_epochs * batches_per_epoch):
        curr_epoch = i // batches_per_epoch
        curr_batch = i - curr_epoch * batches_per_epoch

        if len(results) != curr_epoch + 1:
            results.append({
                "batches": {"FORWARD": [],
                            "BACKWARD": []
                            },
            })

        # get forward pass, batched and separate
        results[curr_epoch]["batches"]["FORWARD"].append(parse_batch(batch_size, "FORWARD"))

        # get backward pass, batched and separate
        results[curr_epoch]["batches"]["BACKWARD"].append(parse_batch(batch_size, "BACKWARD"))

        if i % batches_per_epoch == 0:
            assert lines.pop(0) == "train acc\n"
            for j in range(train_size // test_batch_size):
                results[curr_epoch]["train_acc"] = parse_batch(test_batch_size, "FORWARD")

            assert lines.pop(0) == "val acc\n"
            for j in range(test_size // test_batch_size):
                results[curr_epoch]["val_acc"] = parse_batch(test_batch_size, "FORWARD")

    print("hello there")
