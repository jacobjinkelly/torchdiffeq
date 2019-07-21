"""
parse logged results to see how often ODEFunc was called
"""
import parse
import numpy as np


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
    results = {"BATCHED": {},
               "SEPARATE": {"steps": [],
                            "fn_evals": []
                            }
               }
    assert lines.pop(0) == "BATCHED %s\n" % type
    steps, fn_evals = parse_line(lines.pop(0))
    results["BATCHED"]["steps"] = steps
    results["BATCHED"]["fn_evals"] = fn_evals
    assert lines.pop(0) == "SEPARATE %s\n" % type
    for i in range(batch_size):
        steps, fn_evals = parse_line(lines.pop(0))
        results["SEPARATE"]["steps"].append(steps)
        results["SEPARATE"]["fn_evals"].append(fn_evals)
    results["SEPARATE"]["steps"] = np.array(results["SEPARATE"]["steps"])
    results["SEPARATE"]["fn_evals"] = np.array(results["SEPARATE"]["fn_evals"])
    return results


def analyze_batch_results(batch):
    """
    Analyze formatted batch results.
    """
    # process
    batched_fn_evals = batch["BATCHED"]["fn_evals"]
    data = batch["SEPARATE"]["fn_evals"]
    hist = np.histogram(data, bins=np.arange(np.min(data), np.max(data) + 2))
    counts, bins = hist
    argsort = np.argsort(counts)
    counts = counts[argsort]
    bins = bins[argsort]

    # things to track

    # how many examples in batch require more/less fn evals then if batched together
    less_fn_evals = 0
    more_fn_evals = 0

    # min/max saved/additional (respectively) fn evals
    min_fn_evals = batched_fn_evals - np.min(data)
    max_fn_evals = np.max(data) - batched_fn_evals

    for bin, count in zip(bins, counts):
        if count > 0:
            if bin < batched_fn_evals:
                less_fn_evals += count
            if bin > batched_fn_evals:
                more_fn_evals += count
    return 100 * less_fn_evals / batch_size, 100 * more_fn_evals / batch_size, min_fn_evals, max_fn_evals


def analyize_epoch_result(result, type, epoch):
    """
    Print the analyzed epoch result.
    """
    more, less = [], []
    min_, max_ = [], []
    for batch_result in result:
        analyzed_batch_result = analyze_batch_results(batch_result)
        more.append(analyzed_batch_result[0])
        less.append(analyzed_batch_result[1])
        min_.append(analyzed_batch_result[2])
        max_.append(analyzed_batch_result[3])
    more, less, min_, max_ = np.array(more), np.array(less), np.array(min_), np.array(max_)

    print("{0:d}\t{1}{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}".format(
            epoch, type, np.mean(more), np.mean(min_), np.mean(less), np.mean(max_)))


if __name__ == "__main__":
    # SET THESE
    n_epochs = 4
    batch_size = 128
    test_batch_size = 1000
    train_size = 1000
    test_size = 1000

    # read in the file
    f = open("tmp.txt", "r")
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

    # simulate the code for easy parsing
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

    # analyze the results
    print("Columns:\n"
          "1. Epoch\n"
          "2. FORWARD or BACKWARD pass\n"
          "3. Average % of Batch Requiring More Function Evaluations\n"
          "4. Average Number of Max Additional Function Evaluations\n"
          "5. Average % of Batch Requiring Less Function Evaluations\n"
          "6. Average Number of Min Saved Function Evaluations")
    print("1\t2\t\t3\t4\t5\t6")
    for epoch, epoch_result in zip(range(len(results)), results):
        # only look at batches for now, this is what actually matters for efficiency
        analyize_epoch_result(epoch_result["batches"]["FORWARD"], "FORWARD\t\t", epoch)
        analyize_epoch_result(epoch_result["batches"]["BACKWARD"], "BACKWARD\t", epoch)
