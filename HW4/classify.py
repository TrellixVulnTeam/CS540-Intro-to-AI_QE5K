import math
import os


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """

    bow = {}
    num_none = 0
    with open(filepath, encoding="utf-8") as file_object:
        for line in file_object:
            line = line.rstrip()
            if line in vocab:
                if line not in bow:
                    bow[line] = 0
                bow[line] += 1
            else:
                num_none += 1
    if num_none:
        bow[None] = num_none
    return bow


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """

    dataset = []
    dir_partials = ["2016", "2020"]
    for dir_partial in dir_partials:
        for file in os.listdir(os.path.join(directory, dir_partial)):
            dataset.append({"label": dir_partial, "bow": create_bow(vocab, os.path.join(directory, dir_partial, file))})
    return dataset


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    v_num = {}
    vocab = set()
    dir_partials = ["2016", "2020"]
    for dir_partial in dir_partials:
        for file in os.listdir(os.path.join(directory, dir_partial)):
            with open(os.path.join(directory, dir_partial, file), encoding="utf-8") as file_object:
                for line in file_object:
                    line = line.rstrip()
                    if line not in vocab:
                        if line not in v_num:
                            v_num[line] = 0
                        v_num[line] += 1
                        if v_num[line] >= cutoff:
                            vocab.add(line)
    return sorted(list(vocab))


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    nums = {"2020": 0, "2016": 0}
    num_total = len(training_data)
    for data in training_data:
        if data["label"] == "2016":
            nums["2016"] += 1
        else:
            nums["2020"] += 1
    for label in label_list:
        logprob[label] = math.log(nums[label] + smooth) - math.log(num_total + len(label_list) * smooth)
    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    cop_vocab = vocab[:]
    cop_vocab.append(None)
    word_prob = {}
    total_num_label = 0
    for data in training_data:
        if data["label"] == label:
            for k, v in data["bow"].items():
                if k not in word_prob:
                    word_prob[k] = 0
                word_prob[k] += v
                total_num_label += v

    for voc in cop_vocab:
        if voc in word_prob:
            word_prob[voc] = math.log(word_prob[voc] + smooth) - math.log(total_num_label + smooth * len(cop_vocab))
        else:
            word_prob[voc] = math.log(smooth) - math.log(total_num_label + smooth * len(cop_vocab))
    return word_prob


##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """

    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    return {
        'vocabulary': vocab,
        'log prior': prior(training_data, ["2020", "2016"]),
        'log p(w|y=2016)': p_word_given_label(vocab, training_data, "2016"),
        'log p(w|y=2020)': p_word_given_label(vocab, training_data, "2020")
    }


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """

    retval = {
        'log p(y=2016|x)': model["log prior"]["2016"],
        'log p(y=2020|x)': model["log prior"]["2020"]}
    model_2016 = model["log p(w|y=2016)"]
    model_2020 = model["log p(w|y=2020)"]
    with open(filepath, encoding="utf-8") as file_object:
        for line in file_object:
            line = line.rstrip()
            retval["log p(y=2016|x)"] += model_2016[line] if line in model_2016 else model_2016[None]
            retval["log p(y=2020|x)"] += model_2020[line] if line in model_2020 else model_2020[None]
    retval["predicted y"] = "2016" if retval["log p(y=2016|x)"] > retval["log p(y=2020|x)"] else "2020"
    return retval
