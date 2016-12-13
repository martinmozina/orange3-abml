import numpy as np
from Orange.evaluation import CrossValidation
from Orange.data import Domain, Table, DiscreteVariable

def find_critical(ablearner, rule_learner, data, k=5, threshold = 0.9):
    """
    :param ablearner: argument-based learner to be tested
    :param rule_learner: a rule learner
    :param data: learning data
    :param k: folds in cross-validation
    :param threshold: quality threshold to determine whether rule is good or bad
    :return: how critical each instance in data is
    """
    # first get how problematic is each example (cross-validation)
    # E ... the difference between probability of predicted most probable class
    # and the probability of the example's class.
    # if example is correctly predicted or if example is already covered
    # by an argumented rule, E equals 0.
    # CV
    res = CrossValidation(data, [ablearner], k=k)
    # get error measure for each instance
    prob = np.empty(res.probabilities[0].shape)
    prob[res.row_indices] = res.probabilities[0]
    actual = np.empty(res.actual.shape, dtype=int)
    actual[res.row_indices] = res.actual.astype(int)
    mprob = np.amax(prob, axis=1)
    predprob = np.choose(actual, prob.T)
    problematic = mprob - predprob

    # get examples covered by argumented rules or by rules that have no
    # negative examples or by rules that have quality higher than
    # threshold
    rules = rule_learner(data).rule_list
    arg_covered = np.zeros(data.X.shape[0], dtype=bool)
    for r in rules:
        if r.default_rule.length > 0 or \
                r.curr_class_dist[r.target_class] == r.curr_class_dist.sum() or \
                r.quality > threshold:
            arg_covered |= r.covered_examples
    problematic *= 1 - arg_covered

    # then learn rules to distinguish between problematic and non-problematic
    # we would like to find groups of similar problematic examples
    # PC ... predicted probability that example is problematic.
    y = np.array(problematic > 0).astype(int)
    new_domain = Domain(data.domain.attributes, DiscreteVariable('prob', values=("0","1")))
    new_data = Table.from_table(new_domain, data)
    new_data.Y = y

    # learn rules for target_class=1 only
    tc = rule_learner.target_class
    rule_learner.target_class = 1
    rules = rule_learner(new_data).rule_list
    explainable = np.zeros(data.X.shape[0])
    for r in rules:
        explainable = np.maximum(explainable, r.covered_examples * r.quality)
    rule_learner.target_class = tc

    # return how critical each example is: C = E * PC
    critical = explainable * problematic
    return critical, np.argsort(critical)

