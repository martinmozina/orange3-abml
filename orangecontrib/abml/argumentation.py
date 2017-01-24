import numpy as np
import Orange
from Orange.data import Table
from Orange.classification.rules import Selector
from sklearn.model_selection import StratifiedKFold
from Orange import distance
from Orange.clustering import hierarchical

ARGUMENTS = "Arguments"

def cluster(X, n):
    """ Cluster X into n clusters and return the indices of most typical instance
    for each cluster. Hierarchical clustering with Ward function for 
    linkage is used. """
    dist_matrix = distance.Euclidean(X)
    c_clusters = min(n, X.shape[0])
    hierar = hierarchical.HierarchicalClustering(n_clusters=c_clusters)
    hierar.linkage = hierarchical.WARD
    hierar.fit(dist_matrix)
    lab = hierar.labels

    centroids = []
    for i in range(c_clusters):
        cl_ind = lab == i
        cl_pos = np.where(cl_ind)[0]
        dist = dist_matrix[cl_ind, :][:, cl_ind]
        most_repr = dist.sum(axis=0).argmin()
        centroids.append(cl_pos[most_repr])
    return centroids

def rf(rule):
    """ Classification accuracy of rule measured with relative frequency. """
    dist = rule.curr_class_dist
    return dist[rule.target_class] / dist.sum()


def find_critical(learner, data, n=5, k=5, threshold = 0.6,
                  rule_threshold = 0.9, random_state=0):
    """
    :param learner: argument-based learner to be tested
    :param data: learning data
    :param n: number of critical examples
    :param k: folds in cross-validation
    :param threshold: predicted probability threshold when example can not be
                      critical anymore
    :param rule_threshold: quality threshold to determine whether rule is good or bad
    :param random_state: random state to be used in StratifiedKFold function
    :return: n most critical examples (with estimation of 'criticality')
    """
    # first get how problematic is each example (cross-validation)
    # E ... the difference between probability of predicted most probable class
    # and the probability of the example's class.
    # if example is correctly predicted or if example is already covered
    # by an argumented rule, E equals 0.
    # CV
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    problematic = np.zeros(len(data))
    problematic_rules = [[] for d in data]
    for learn_ind, test_ind in skf.split(data.X, data.Y):
        learn = Table(data.domain, data[learn_ind])
        test = Table(data.domain, data[test_ind])

        classifier = learner(learn)
        rules = classifier.rule_list
        # eval rules on test data
        cov = classifier.coverage(test)
        # for each test example find out whether it is covered by a pure rule
        # or a rule with quality higher than threshold.
        covered = np.zeros(len(test), dtype=bool)
        for ri, r in enumerate(rules):
            if r.quality >= rule_threshold or \
                    r.curr_class_dist[r.target_class] == r.curr_class_dist.sum():
                target = r.target_class == test.Y
                covered |= cov[:,ri] & target

        probs = classifier(test, 1)
        for ti, t in enumerate(test_ind):
            # first check best rule, if same class, it can not be problematic
            d, p = test[ti], probs[ti]
            c = int(d.get_class())
            if d[ARGUMENTS] in ("", "?") and not covered[ti] and p[c] < threshold:
                problematic[t] = 1 - p[c]
                problematic_rules[t] = [r for ri, r in enumerate(rules) if cov[ti, ri]]

    # cluster problematic examples and select n most critical examples
    prob = problematic > 0
    indices = np.where(prob)[0]
    X = data.X[prob]
    dist_matrix = distance.Euclidean(X)
    c_clusters = min(n, X.shape[0]//5 + 1)
    hierar = hierarchical.HierarchicalClustering(n_clusters=c_clusters)
    hierar.linkage = hierarchical.WARD
    hierar.fit(dist_matrix)
    lab = hierar.labels

    critical = []
    for i in range(c_clusters):
        cl_ind = lab == i
        cl_pos = np.where(cl_ind)[0]
        dist = dist_matrix[cl_ind, :][:, cl_ind]
        most_repr = dist.sum(axis=0).argmin()
        critical.append(indices[cl_pos[most_repr]])

    return critical, problematic[critical], [problematic_rules[i] for i in critical]

def analyze_argument(learner, data, index, n=5):
    """
    Analysing argumented example consists of finding counter examples,
    suggesting "safe" or "consistent" conditions and argument pruning.

    :param learner: argument-based learner to be tested
    :param data: learning data
    :param index: index of argumented example
    :param n: number of counter examples to be returned
    :return: n counter examples, suggested conditions, results of pruning rule
        that was learned from argumented example.
    """

    # run 5-times repeated cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    counters = []
    learned = []
    for learn, test in skf.split(data.X, data.Y):
        if index not in learn:
            continue
        learndata = Table(data.domain, data[learn])
        testdata = Table(data.domain, data[test])
        new_index = np.where(learn == index) # new argumented example position

        learner.target_instances = [new_index]
        rules = learner(learndata).rule_list

        if not rules: 
            # rules was not learned from argumented example ...
            # loop can be continued from here
            continue

        # learner should learn exactly one rule
        assert len(rules) == 1
        rule = rules[0]
        learned.append(rule) # append rule

        # counter examples are examples covered by rule in testdata and
        # learndata, but from the opposite class
        test_cov = rule.evaluate_data(testdata.X)
        test_counter = test_cov & (testdata.Y != rule.target_class)
        counters += list(test[test_counter])


    counterX = data.X[counters]
    centroids = cluster(counterX, n)
    counters = np.array(counters)[centroids]

    # find common conditions in learned
    if len(learned) < n-1:
        # not all rules were learned
        selectors = []
    else:
        selectors = set.intersection(*[set((s.column, s.op, s.value) for s in r.selectors) for r in learned])
        selectors = [Selector(column=col, op=op, value=value) for col, op, value in selectors]
    # create a rule from common selectors
    rule.selectors = selectors
    X, Y, W = data.X, data.Y.astype(dtype=int), \
              data.W if data.W else None
    rule.filter_and_store(X, Y, W, rule.target_class)
    rule.do_evaluate()
    rule.create_model()

    # Argument pruning (check whether some conditions should be pruned from argument)
    # first learn a rule on full data
    learner.target_instances = [index]
    rules = learner(data).rule_list
    if not rules:
        prune = [(None, 0)]
    else:
        full_rule = rules[0]
        prune = [(full_rule, rf(full_rule))]
        for si, s in enumerate(full_rule.selectors):
            # create a rule without this selector
            tmp_rule = Orange.classification.rules.Rule(selectors = [r for r in rule.selectors if r != s],
                                                        domain=data.domain)
            tmp_rule.filter_and_store(X, Y, W, rule.target_class)
            tmp_rule.create_model()
            prune.append((tmp_rule, rf(tmp_rule)))

    return counters, rule, prune

        

