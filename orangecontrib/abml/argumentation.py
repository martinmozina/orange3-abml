import numpy as np
import Orange
from Orange.data import Table
from Orange.classification.rules import Selector
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist, squareform
from Orange import distance
from Orange.clustering import hierarchical
from orangecontrib.abml.abrules import ABRuleLearner, argument_re

ARGUMENTS = "Arguments"
SIMILAR = 30

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

def kmeans(dist, weights, initial):
    centers = initial
    cent_set = set()
    steps = 0
    while str(centers) not in cent_set and steps < 100:
        cent_set.add(str(centers))
        steps += 1
        # compute clusters that belong to each center
        weighted = dist[centers]*weights
        mins = np.argmin(weighted, 0)
        # recalulate centers
        new_centers = []
        for i in range(len(centers)):
            ind = mins == i
            # select only instances belonging to this cluster
            seldist = dist[np.ix_(ind, ind)]*weights[ind]
            # central cluster is the one having the smallest distance sum to others
            ind_vals = np.where(ind)[0]
            central = ind_vals[np.argmin(seldist.sum(1))]
            new_centers.append(central)
        centers = new_centers
    return centers

def rf(rule):
    """ Classification accuracy of rule measured with relative frequency. """
    dist = rule.curr_class_dist
    return dist[rule.target_class] / dist.sum()


def find_critical(learner, data, n=5, k=5, random_state=0):
    """
    :param learner: argument-based learner to be tested
    :param data: learning data
    :param n: number of critical examples
    :param k: folds in cross-validation
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
        # move test_ind with arguments to learn_ind
        arg_ind = []
        if ARGUMENTS in data.domain:
            for t in test_ind:
                if data[t][ARGUMENTS] not in ("", "?"):
                    arg_ind.append(t)
        learn_ind = np.array(sorted(list(learn_ind)+arg_ind), dtype=int)
        test_ind = np.array([t for t in test_ind if t not in arg_ind], dtype=int)
        learn = Table(data.domain, data[learn_ind])
        test = Table(data.domain, data[test_ind])

        classifier = learner(learn)
        rules = classifier.rule_list
        # eval rules on test data
        cov = classifier.coverage(test)

        # for each test instance find out best covering rule from the same class
        best_covered = np.zeros(len(test))
        for ri, r in enumerate(rules):
            target = r.target_class == test.Y
            best_covered = np.maximum(best_covered,  (cov[:, ri] & target) * r.quality )

        # compute how problematic each instance is ...
        probs = classifier(test, 1)
        for ti, t in enumerate(test_ind):
            # first check best rule, if same class, it can not be problematic
            d, p = test[ti], probs[ti]
            c = int(d.get_class())
            # find best rule covering this example (best_rule * prediction)
            problematic[t] = (1 - best_covered[ti]) * (1 - p[c])
            problematic_rules[t] = [r for ri, r in enumerate(rules) if cov[ti, ri]]

    # compute Mahalanobis distance between instances
    dist_matrix = squareform(pdist(data.X, metric="seuclidean"))

    # compute distances between instances
    #dist_matrix = distance.Euclidean(data.X)
    #print(dist_matrix)
    #dist_matrix /= np.max(dist_matrix)

    # criticality is a combination of how much is the instance problematic
    # and its distance to other problematic examples of the same class
    # for loop over classes
    vals = np.unique(data.Y.astype(dtype=int))
    k = int(np.ceil(n/len(vals)))
    #criticality = np.zeros(len(data))
    crit_ind = []
    for i in vals:
        inst = (data.Y == i) & (problematic > 1e-6)
        inst_pos = np.where(inst)[0]
        wdist = dist_matrix[np.ix_(inst, inst)]
        # select k most problematic instances
        prob = problematic[inst]
        ind = np.argpartition(prob, -k)[-k:]
        centers = kmeans(wdist, prob, ind)
        for c in centers:
            crit_ind.append(inst_pos[c])

    # sort critical indices given problematicness
    crit_ind = sorted(crit_ind, key = lambda x: -problematic[x])

    """vals = np.unique(data.Y.astype(dtype=int))
    criticality = np.zeros(len(data))
    for i in vals:
        inst = data.Y == i
        prob = problematic[inst]
        wdist = dist_matrix[np.ix_(inst,inst)] * prob
        wdist = wdist.sum(axis=1) / (prob.sum() + 1e-6)
        wdist = wdist / (np.max(wdist) + 1e-6)
        criticality[inst] = prob / (wdist + 1e-6)
        #criticality[inst] = (2 - wdist) * prob
    """
    """# get most critical instances
    crit_ind = []
    tmp_crit = np.array(criticality)
    while len(crit_ind) < n:
        # sort
        sorted_ind = tmp_crit.argsort()
        # add last value
        crit_ind.append(sorted_ind[-1])
        # recompute tmp_crit according to selected example
        inst = data.Y == data.Y[crit_ind[-1]]
        tmp_crit[inst] *= dist_matrix[inst, crit_ind[-1]]"""

    return (crit_ind, problematic[crit_ind],
           [problematic_rules[i] for i in crit_ind])
    #return (crit_ind, criticality[crit_ind], problematic[crit_ind], 
    #       [problematic_rules[i] for i in crit_ind])

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

    # learn rules; find best rule for each example
    X, Y = data.X, data.Y.astype(dtype=int)
    rules = learner(data).rule_list
    best_covered = np.zeros(len(data))
    for ri, r in enumerate(rules):
        target = r.target_class == Y
        best_covered = np.maximum(best_covered,  (r.covered_examples & target) * r.quality )

    # run 5-times repeated cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    counters = []
    learned = []
    for learn, test in skf.split(data.X, data.Y):
        # move test_ind with arguments to learn_ind
        arg_ind = []
        if ARGUMENTS in data.domain:
            for t in test:
                if data[t][ARGUMENTS] not in ("", "?"):
                    arg_ind.append(t)
        learn = np.array(sorted(list(learn)+arg_ind), dtype=int)
        test = np.array([t for t in test if t not in arg_ind], dtype=int)
        learndata = Table(data.domain, data[learn])
        testdata = Table(data.domain, data[test])
        new_index = np.where(learn == index)[0][0] # new argumented example position
        
        learner.target_instances = [new_index]
        rules = learner(learndata).rule_list

        if not rules: 
            # rule was not learned from argumented example ...
            # create a rule from argument
            print("Couldnt learn a rule, converting argument to rule.")
            args = argument_re.findall(str(learndata[new_index][ARGUMENTS]))
            rule, unfinished = ABRuleLearner.create_rule_from_argument(args[0], learndata, new_index)
        else:
            # learner should learn exactly one rule
            assert len(rules) == 1
            rule = rules[0]
            print("Learned rule: ", rule)
        
        learned.append(rule) # append rule

        # counter examples are examples covered by rule in testdata and
        # learndata, but from the opposite class
        test_cov = rule.evaluate_data(testdata.X)
        test_counter = test_cov & (testdata.Y != rule.target_class)
        counters += list(test[test_counter])
        learn_counter = rule.covered_examples & (learndata.Y != rule.target_class)
        counters += list(learn[learn_counter])

    # from counters, present those that have worse covered rules
    counters_vals = np.ones(len(data))
    counters_vals[counters] = best_covered[counters]
    counters = counters_vals.argsort()[:n]
    counters = [c for c in counters if counters_vals[c] < 1]
    counters_vals = counters_vals[counters]

    # find common conditions in learned
    if len(learned) < 2: #n-1:
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
    # Can not evaluate rule when it is manually created and not learned!
    ##rule.do_evaluate()
    rule.create_model()

    # Argument pruning (check whether some conditions should be pruned from argument)
    # first learn a rule on full data
    #learner.target_instances = [index]
    #rules = learner(data).rule_list
    full_rule = rule
    if len(full_rule.selectors) == 0:
        prune = [(None, 0)]
    else:
        #full_rule = rules[0]
        prune = [(full_rule, rf(full_rule))]
        for si, s in enumerate(full_rule.selectors):
            # create a rule without this selector
            tmp_rule = Orange.classification.rules.Rule(selectors = [r for r in rule.selectors if r != s],
                                                        domain=data.domain)
            tmp_rule.filter_and_store(X, Y, W, rule.target_class)
            tmp_rule.create_model()
            prune.append((tmp_rule, rf(tmp_rule)))

    return counters, counters_vals, rule, prune

        

