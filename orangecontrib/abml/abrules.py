from warnings import warn
import re
import numpy as np
import Orange
from orangecontrib.evcrules.rules import RulesStar
from Orange.classification.rules import Rule, Selector, _RuleClassifier, \
    Evaluator, CN2UnorderedClassifier, get_dist, LRSValidator, Validator

validArguments_re = re.compile(r"""[" \s]*                         # remove any special characters at the beginning
                     ~?                                         # argument could be negative (~) or positive (without ~)
                     {                                          # left parenthesis of the argument
                     \s*[\w\W]+                                 # first attribute of the argument
                     (\s*,\s*[\w\W]+)*                          # following attributes in the argument
                     }                                          # right parenthesis of the argument
                     (\s*,\s*~?{\s*[\w\W]+(\s*,\s*[\w\W]+)*})*  # following arguments
                     [" \s]*                                    # remove any special characters at the end
                     """
                     , re.VERBOSE)

# splitting regular expressions
argument_re = re.compile(r'\{[^{}]+\}')

class ABRuleLearner(RulesStar):
    """
    Requires: type string meta attribute named <code>Arguments</code>
    """

    def __init__(self, preprocessors=None, base_rules=None, m=2, evc=True,
                 max_rule_length=5, width=100, default_alpha=1.0,
                 parent_alpha=1.0, add_sub_rules=False):
        super().__init__(preprocessors=preprocessors,
                         base_rules=base_rules,
                         m=m, evc=evc, max_rule_length=max_rule_length,
                         width=width, default_alpha=default_alpha,
                         parent_alpha=parent_alpha, add_sub_rules=add_sub_rules)

    def fit_storage(self, data):
        if "Arguments" not in data.domain:
            warn("No meta attribute representing Arguments! Learning without arguments.")
            learner = RulesStar(preprocessors=self.preprocessors,
                                base_rules=self.base_rules, m=self.m, evc=self.evc,
                                max_rule_length=self.max_rule_length, width=self.width,
                                default_alpha=self.default_alpha, parent_alpha=self.parent_alpha,
                                add_sub_rules=self.add_sub_rules)
            learner.evds = self.evds
            return learner(data)

        X, Y, W = data.X, data.Y, data.W if data.W else None
        Y = Y.astype(dtype=int)

        # parse arguments and set constraints
        self.base_rules, self.constraints = self.parse_args(data, X, Y, W)
        self.cons_index = np.equal(self.constraints, None) == False

        return super().fit_storage(data)

    def parse_args(self, data, X, Y, W):
        base_rules = []
        constraints = [None for i in range(X.shape[0])]
        arg_index = data.domain.index("Arguments")
        metas = data.metas
        for inst, args in enumerate(metas[:, -arg_index-1]):
            if not args:
                continue
            if not validArguments_re.match(args):
                warn('Args "{}" do not match predefined arguments format'.format(args))
                continue
            constraints[inst] = []
            args = argument_re.findall(args)
            for arg in args:
                neg = arg.startswith("~")
                if neg:
                    warn('Negative arguments are not yet supported. Skipping them.')
                    continue
                arg = arg.strip("{}").strip()
                att_cons = [att.strip() for att in arg.split(",")]
                # create a rule from fixed constraints
                # undefined constraints leave for now
                selectors = []
                unfinished = []
                for ac in att_cons:
                    column, op, value = self.parse_constraint(ac, data, inst)
                    if column == None:
                        warn("Can not parse {}. Please check the type of attribute.".format(ac))
                        continue
                    if value == None: # unspecified argument
                        unfinished.append((column, op))
                    else:
                        selectors.append(Selector(column=column, op=op, value=value))
                rule = Rule(selectors=selectors, domain=data.domain)
                rule.filter_and_store(X, Y, W, Y[inst])
                # if we have any unfinished selectors, we have to find some values for that
                spec_rules = self.specialize(rule, unfinished, X, Y, W, inst)
                for sr in spec_rules:
                    constraints[inst].append(sr)
                    base_rules.append(sr)
        return base_rules, np.array(constraints, dtype=object)

    def specialize(self, rule, unfinished_selectors, X, Y, W, instance_index):
        us_tuple = list((sel[0], sel[1]) for sel in unfinished_selectors)
        rule.general_validator = self.rule_finder.general_validator
        self.rule_finder.search_strategy.storage = {}
        rules = [rule]
        star = [rule]
        while star:
            new_star = []
            for rs in star:
                refined = self.rule_finder.search_strategy.refine_rule(X, Y, W, rs)
                # check each refined rule whether it is consistent with unfinished_selectors
                for ref_rule in refined:
                    # check last selector if it is consistent with unfinished_selectors
                    sel = ref_rule.selectors[-1]
                    if (sel.column, sel.op) in us_tuple:
                        ref_rule.filter_and_store(X, Y, W, rule.target_class)
                        if ref_rule.covered_examples[instance_index]:
                            rules.append(ref_rule)
                            new_star.append(ref_rule)
            star = new_star
        return rules

    def parse_constraint(self, att_cons, data, inst):
        sp = re.split('[(>=)(<=)<>]', att_cons)
        if len(sp) == 1:
            neg = att_cons.startswith("~")
            if neg:
                att = att_cons.strip("~").strip()
            else:
                att = att_cons
            att_col = data.domain.index(att)
            if not data.domain.attributes[att_col].is_discrete:
                return None, None, None
            return att_col, "!=" if neg else "==", data.X[inst, att_col]
        else:
            att = sp[0]
            try:
                val = float(sp[1])
            except:
                val = None
            if ">" in att_cons:
                op = ">="
            else:
                op = "<="
            att_col = data.domain.index(att)
            if not data.domain.attributes[att_col].is_continuous:
                return None, None, None
            return att_col, op, val

    def create_initial_star(self, X, Y, W, prior):
        star = []
        for cli, cls in enumerate(self.domain.class_var.values):
            # select base rules that have class cls
            base_cls = [r for r in self.base_rules if cli == r.target_class]
            # add default to base
            base_cls.append(Rule(selectors=[], domain=self.domain))
            rules = self.rule_finder.search_strategy.initialise_rule(
                X, Y, W, cli, base_cls, self.domain, prior, prior,
                self.evaluator, self.rule_finder.complexity_evaluator,
                self.rule_validator, self.rule_finder.general_validator)
            star.extend(rules)
        for r in star:
            r.default_rule = r
            r.do_evaluate()
        return star

    def update_best(self, bestr, bestq, rule, Y):
        indices = (rule.covered_examples) & (rule.target_class == Y) & \
                  (rule.quality-0.005 > bestq)
        # remove indices where rule is not consistent with constraints
        for ind in np.nonzero(self.cons_index & indices)[0]:
            if rule.default_rule not in self.constraints[ind]:
                indices[ind] = 0
        bestr[indices] = rule
        bestq[indices] = rule.quality


if __name__ == '__main__':
    import pickle
    data = Orange.data.Table('adult_sample')
    learner = ABRuleLearner()
    learner.evds = pickle.load(open("adult_evds.pickle", "rb"))
    #learner.calculate_evds(data)
    pickle.dump(learner.evds, open("adult_evds.pickle", "wb"))
    classifier = learner(data)

    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule, rule.quality)
    print()

