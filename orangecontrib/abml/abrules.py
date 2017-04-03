from warnings import warn
import re
import numpy as np
import Orange
from copy import copy
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
                 parent_alpha=1.0, add_sub_rules=False, target_instances=None):
        super().__init__(preprocessors=preprocessors,
                         base_rules=base_rules,
                         m=m, evc=evc, max_rule_length=max_rule_length,
                         width=width, default_alpha=default_alpha,
                         parent_alpha=parent_alpha, add_sub_rules=add_sub_rules,
                         target_instances=target_instances)

    def fit_storage(self, data):
        # parse arguments and set constraints
        self.base_rules, self.constraints = self.parse_args(data)
        self.cons_index = np.equal(self.constraints, None) == False

        return super().fit_storage(data)

    def parse_args(self, data):
        base_rules = []
        constraints = [None for i in range(len(data))]
        if "Arguments" not in data.domain:
            return base_rules, constraints
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
                rule, unfinished = ABRuleLearner.create_rule_from_argument(arg, data, inst)
                # if we have any unfinished selectors, we have to find some values for that
                if unfinished:
                    spec_rules = self.specialize(rule, unfinished, data, inst)
                else:
                    spec_rules = [rule]
                for sr in spec_rules:
                    sr.create_model()
                    constraints[inst].append(str(sr))
                    base_rules.append(sr)
        return base_rules, np.array(constraints, dtype=object)

    @staticmethod
    def create_rule_from_argument(arg, data, inst):
        X, Y, W = data.X, data.Y, data.W if data.W else None
        Y = Y.astype(dtype=int)

        neg = arg.startswith("~")
        if neg:
            warn('Negative arguments are not yet supported. Skipping them.')
            return None, None
        arg = arg.strip("{}").strip()
        att_cons = [att.strip() for att in arg.split(",")]
        # create a rule from fixed constraints
        # undefined constraints leave for now
        selectors = []
        unfinished = []
        for aci, ac in enumerate(att_cons):
            column, op, value = ABRuleLearner.parse_constraint(ac, data, inst)
            if column == None:
                warn("Can not parse {}. Please check the type of attribute.".format(ac))
                continue
            elif isinstance(value, str) and value.startswith('?'):
                value = float(value[1:])
                unfinished.append(aci)
            elif isinstance(value, str):
                # set maximum/minimum value
                if op == ">=":
                    value = np.min(data.X[column])
                else:
                    value = np.max(data.X[column])
            selectors.append(Selector(column=column, op=op, value=value))
        rule = Rule(selectors=selectors, domain=data.domain)
        rule.filter_and_store(X, Y, W, Y[inst])
        return rule, unfinished

    def specialize(self, rule, unfinished_selectors, data, instance_index):
        X, Y, W = data.X, data.Y, data.W if data.W else None
        Y = Y.astype(dtype=int)

        rule.general_validator = self.rule_finder.general_validator
        self.rule_finder.search_strategy.storage = {}
        rules = [rule]
        star = [rule]
        while star:
            new_star = []
            for rs in star:
                rs.create_model()
                refined = self.rule_finder.search_strategy.refine_rule(X, Y, W, rs)
                # check each refined rule whether it is consistent with unfinished_selectors
                for ref_rule in refined:
                    # check last selector if it is consistent with unfinished_selectors
                    sel = ref_rule.selectors[-1]
                    for i, (old_sel) in enumerate(ref_rule.selectors[:-1]):
                        if (old_sel.column, old_sel.op) == (sel.column, sel.op) and \
                                        i in unfinished_selectors:
                            # this rules is candidate for further specialization
                            # create a copy of rule
                            new_rule = Rule(selectors=copy(rule.selectors),
                                             domain=rule.domain,
                                             initial_class_dist=rule.initial_class_dist,
                                             prior_class_dist=rule.prior_class_dist,
                                             quality_evaluator=rule.quality_evaluator,
                                             complexity_evaluator=rule.complexity_evaluator,
                                             significance_validator=rule.significance_validator,
                                             general_validator=rule.general_validator)
                            new_rule.selectors[i] = Selector(column=sel.column, op=sel.op, value=sel.value)
                            new_rule.filter_and_store(X, Y, W, rule.target_class)
                            if new_rule.covered_examples[instance_index]:
                                rules.append(new_rule)
                                new_star.append(new_rule)
                            break
            star = new_star
        return rules

    @staticmethod
    def parse_constraint(att_cons, data, inst):
        sp = re.split('>=|<=', att_cons)
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
                val = str(sp[1])
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
            if self.target_class is None or cli == self.target_class or cls == self.target_class:
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
            if r.length > 0:
                for ind in np.nonzero(self.cons_index)[0]:
                    r.create_model()
                    str_r = str(r)
                    for cri, cr in enumerate(self.constraints[ind]):
                        if isinstance(cr, str) and str_r == cr:
                            self.constraints[ind][cri] = r # replace string with rule
            r.do_evaluate()
        return star

    def update_best(self, bestr, bestq, rule, Y):
        indices = (rule.covered_examples) & (rule.target_class == Y) & \
                  (rule.quality-0.005 > bestq)
        # remove indices where rule is not consistent with constraints
        if self.target_instances:
            ind2 = np.zeros(indices.shape, dtype=bool)
            ind2[self.target_instances] = indices[self.target_instances]
            indices = ind2
        for ind in np.nonzero(self.cons_index & indices)[0]:
            indices[ind] = 0
            for cn in self.constraints[ind]:
                if rule.default_rule is cn:
                    indices[ind] = 1
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

