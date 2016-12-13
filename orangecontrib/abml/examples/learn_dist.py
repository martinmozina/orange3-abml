import pickle
from Orange.data import Table
import orangecontrib.abml.abrules as rules

data = Table('learndata')

rule_learner = rules.ABRuleLearner(add_sub_rules=True)
rule_learner.calculate_evds(data)
pickle.dump(rule_learner.evds, open("evds.pickle", "wb"))
