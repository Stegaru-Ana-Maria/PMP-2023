from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


model = BayesianNetwork([('E', 'A'), ('I', 'A')])


cpd_e = TabularCPD(variable="E", variable_card=2, values=[[0.9995], [0.0005]])

cpd_i = TabularCPD(variable="I", variable_card=2, values=[[0.99, 0.01]])
cpd_a = TabularCPD(variable="A", variable_card=2,
                        values=[[0.94, 0.06, 0.02, 0.98],
                                [0.01, 0.99, 0.03, 0.97]],
                        evidence=["E", "I"], evidence_card=[2, 2])


model.add_cpds(cpd_e, cpd_i, cpd_a)


assert model.check_model()


inference = VariableElimination(model)

#Interfata
infer = VariableElimination(model)
result = infer.query(variables=['E'], evidence={'A': 1})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()


#EX2
# Calcularea probabilității să fi avut loc un cutremur dată fiind declanșarea alarmei de incendiu
result = inference.query(variables=['E'], evidence={'A': 1})
print(result)
result = inference.query(variables=['E'], evidence={'A': 0})
print(result)


#EX3

# Calcularea probabilității ca un incendiu să fi avut loc fără activarea alarmei de incendiu
result = inference.query(variables=['I'], evidence={'A': 0})
print(result)
result = inference.query(variables=['I'], evidence={'A': 1})
print(result)
