import numpy as np
import factorgraph as fg
# Make an empty graph
g = fg.Graph()

# Add some discrete random variables (RVs)
g.rv('a', 2)#a1,b1
g.rv('b', 3)#a2,b2,a5
g.rv('c',1)#a6
g.rv('d',1)#a7

# Add some factors, unary and binary
g.factor(['a'], potential=np.array([0.5, 0.5]))
g.factor(['b'], potential=np.array([0.33, 0.33, 0.33]))
g.factor(['c'], potential=np.array([1.0]))
g.factor(['d'], potential=np.array([1.0]))
g.factor(['b', 'a'], potential=np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.1],
]))
# g.factor(['a', 'b'], potential=np.array([
#         [0.8, 0.1, 0.1],
#         [0.1, 0.8, 0.1],
# ]))
g.factor(['c', 'b'], potential=np.array([
        [0.1, 0.1, 0.9],
]))
# g.factor(['b', 'c'], potential=np.array([
#         [0.1],
#         [0.1],
#         [0.8],
# ]))
g.factor(['d', 'b'], potential=np.array([
        [0.9, 0.1, 0.1],
]))
# g.factor(['b', 'd'], potential=np.array([
#         [0.8],
#         [0.1],
#         [0.1],
# ]))
# Run (loopy) belief propagation (LBP)
iters, converged = g.lbp(normalize=True)
print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
print

# Print out the final messages from LBP
g.print_messages()
print

# Print out the final marginals
g.print_rv_marginals(normalize=True)
