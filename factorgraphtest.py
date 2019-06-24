import numpy as np
import factorgraph as fg
# Make an empty graph
g = fg.Graph()

# Add some discrete random variables (RVs)
g.rv('a1', 2)#a1,b1
g.rv('a2', 2)#a2,b2,a5
g.rv('a5',2)#a6
g.rv('a4',2)#a7
g.rv('a3',2)#a7
potential1 = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
])
potential2 = np.array([
        [0.3, 0.7],
        [0.9, 0.1],
])
potential3 = np.array([0.1,0.9,0.9,0.1,0.1,0.1,0.1,0.1]).reshape(2,2,2)
# Add some factors, unary and binary
#g.factor(['a2'], potential=np.array([0.1, 0.9]), ftype='1')
# g.factor(['b2'], potential=np.array([0.67,0.33]))
# g.factor(['a2'], potential=np.array([0.67,0.33]))
# g.factor(['a5'], potential=np.array([0.67,0.33]))
g.factor(['a2', 'a1'], potential=potential1, ftype='1')
g.factor(['a1', 'a4', 'a5'], potential=None, ftype='2')
g.factor(['a3', 'a5'], potential=potential1, ftype='1')
#g.factor(['a2', 'b2', 'a5'], potential=None, ftype='2')
g.factor(['a2', 'a3'], potential=None, ftype='2')
# g.factor(['b2', 'a5'], potential=None, ftype='2')
#g.factor(['a2', 'b2', 'a5'], potential=potential3)
# g.factor(['b', 'd'], potential=np.array([
#         [0.8],
#         [0.1],
#         [0.1],
# ]))
# Run (loopy) belief propagation (LBP)
iters, converged = g.lbp(normalize=True, progress=True)
print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
print

# Print out the final messages from LBP
g.print_messages()
print

# Print out the final marginals
g.print_rv_marginals(normalize=True)
