import graphviz

with open('tree.dot') as f:
    dot_graph=f.read()
dot=graphviz.Source(dot_graph)
dot.format='png'
dot.render(filename='tree')