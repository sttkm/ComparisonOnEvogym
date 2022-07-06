
from neat_.feedforward import FeedForwardNetwork
from neat_ import create_cppn

class EvogymHyperDecoder():
    def __init__(self, substrate, use_hidden=False, activation='sigmoid'):

        self.activation = activation

        connections = [('input', 'output')]
        downstream_nodes = ['output']

        substrate.make_substrate()
        if use_hidden:
            substrate.add_hidden('hidden', 1)
            connections.extend([('input', 'hidden'), ('hidden', 'output')])
            downstream_nodes.append('hidden')

        substrate.make_vectors()

        self.edge_labels, self.edge_inputs = substrate.get_connection_inputs(connections)
        self.node_labels, self.node_inputs = substrate.get_node_inputs(downstream_nodes)
        self.input_nodes = substrate.get_nodes('input')
        self.output_nodes = substrate.get_nodes('output')
        self.input_dims = substrate.get_dim_size()
        self.output_dims = 1

    def decode(self, genome, config):
        nodes = create_cppn(
            genome, config,
            leaf_names=self.edge_inputs.keys(),
            node_names=['value'])


        biases = nodes[0](**self.node_inputs).numpy()
        biases = biases * 5

        biases = {node: bias for node,bias in zip(self.node_labels, biases)}

        weights = nodes[0](**self.edge_inputs).numpy()
        weights = weights * 5

        connections = {edge: weight for edge,weight in zip(self.edge_labels, weights)}

        return FeedForwardNetwork.create_from_cppn(
            config=config,
            input_keys=self.input_nodes,
            output_keys=self.output_nodes,
            biases=biases,
            weights=connections,
            weight_thr=0.5,
            default_activation=self.activation)