# import tensorflow as tf
import numpy as np
from operator import add


def hyperbolic_tangent(x):
    return np.tanh(x)


def inverse_hyperbolic_tangent(x):
    assert(-1 < x < 1)
    return np.arctanh(x)


def derivative_hyperbolic_tangent(x):
    return 1 - (np.tanh(x) ** 2)


def identity(x):
    return x


def derivative_identity(_):
    return 1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


class NNunit:
    def __init__(self, num_inputs, activation_function, derivative_activation_function):
        self.n = num_inputs  # numero di input
        self.weights = []  # pesi
        self.bias = 1
        self.actfunct = activation_function
        self.der_actfunct = derivative_activation_function
        self.init_weights()

    def init_weights(self):
        # i pesi sono presi da una distribuzione uniforme di media 0 e varianza...
        self.weights = np.random.uniform(-0.7, 0.7, self.n)

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def net(self, inputs):
        result = self.bias
        for i in range(0, self.n):
            result = result + self.weights[i] * inputs[i]
        return result

    def output(self, inputs):
        return self.actfunct(self.net(inputs))


class NN:
    def __init__(self, num_inputs, num_levels, num_units_per_level, operation_type="regression"):
        self.n = num_inputs
        self.l = num_levels
        self.p = num_units_per_level
        self.type = operation_type
        self.network = []
        self.init_network()

    def init_network(self):
        self.network = [[0 for _ in range(self.p[j])] for j in range(self.l)]
        for j in range(self.l):
            for i in range(self.p[j]):
                if j == 0:
                    # primo livello
                    self.network[j][i] = NNunit(self.n, hyperbolic_tangent, derivative_hyperbolic_tangent)
                elif j == self.l-1:
                    # ultimo livello
                    if self.type == "regression":
                        self.network[j][i] = NNunit(self.p[j-1], identity, derivative_identity)
                    elif self.type == "classification":
                        self.network[j][i] = NNunit(self.p[j-1], hyperbolic_tangent, derivative_hyperbolic_tangent)
                else:
                    # hidden units
                    self.network[j][i] = NNunit(self.p[j-1], hyperbolic_tangent, derivative_hyperbolic_tangent)

    @staticmethod
    def initialize_edge_matrix(num_inputs, num_levels, num_units_per_level):
        # inizializza una matrice di tre dimensioni: la prima dimensione corrisponde ad un livello della rete neurale;
        # la seconda dimensione ad un nodo della rete e l'ultima ad una sinapsi (arco entrante) di quel nodo.
        # La matrice contiene tutti valori nulli.
        matrix = [[[] for _ in range(num_units_per_level[j])] for j in range(num_levels)]
        for j in range(num_levels):
            for i in range(num_units_per_level[j]):
                if j == 0:
                    matrix[0][i] = [0] * num_inputs
                else:
                    matrix[j][i] = [0] * num_units_per_level[j-1]

        return matrix

    @staticmethod
    def initialize_vertex_vector(num_levels, num_units_per_level):
        # inizializza un vettore di zeri, ognuno dei quali associato ad un nodo della rete
        return [[0 for _ in range(num_units_per_level[j])] for j in range(num_levels)]

    def print_network(self):
        for j in range(self.l):
            print "LIVELLO ", j + 1
            for i in range(self.p[j]):
                print "\tNodo ", i + 1, ", numero di sinapsi uguale a ", self.network[j][i].n, "; pesi sulle sinapsi: ", \
                    self.network[j][i].weights, "; bias ", self.network[j][i].bias

    def print_weights(self):
        print "WEIGHTS:"
        for j in range(self.l):
            print "LIVELLO ", j + 1
            for i in range(self.p[j]):
                print "\tPesi del nodo ", i + 1, ": ", self.network[j][i].weights, ", bias ", self.network[j][i].bias

    def output(self, inputs):
        # output di ogni nodo della rete neurale
        partial_results = [[0 for _ in range(self.p[j])] for j in range(self.l)]
        for j in range(self.l):
            for i in range(self.p[j]):
                if j == 0:
                    partial_results[0][i] = self.network[0][i].output(inputs)
                else:
                    partial_results[j][i] = self.network[j][i].output(partial_results[j-1][:])
        # restituisco solo gli output dei nodi nell'ultimo livello della rete neurale
        return partial_results[self.l-1][:]

    def output_outs_nets(self, inputs):
        # output di ogni nodo della rete neurale
        partial_results = [[0 for _ in range(self.p[j])] for j in range(self.l)]
        # "net" di ogni nodo della rete neurale
        nets = [[0 for _ in range(self.p[j])] for j in range(self.l)]
        for j in range(self.l):
            for i in range(self.p[j]):
                if j == 0:
                    partial_results[0][i] = self.network[0][i].output(inputs)
                    nets[0][i] = self.network[0][i].net(inputs)
                else:
                    partial_results[j][i] = self.network[j][i].output(partial_results[j-1][:])
                    nets[j][i] = self.network[j][i].net(partial_results[j-1][:])
        return partial_results, nets

    def train(self, variables, targets, prev_delta_weights, prev_delta_bias,
              regularization_coeff=0.02, momentum_coeff=0.5, eta=0.05):
        # variables e targets sono vettori di vettori
        return self.batch_backpropagation(variables, targets,
                                          regularization_coeff, momentum_coeff, eta, prev_delta_weights, prev_delta_bias)

    def batch_backpropagation(self, variables, targets, reg_coeff, momentum_coeff, eta, prev_delta_weights, prev_delta_bias):
        # considerare l'errore medio su un epoch di dati
        assert(len(variables) == len(targets))
        number_examples = len(variables)

        # variazione su ogni peso della rete neurale
        delta_weights = NN.initialize_edge_matrix(self.n, self.l, self.p)
        # delta_weights: matrice tridimensionale -> livello * nodo di quel livello * connessioni di quel nodo

        # variazione dei valori del bias
        delta_bias = NN.initialize_vertex_vector(self.l, self.p)

        for k in range(number_examples):
            # output e net della rete neurale in corrispondeza dell'input k-esimo
            outs, nets = self.output_outs_nets(variables[k])
            delta = NN.initialize_vertex_vector(self.l, self.p)
            for j in range(self.l):
                if j == 0:
                    # ultimo livello
                    for i in range(self.p[-1]):
                        # la funzione di attivazione ha derivata 1 nell'ultimo livello, quindi deltai e ei coincidono
                        if isinstance(targets[k], list):
                            ei = targets[k][i] - outs[-1][i]
                        else:  # considero il caso in cui ho un solo valore di output
                            assert isinstance(targets[k], float)
                            ei = targets[k] - outs[-1][i]
                        delta[-1][i] = ei
                        for h in range(self.p[-2]):
                            delta_weights[-1][i][h] += delta[-1][i] * outs[-2][h]  # eta lo considero dopo
                            delta_bias[-1][i] += delta[-1][i]
                else:
                    # altri livelli
                    for i in range(self.p[-j-1]):
                        temp = 0  # variabile da moltiplicare per deltai per ottenere il risultato finale
                        for h in range(self.p[-j]):
                            temp += delta[-j][h] * self.network[-j][h].weights[i]
                        delta[-j-1][i] = temp * self.network[-j-1][i].der_actfunct(nets[-j-1][i])

                        delta_bias[-j-1][i] += delta[-j-1][i]

                        if j == self.l-1:
                            # primo livello
                            for h in range(self.n):
                                delta_weights[0][i][h] += delta[0][i] * variables[k][h]
                        else:
                            for h in range(self.p[-j-2]):
                                delta_weights[-j-1][i][h] += delta[-j-1][i] * outs[-j-2][h]

        # ho accumulato tutti i delta_weights su una intera epoch di dati
        # aggiorno i pesi sulle sinapsi
        for j in range(self.l):
            for i in range(self.p[j]):
                for h in range(len(delta_weights[j][i])):
                    delta_weights[j][i][h] *= eta/number_examples
                    # Regolarizzazione ---->> (- eta * reg_coeff * x)
                    delta_weights[j][i][h] -= eta * reg_coeff * delta_weights[j][i][h]
                    # Momentum
                    delta_weights[j][i][h] += momentum_coeff * prev_delta_weights[j][i][h]
                #  No regularization on bias
                delta_bias[j][i] *= eta/number_examples
                delta_bias[j][i] += momentum_coeff * prev_delta_bias[j][i]

                self.network[j][i].set_weights(map(add, self.network[j][i].weights, delta_weights[j][i]),
                                               self.network[j][i].bias + delta_bias[j][i])
        return delta_weights, delta_bias
