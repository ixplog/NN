# coding=utf-8
import csv
from random import shuffle
from nn import *
from datetime import datetime
# from matplotlib import pyplot as plt
# import sys


print "MAIN File!"
print "Sto creando una rete neurale che prenda 10 numeri in input e ne restituisca 2 in output..."

# Iperparametri
num_levels = 4
num_units_per_level = [5, 5, 5, 2]  # l'ultimo valore deve essere 2
eta = 0.05  # 0.05, 0.1, 0.5
regularization_coefficient = 0.02  # 0.01, 0.02, 0.05
momentum_coefficient = 0.5

neural_network = NN(10, num_levels, num_units_per_level)
neural_network.print_network()

print ""
print "Leggo i dati..."
data = []
with open("data/LOC-OSM-TR.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 13:  # escludo i commenti nel file csv
            data.append([float(row[i]) for i in range(13)])  # conversione a float di ogni dato

# HOLD OUT!
# divido il training set in tre blocchi: uno contiene il 50% degli elementi, gli altri il 25% ciascuno
# shuffle!
shuffle(data)
tr = data[:len(data) / 2]
val = data[len(data) / 2:3 * len(data) / 4]
test = data[3 * len(data) / 4:]

# strutture dati di supporto, all'inizio contengono solamente zeri
prev_delta_weights = NN.initialize_edge_matrix(neural_network.n, neural_network.l, neural_network.p)
prev_delta_bias = NN.initialize_vertex_vector(neural_network.l, neural_network.p)

plotting_errors = []  # errore_di_training, errore_di_valutazione

# ultimi 100 errori di validazione
val_errors = [0] * 100
index = 0  # rappresenta l'indice del prossimo elemento da rimuovere nel vettore "val_errors"

stop = False
epochs = 0
while (not stop) and epochs < 5000:
    tr_error = 0
    val_error = 0

    # randomizzazione dell'ordine dei dati di una epoch
    shuffle(tr)

    inputs = []
    outputs = []
    for i in range(len(tr)):
        inputs.append(tr[i][1:11])
        outputs.append(tr[i][11:13])
    prev_delta_weights, prev_delta_bias = neural_network.train(inputs, outputs, prev_delta_weights, prev_delta_bias,
                                                               regularization_coefficient, momentum_coefficient, eta)

    epochs += 1

    print ""

    # training error
    for element_tr in tr:
        o = neural_network.output(element_tr[1:11])
        t = element_tr[11:13]
        error_i = 0
        for i in range(len(o)):
            error_i += (t[i] - o[i]) ** 2
        tr_error += np.sqrt(error_i)
    # validation error
    for element_val in val:
        o = neural_network.output(element_val[1:11])
        t = element_val[11:13]
        error_i = 0
        for i in range(len(o)):
            error_i += (t[i] - o[i]) ** 2
        val_error += np.sqrt(error_i)

    val_error /= len(val)
    tr_error /= len(tr)
    print "# EPOCH: ", epochs, "\tERROR (valuation): ", val_error, "\tERROR (training): ", tr_error

    plotting_errors.append([tr_error, val_error])

    # strategia di stop: se la media degli errori di validazione sulle ultime 25 osservazioni non si discosta molto dalla
    # media dell'errore di validazione sulle precedenti 75 osservazioni allora mi fermo.
    if epochs < 100:
        val_errors[epochs] = val_error
    else:
        val_errors[index] = val_error
        index = (index + 1) % 100
        last = [0] * 25
        previous = [0] * 75
        for j in range(25):
            last[j] = val_errors[(index + j) % 100]
        for j in range(75):
            previous[j] = val_errors[(index + 25 + j) % 100]
        av1 = sum(last) / float(len(last))
        av2 = sum(previous) / float(len(previous))
        if abs((av1 - av2)/av2) < 0.00001:  # 0.01%
            stop = True
            # scrivere su file i risultati ottenuti, stimando l'errore sul test set...
            with open("main_results", "w") as f:
                f.write("RISULTATI " + str(datetime.now()) + "\n")
                f.write("Iperparametri scelti:\n\tNumero di livelli della rete neurale: " + str(num_levels) +
                        "\n\tNumero di unità per ogni livello: " + str(num_units_per_level) +
                        "\n\tCoefficiente di regolarizzazione: " + str(regularization_coefficient) +
                        "\n\tMomentum coefficient: " + str(momentum_coefficient) + "\n\tLearning Parameter: " + str(eta))

                f.write("\n\n")

                # Errore sul test set
                test_error = 0
                for element in test:
                    o = neural_network.output(element[1:11])
                    t = element[11:13]
                    error_i = 0
                    for i in range(len(o)):
                        error_i += (t[i] - o[i]) ** 2
                    test_error += np.sqrt(error_i)
                test_error /= len(test)

                f.write("TEST ERROR: " + str(test_error) + "\n")
                f.write("Validation error: " + str(val_error) + "\n")
                f.write("Training error: " + str(tr_error) + "\n\n")

                f.write("Numero di iterazioni effettuate: " + str(epochs) + "\n\n")

                # previsione su dati mai visti
                f.write("PREVISIONE SUI DATI SENZA ETICHETTA\n")
                newdata = []
                with open("data/LOC-OSM-TS.csv") as g:
                    reader = csv.reader(g)
                    for row in reader:
                        if len(row) == 11:  # escludo i commenti nel file csv
                            newdata.append([float(row[i]) for i in range(11)])  # conversione a float di ogni dato
                for i in range(len(newdata)):
                    o = neural_network.output(newdata[i][1:11])
                    f.write(str(newdata[i][0]) + ", " + str(o[0]) + ", " + str(o[1]))
                    f.write("\n")
                f.write("\n\n")

                # salvo gli errori di training e di valutazione
                f.write("\nERRORI DI TRAINING E DI VALUTAZIONE:\n\n")
                for i in range(epochs):
                    f.write(str(i) + " " + str(plotting_errors[i][0]) + " " + str(plotting_errors[i][1]) + "\n")
