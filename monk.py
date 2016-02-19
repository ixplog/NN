import csv
from random import shuffle
from nn import *
# from matplotlib import pyplot as plt

print "MONK data set"
num_monk_data_set = int(raw_input("Insert data seta number: "))
assert num_monk_data_set == 1 or num_monk_data_set == 2 or num_monk_data_set == 3

print "MONK ", num_monk_data_set
print "Creazione di una rete neurale: 6 input e un output, operazione di classificazione!"
neural_network = NN(6, 3, [8, 5, 1], "classification")
neural_network.print_network()

print ""
print "Leggo i dati..."
test_data = []
with open("MONK/monks-" + str(num_monk_data_set) + ".test") as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        append_row = []
        for i in range(1, len(row)):  # non considero il primo spazio
            try:
                append_row.append(float(row[i]))
            except ValueError:
                append_row.append(row[i])
        test_data.append(append_row)

train_data = []
with open("MONK/monks-" + str(num_monk_data_set) + ".test") as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        append_row = []
        for i in range(1, len(row)):  # non considero il primo spazio
            try:
                append_row.append(float(row[i]))
            except ValueError:
                append_row.append(row[i])
        train_data.append(append_row)


val_accuracy = 0
epochs = 0
plotting_data = []
while val_accuracy != 1 and epochs < 50000:
    tr_accuracy = 0
    val_accuracy = 0

    # randomizzazione dell'ordine dei dati di una epoch
    shuffle(train_data)

    inputs = []
    outputs = []
    for i in range(len(train_data)):
        inputs.append(train_data[i][1:7])
        outputs.append(train_data[i][0])
    neural_network.train(inputs, outputs,
                         NN.initialize_edge_matrix(neural_network.n, neural_network.l, neural_network.p),
                         NN.initialize_vertex_vector(neural_network.l, neural_network.p))

    epochs += 1

    print ""

    for element_tr in train_data:
        o = neural_network.output(element_tr[1:7])
        t = element_tr[0]
        if o[0] <= 0.5 and t == 0:
            tr_accuracy += 1
        elif o[0] > 0.5 and t == 1:
            tr_accuracy += 1
    for element_val in test_data:
        o = neural_network.output(element_val[1:7])
        if o[0] <= 0.5 and element_val[0] == 0:
            val_accuracy += 1
        elif o[0] > 0.5 and element_val[0] == 1:
            val_accuracy += 1

    val_accuracy /= float(len(test_data))
    tr_accuracy /= float(len(train_data))
    plotting_data.append([epochs, tr_accuracy, val_accuracy])
    print "# EPOCH: ", epochs, "\tACCURACY (valuation): ", val_accuracy, "\tACCURACY (training): ", tr_accuracy


with open("plotting_data", "w") as f:
    for i in range(len(plotting_data)):
        for j in range(len(plotting_data[i])):
            f.write(str(plotting_data[i][j]) + " ")
        f.write("\n")
