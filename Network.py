def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, batch_size = 1, learning_rate = 0.01, verbose = True):
    costs = []
    for e in range(epochs):
        error = 0
        i=0
        for x, y in zip(x_train, y_train):
            i+=1
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad)
            if i%batch_size==0:
                for layer in network:
                    layer.update(learning_rate,batch_size)
        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
        costs.append(error)
    return costs
