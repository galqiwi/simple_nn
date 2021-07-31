class Model:
    def __init__(self):
        self.layers = list()
        self.run_data = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_v):
        self.run_data = list()
        self.run_data.append(input_v)
        for layer in self.layers:
            self.run_data.append(
                layer.forward(self.run_data[-1])
            )

        return self.run_data[-1]

    def fit(self, input_v, desired_v, loss, optimizer, return_data_gradient = False):
        output_v = self.forward(input_v)
        error = loss.forward(output_v, desired_v)
        output_grad_v = loss.backward(output_v, desired_v, error)

        for layer, layer_input_v, layer_output_v in zip(self.layers[::-1], self.run_data[:-1][::-1], self.run_data[1:][::-1]):
            output_grad_v = layer.backward(layer_input_v, layer_output_v, output_grad_v)

        hidden_values = list()
        hidden_values_grad = list()

        for layer in self.layers:
            hidden_values += layer.get_values()
            hidden_values_grad += layer.get_gradients()

        optimizer.optimize(hidden_values, hidden_values_grad)
        
        if return_data_gradient:
            return (error, output_grad_v)
        else:
            return error
