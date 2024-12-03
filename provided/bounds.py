import torch
from provided.network import SimpleNeuralNetwork
from torch import Tensor


class IntervalBoundPropagation:

    def __init__(self, network: SimpleNeuralNetwork):
        self.network = network

    def compute_bounds_forward(self, input_bounds: Tensor) -> Tensor:
        """
        Computes the forward propagation of interval bounds through each layer
        of the network sequentially.

        Args:
            input_bounds (Tensor): The input bounds represented as a tensor with shape
                (batch_size, input_dim, 2), where the last dimension holds the lower and
                upper bounds for each input feature.

        Returns:
            Tensor: The propagated bounds after passing through all layers of the network,
            with shape (batch_size, output_dim, 2). The last dimension holds the lower and
            upper bounds for each output feature.
        """
        # Compute bounds through the first layer
        layer1_bounds = self.propagate_bounds(
            input_bounds, self.network.W1, self.network.b1
        )

        # Compute bounds through the second layer
        layer2_bounds = self.propagate_bounds(
            layer1_bounds.to(torch.float32), self.network.W2, self.network.b2
        )

        # Compute bounds through the third (output) layer
        output_bounds = self.propagate_bounds(
            layer2_bounds.to(torch.float32), self.network.W3, self.network.b3
        )

        return output_bounds

    @staticmethod
    def propagate_bounds(input_bounds: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
        """
        Propagates interval bounds through a linear layer in a neural network.

        Args:
            input_bounds (Tensor): The input bounds represented as a tensor with shape
                (batch_size, input_dim, 2), where the last dimension holds the lower and
                upper bounds for each input feature.
            weights (Tensor): The weights of the linear layer with shape (input_dim, output_dim).
            bias (Tensor): The bias of the linear layer with shape (output_dim,).

        Returns:
            Tensor: The propagated bounds after the forward pass through the linear layer,
            with shape (batch_size, output_dim, 2). The last dimension holds the lower and
            upper bounds for each output feature.

        Example:
            Given a network definition with input_dim=3 and an initial layer's weights:

            input_bounds.shape = (32, 3, 2)  # batch_size=32, input_dim=3
            weights.shape = (4, 3)  # input_dim=3, output_dim=4
            bias.shape = (4,)  # output_dim=4
            output = propagate_bounds(input_bounds, weights, bias)
            output.shape  # (32, 4, 2)
        """
        batch_size, input_dim, _ = input_bounds.shape
        output_dim = weights.shape[0]

        bounds_out = torch.empty(
            (batch_size, output_dim, 2),
            device="cpu",
            dtype=torch.float64,
        )

        ########### YOUR CODE HERE ############

        # Separate lower and upper bounds
        input_lower = input_bounds[:, :, 0]  # (batch_size, input_dim)
        input_upper = input_bounds[:, :, 1]  # (batch_size, input_dim)

        # Compute the linear pass
        # Can also use einops.einsum() in place of torch.einsum()
        for i in range(batch_size):
            for j in range(output_dim):
                lower_bound = upper_bound = bias[0][j]
                for k in range(input_dim):
                    if weights[j, k] >= 0:
                        lower_bound += weights[j, k] * input_lower[i, k]
                        upper_bound += weights[j, k] * input_upper[i, k]
                    else:
                        lower_bound += weights[j, k] * input_upper[i, k]
                        upper_bound += weights[j, k] * input_lower[i, k]
                bounds_out[i, j, 0] = lower_bound
                bounds_out[i, j, 1] = upper_bound

        ########### END YOUR CODE  ############

        return bounds_out
