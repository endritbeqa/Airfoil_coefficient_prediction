from flax import linen as nn


class Head(nn.Module):
    """
    Multilayer perceptron layer

    Attributes
    ----------
    hidden_size: int
        dimensionality of embeddings
    dim_mlp: int
        dimensionality of multilayer perceptron layer
    dropout_rate: float
        Dropout rate. Float between 0 and 1.
    """

    hidden_sizes: list
    num_targets: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train):
        """
        Applies MLP layer on the inputs.

        Parameters
        ----------
        x: jnp.ndarray
            Input of MLP layer
        deterministic: bool
            If false, the attention weight is masked randomly using dropout,
            whereas if true, the attention weights are deterministic.

        Returns
        -------
            Output of MLP layer.
        """

        b, dim_mlp, vector_length = x.shape

        x = x.reshape(b, dim_mlp*vector_length)

        for i, num_nodes in enumerate(self.hidden_sizes):

            x = nn.LayerNorm()(x)
            x = nn.Dense(features=num_nodes)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic= not train)(x)

            
        x = nn.Dense(features=self.num_targets)(x)

        return x
