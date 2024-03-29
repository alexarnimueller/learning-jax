import math
from typing import Callable, Sequence

import flax.linen as nn
import numpy as np

import jax
import jax.numpy as jnp
from utils import expand_mask, init_func


class SimpleClassifier(nn.Module):
    num_hidden: int  # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # while defining necessary layers (instead of in def setup())
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


class BaseNetwork(nn.Module):
    act_fn: nn.Module
    num_classes: int = 10
    hidden_sizes: Sequence = (512, 256, 256, 128)
    kernel_init: Callable = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x, return_activations=False):
        x = x.reshape(x.shape[0], -1)  # Reshape images to a flat vector
        # We collect all activations throughout the network for later visualizations
        # Remember that in jitted functions, unused tensors will anyways be removed.
        activations = []
        for hd in self.hidden_sizes:
            x = nn.Dense(hd, kernel_init=init_func(x), bias_init=init_func(x))(x)
            activations.append(x)
            x = self.act_fn(x)
            activations.append(x)
        x = nn.Dense(
            self.num_classes, kernel_init=init_func(x), bias_init=init_func(x)
        )(x)
        return x if not return_activations else (x, activations)


googlenet_kernel_init = nn.initializers.kaiming_normal()


class InceptionBlock(nn.Module):
    c_red: dict  # Dictionary of reduced dimensionalities with keys "1x1", "3x3", "5x5", and "max"
    c_out: dict  # Dictionary of output feature sizes with keys "1x1", "3x3", "5x5", and "max"
    act_fn: callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        # 1x1 convolution branch
        x_1x1 = nn.Conv(
            self.c_out["1x1"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_1x1 = nn.BatchNorm()(x_1x1, use_running_average=not train)
        x_1x1 = self.act_fn(x_1x1)

        # 3x3 convolution branch
        x_3x3 = nn.Conv(
            self.c_red["3x3"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        x_3x3 = self.act_fn(x_3x3)
        x_3x3 = nn.Conv(
            self.c_out["3x3"],
            kernel_size=(3, 3),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x_3x3)
        x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        x_3x3 = self.act_fn(x_3x3)

        # 5x5 convolution branch
        x_5x5 = nn.Conv(
            self.c_red["5x5"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        x_5x5 = self.act_fn(x_5x5)
        x_5x5 = nn.Conv(
            self.c_out["5x5"],
            kernel_size=(5, 5),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x_5x5)
        x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        x_5x5 = self.act_fn(x_5x5)

        # Max-pool branch
        x_max = nn.max_pool(x, (3, 3), strides=(2, 2))
        x_max = nn.Conv(
            self.c_out["max"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_max = nn.BatchNorm()(x_max, use_running_average=not train)
        x_max = self.act_fn(x_max)

        x_out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)
        return x_out


class GoogleNet(nn.Module):
    num_classes: int
    act_fn: callable

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            64, kernel_size=(3, 3), kernel_init=googlenet_kernel_init, use_bias=False
        )(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)

        # Stacking inception blocks
        inception_blocks = [
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.act_fn,
            ),
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.act_fn,
            ),
            lambda inp: nn.max_pool(inp, (3, 3), strides=(2, 2)),  # 32x32 => 16x16
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.act_fn,
            ),
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            InceptionBlock(
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.act_fn,
            ),
            lambda inp: nn.max_pool(inp, (3, 3), strides=(2, 2)),  # 16x16 => 8x8
            InceptionBlock(
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
            InceptionBlock(
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.act_fn,
            ),
        ]
        for block in inception_blocks:
            x = block(x, train=train) if isinstance(block, InceptionBlock) else block(x)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


class EncoderBlock(nn.Module):
    input_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for ll in self.linear:
            linear_out = (
                ll(linear_out)
                if not isinstance(ll, nn.Dropout)
                else ll(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.layers = [
            EncoderBlock(
                self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, mask=None, train=True):
        for ll in self.layers:
            x = ll(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for ll in self.layers:
            _, attn_map = ll.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = ll(x, mask=mask, train=train)
        return attention_maps


class PositionalEncoding(nn.Module):
    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, : x.shape[1]]
        return x


class TransformerPredictor(nn.Module):
    model_dim: int  # Hidden dimensionality to use inside the Transformer
    num_classes: int  # Number of classes to predict per sequence element
    num_heads: int  # Number of heads to use in the Multi-Head Attention blocks
    num_layers: int  # Number of encoder blocks to use
    dropout_prob: float = 0.0  # Dropout to apply inside the model
    input_dropout_prob: float = 0.0  # Dropout to apply on the input features

    def setup(self):
        # Input dim -> Model dim
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.model_dim)
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(self.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
        )
        # Output classifier per sequence lement
        self.output_net = [
            nn.Dense(self.model_dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes),
        ]

    def __call__(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
            train - If True, dropout is stochastic
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask, train=train)
        for ll in self.output_net:
            x = (
                ll(x)
                if not isinstance(ll, nn.Dropout)
                else ll(x, deterministic=not train)
            )
        return x

    def get_attention_maps(
        self, x, mask=None, add_positional_encoding=True, train=True
    ):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask, train=train)
        return attention_maps
