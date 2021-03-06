"""Comparison tests between pre and post translation layers."""
from hypothesis import given, settings, reproduce_failure
from hypothesis.strategies import floats, integers, data, booleans
from hypothesis.extra.numpy import arrays

import numpy as np
import keras
import torch

from ktp import translate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@given(integers(min_value=1, max_value=5), integers(min_value=2, max_value=30), integers(min_value=2, max_value=30), integers(min_value=1, max_value=15))
def test_1d_local_convolution(batch_exp, input_height, in_channels, filters):
    """Compare 1D keras and PyTorch layers to flattened 2D keras and PyTorch layers.
    Currently a 1D PyTorch layer is just a flattened 2D PyTorch layer, so really this is a test that the translation process works correctly.
    """
    batch_size = 2 ** batch_exp
    kernel_size = np.random.randint(1, input_height)
    input1_shape = (input_height, in_channels)
    input2_shape = (input_height, 1, in_channels)

    # First model
    keras_1d_model = keras.Sequential()
    keras_1d_layer = keras.layers.LocallyConnected1D(filters, kernel_size, input_shape=input1_shape)
    keras_1d_model.add(keras_1d_layer)

    # Flattened keras 2D locally connected
    keras_flattened_2d_model = keras.Sequential()
    keras_flattened_2d_layer = keras.layers.LocallyConnected2D(filters, (kernel_size, 1), input_shape=input2_shape)
    keras_flattened_2d_model.add(keras_flattened_2d_layer)

    # Copy weights
    weights = np.random.uniform(-100, 100, keras_1d_layer.kernel_shape).astype(np.float32)
    keras_1d_model.set_weights([weights])
    keras_flattened_2d_model.set_weights([weights.reshape(keras_flattened_2d_layer.kernel_shape)])

    # Flattened PyTorch 2D locally connected.
    torch_flattened_model = translate.translate_layer(keras_flattened_2d_layer)[0].to(device)
    torch_1d_model = translate.translate_1d_locally_connected(keras_1d_layer)[0].to(device)

    # Generate inputs.
    keras_1d_input = np.arange(batch_size * input_height * in_channels).reshape((batch_size, input_height, in_channels)).astype(np.float32)
    keras_2d_input = keras_1d_input.reshape((batch_size, input_height, 1, in_channels))
    torch_input = torch.Tensor(keras_2d_input).to(device)

    # Compute and compare outputs.
    keras_1d_output = keras_1d_model.predict(keras_1d_input)
    keras_flattened_output = keras_flattened_2d_model.predict(keras_2d_input)
    torch_flattened_output = torch_flattened_model(torch_input).cpu().data.numpy()
    torch_1d_output = torch_1d_model(torch_input).cpu().data.numpy()
    output_shape = keras_flattened_output.shape
    reshaped_1d = keras_1d_output.reshape(output_shape)
    assert all((
        (reshaped_1d == keras_flattened_output).all(),
        np.isclose(reshaped_1d, torch_flattened_output, atol=1e-3, rtol=1e-3).all(),
        np.isclose(reshaped_1d, torch_1d_output, atol=1e-3, rtol=1e-3).all()))


@given(integers(min_value=1, max_value=5), integers(min_value=2, max_value=30), integers(min_value=2, max_value=30), integers(min_value=1, max_value=15),
       integers(min_value=1, max_value=20), data(), booleans())
def test_2d_local_convolution(batch_exp, input_height, input_width, in_channels, filters, drawer, is_first):
    """Compare keras and PyTorch 2D locally connected layers on randomly generated values."""
    batch_size = 2 ** batch_exp
    if is_first:
        data_format = "channels_first"
    else:
        data_format = "channels_last"
    kernel_height = drawer.draw(integers(min_value=1, max_value=input_height-1))
    kernel_width = drawer.draw(integers(min_value=1, max_value=input_width-1))

    if data_format == "channels_last":
        input_shape = (input_height, input_width, in_channels)
    else:
        input_shape = (in_channels, input_height, input_width)

    # Create keras model
    keras_model = keras.Sequential()
    keras_local = keras.layers.LocallyConnected2D(filters, (kernel_height, kernel_width),
                                                  use_bias=True,
                                                  bias_initializer="random_uniform",
                                                  input_shape=input_shape,
                                                  data_format=data_format)
    keras_model.add(keras_local)

    # Setup weights
    weights = drawer.draw(arrays(shape=keras_local.kernel.shape.as_list(), dtype=np.float32, elements=floats(min_value=-1e6, max_value=1e6)))
    keras_model.set_weights([weights])

    # Translate keras to PyTorch model
    torch_model = translate.translate_layer(keras_local)[0].to(device)

    keras_input = np.random.uniform(-10, 10, (batch_size,) + input_shape).astype(np.float32)
    keras_output = keras_model.predict(keras_input)

    torch_input = torch.Tensor(keras_input).to(device)
    torch_output = torch_model(torch_input).cpu().data.numpy()
    nan_fill = np.random.uniform(-5, 5)
    assert (np.isnan(keras_output) == np.isnan(torch_output)).all()
    keras_output[np.isnan(keras_output)] = nan_fill
    torch_output[np.isnan(torch_output)] = nan_fill
    assert np.isclose(keras_output, torch_output, atol=1e-3, rtol=1e-3).all()


@given(integers(min_value=1, max_value=5), integers(min_value=1, max_value=100), integers(min_value=1, max_value=100), integers(min_value=1))
def test_dense(batch_exp, input_size, output_size, seed):
    batch_size = 2 ** batch_exp
    np.random.seed(seed)
    keras_model = keras.Sequential()
    keras_dense = keras.layers.Dense(output_size, input_shape=(input_size, ), use_bias=True, bias_initializer='ones')
    keras_model.add(keras_dense)

    torch_model = translate.translate_layer(keras_dense)[0].to(device)
    keras_input = np.random.uniform(-100, 100, (batch_size, input_size)).astype(np.float32)
    keras_output = keras_model.predict(keras_input)

    torch_input = torch.Tensor(keras_input).to(device)
    torch_output = torch_model(torch_input).cpu().data.numpy()
    comparison = np.isclose(keras_output, torch_output, atol=1e-4, rtol=1e-4).all()
    assert comparison


def test_conv1d():
    """Test 1D convolution translation."""
    filters = 3
    kernel_size = 2
    strides = 1
    batch_size = 2
    in_channels = 3
    input_size = 5
    input_shape = (batch_size, input_size, in_channels)

    keras_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True, bias_initializer="ones")
    input_layer = keras.Input(batch_shape=input_shape)
    keras_model = keras.models.Model(input=input_layer, outputs=keras_layer(input_layer))

    new_weights = np.arange(18).reshape(2, 3, 3)
    keras_layer.set_weights([new_weights, keras_layer.get_weights()[1]])

    kinput = np.arange(batch_size * input_size * in_channels).reshape(input_shape)
    kout = keras_model.predict(kinput)

    torch_model, _ = translate.translate_layer(keras_layer)
    tinput = torch.Tensor(kinput).permute(0, 2, 1)
    tout = torch_model(tinput).permute(0, 2, 1)
    assert np.isclose(kout, tout.cpu().data.numpy()).all()
