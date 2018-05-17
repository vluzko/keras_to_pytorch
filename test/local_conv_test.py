"""Comparison tests between different implementations of locally connected layers.
Honestly I probably should have just defined a commutative diagram and tested everything along that, but oh well.
"""
import numpy as np
import keras
import torch

from ktp import translate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: Change to Hypothesis
def test_compare_1d_and_flattened():
    """Compare 1D keras and PyTorch layers to flattened 2D keras and PyTorch layers.
    Currently a 1D PyTorch layer is just a flattened 2D PyTorch layer, so really this is a test that the translation process works correctly.
    """
    for i in range(10):
        input_height = np.random.randint(2, 10)
        in_channels = np.random.randint(1, 10)
        kernel_size = np.random.randint(1, input_height)
        filters = np.random.randint(1, 20)
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
        weights = np.random.uniform(-10, 10, keras_1d_layer.kernel_shape).astype(np.float32)
        keras_1d_model.set_weights([weights])
        keras_flattened_2d_model.set_weights([weights.reshape(keras_flattened_2d_layer.kernel_shape)])

        # Flattened PyTorch 2D locally connected.
        torch_flattened_model = translate.translate_2d_locally_connected(keras_flattened_2d_layer)[0].to(device)
        torch_1d_model = translate.translate_1d_locally_connected(keras_1d_layer)[0].to(device)

        # Generate inputs.
        keras_1d_input = np.arange(input_height * in_channels).reshape((1, input_height, in_channels)).astype(np.float32)
        keras_2d_input = keras_1d_input.reshape((1, input_height, 1, in_channels))
        torch_input = torch.Tensor(keras_2d_input).to(device)

        # Compute and compare outputs.
        keras_1d_output = keras_1d_model.predict(keras_1d_input)
        keras_flattened_output = keras_flattened_2d_model.predict(keras_2d_input)
        torch_flattened_output = torch_flattened_model(torch_input).cpu().data.numpy()
        torch_1d_output = torch_1d_model(torch_input).cpu().data.numpy()
        output_shape = keras_flattened_output.shape
        assert all((
            (keras_1d_output.reshape(output_shape) == keras_flattened_output).all(),
            np.isclose(keras_1d_output.reshape(keras_flattened_output.shape), torch_flattened_output, atol=1e-4, rtol=1e-4).all(),
            np.isclose(keras_1d_output.reshape(keras_flattened_output.shape), torch_1d_output, atol=1e-4, rtol=1e-4).all()))


# TODO: Change to Hypothesis
def test_compare_2d_local_full():
    """Compare keras and PyTorch 2D locally connected layers on randomly generated values."""
    for i in range(10):
        data_format = np.random.choice(("channels_first", "channels_last"))
        input_height = np.random.randint(2, 15)
        input_width = np.random.randint(2, 15)
        in_channels = np.random.randint(2, 15)
        kernel_height = np.random.randint(1, input_height)
        kernel_width = np.random.randint(1, input_width)
        filters = np.random.randint(1, 20)

        if data_format == "channels_last":
            input_shape = (input_height, input_width, in_channels)
        else:
            input_shape = (in_channels, input_height, input_width)

        # Create keras model
        keras_model = keras.Sequential()
        keras_local = keras.layers.LocallyConnected2D(filters, (kernel_height, kernel_width),
                                                      use_bias=False,
                                                      input_shape=input_shape,
                                                      data_format=data_format)
        keras_model.add(keras_local)

        # Setup weights
        weights = np.random.uniform(-10, 10, keras_local.kernel.shape.as_list()).astype(np.float32)
        keras_model.set_weights([weights])

        # Translate keras to PyTorch model
        torch_model = translate.translate_2d_locally_connected(keras_local)[0].to(device)

        keras_input = np.random.uniform(-10, 10, (1,) + input_shape).astype(np.float32)
        keras_output = keras_model.predict(keras_input)

        torch_input = torch.Tensor(keras_input).to(device)
        torch_output = torch_model(torch_input).cpu().data.numpy()
        assert np.isclose(keras_output, torch_output, atol=1e-4, rtol=1e-4).all()
