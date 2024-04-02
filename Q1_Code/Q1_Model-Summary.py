import tensorflow as tf
import os 

os.chdir('/home/coder/workspace/Data/Simulator_Data/')

def display_input_shape(model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Display input shape
    input_shape = model.input_shape
    print("Input shape of the model:")
    print(input_shape)

# Provide the path to your .h5 Keras model file
for output in range(10):
    model_name = f'Model_{output}.h5'
    print(f'Model{output}:')
    display_input_shape(model_name)
