from tensorflow.keras.models import load_model

model = load_model('rnn_model.h5')
print(model.input_shape)