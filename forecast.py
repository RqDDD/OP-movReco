from keras.models import load_model

model = load_model('my_model.h5')


def forecast(model, batch_size, X):
        X = X.reshape(1, X.shape[0], X.shape[1])
        yhat = model.predict(X, batch_size=batch_size)
        return yhat
