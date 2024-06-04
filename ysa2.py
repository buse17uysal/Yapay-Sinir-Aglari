import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("housing.csv", delim_whitespace=True, header=None)


X = dataset.iloc[:, 0:13]
Y = dataset.iloc[:, 13]

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

input_num = x_train.shape[1]
output_num = 1

mlp_model = Sequential()

mlp_model.add(Dense(16, input_dim=input_num))
mlp_model.add(Activation('relu'))
mlp_model.add(Dense(12))
mlp_model.add(Activation('relu'))
mlp_model.add(Dense(8))
mlp_model.add(Activation('relu'))
mlp_model.add(Dense(4))
mlp_model.add(Activation('relu'))
mlp_model.add(Dense(output_num))

mlp_model.summary()

mlp_model.compile(optimizer='adam', loss='mse')

history = mlp_model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=200)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

train_preds_mlp = mlp_model.predict(x_train)
print("Eğitim R2 Skoru (MLP): ", r2_score(y_train, train_preds_mlp))

test_preds_mlp = mlp_model.predict(x_test)
print("Test R2 Skoru (MLP): ", r2_score(y_test, test_preds_mlp))

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

y_train_pred_linear = linear_model.predict(x_train)
y_test_pred_linear = linear_model.predict(x_test)

train_loss_linear = ((y_train - y_train_pred_linear) ** 2).mean()
val_loss_linear = ((y_test - y_test_pred_linear) ** 2).mean()

# Lineer regresyon modelinin performansının görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_test_pred_linear, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Fit')
plt.title('Linear Regression Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

# Lineer regresyon modelinin R2 skorlarının hesaplanması
print("Eğitim R2 Skoru (Linear Regression): ", r2_score(y_train, y_train_pred_linear))
print("Test R2 Skoru (Linear Regression): ", r2_score(y_test, y_test_pred_linear))
