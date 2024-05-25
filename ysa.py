import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("housing.csv",delim_whitespace=True)

X = dataset.iloc[:,0:13]
Y = dataset.iloc[:,13]

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

input_num = x_train.shape[1]
output_num = 1

    
model = Sequential()

model.add(Dense(8, input_dim=input_num))
model.add(Activation('sigmoid'))
model.add(Dense(8))
model.add(Activation('sigmoid'))
model.add(Dense(4))
model.add(Activation('sigmoid'))
model.add(Dense(output_num))

model.summary()



model.compile(optimizer='adam',loss='mse')
history = model.fit(x_train,y_train,batch_size=16,validation_data=(x_test,y_test),epochs=150)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Lineer Regresyon Modeli
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Lineer Regresyon Modelinin Kayıplarını Hesaplama
y_train_pred_linear = linear_model.predict(x_train)
y_test_pred_linear = linear_model.predict(x_test)

train_loss_linear = ((y_train - y_train_pred_linear) ** 2).mean()
val_loss_linear = ((y_test - y_test_pred_linear) ** 2).mean()

# Lineer Regresyon Modelinin Performansını Görselleştirme
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_test_pred_linear, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Fit')
plt.title('Linear Regression Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()


# Katmanlar ve nöron sayıları
layers = ["Girdi (13)", "Gizli 1 (8)", "Gizli 2 (8)", "Gizli 3 (4)", "Çıkış (1)"]
neurons = [13, 8, 8, 4, 1]

# Diyagram oluşturma
plt.figure(figsize=(8, 6))  
layer_distance = 1
neuron_distance = 1

for i, layer in enumerate(layers):
    # Her bir katmanda nöronları çiz
    for j in range(neurons[i]):
        x = i * layer_distance
        y = j * neuron_distance
        plt.scatter(x, y, color='blue', zorder=2)  # Nöronları çiz
        
        # Gizli katmanlarda sigmoid aktivasyon fonksiyonunu göster
        if "Gizli" in layer:
            plt.gca().add_patch(plt.Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, linewidth=1, edgecolor='black', facecolor='red')) 
            plt.text(x, y, "Sigmoid", fontsize=8, ha='center', va='center', color='white', zorder=3)
        
        # Bağlantıları çiz (katmanlar arasında)
        if i > 0:
            for k in range(neurons[i-1]):
                x_prev = (i - 1) * layer_distance
                y_prev = k * neuron_distance
                plt.plot([x_prev, x], [y_prev, y], color='gray', zorder=1)  # Bağlantılar düzeltildi

# Eksenleri düzenleme
plt.yticks([])
plt.xticks(range(len(layers)), layers)
plt.title('Çok Katmanlı Algılayıcı (MLP) Diyagramı')
plt.xlabel('Katmanlar')
plt.xlim(-0.5, len(layers) - 0.5)
plt.ylim(-0.5, max(neurons) * neuron_distance - 0.5)
plt.gca().invert_yaxis()  # Y eksenini ters çevirme
plt.grid(False)
plt.tight_layout()
plt.show()


train_preds = model.predict(x_train)
print("Eğitim R2 Skoru : ",r2_score(y_train,train_preds))

test_preds = model.predict(x_test)
print("Test R2 Skoru : ",r2_score(y_test,test_preds))

