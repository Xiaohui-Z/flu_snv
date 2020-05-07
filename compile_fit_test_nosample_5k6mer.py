import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import pandas as pd
import numpy as np

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 114.
    # x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=2)
    return x, y


allmatrix = np.array(pd.read_csv('5k6mer_allmatrix_com.csv', header=0, index_col=0, low_memory= True))
target = np.loadtxt('5k6mer_target_com', dtype='int32')
print("allmatrix shape: {}；target shape: {}".format(allmatrix.shape, target.shape))

x = tf.convert_to_tensor(allmatrix, dtype=tf.int32)
y = tf.convert_to_tensor(target, dtype=tf.int32)
idx = tf.range(19826)
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:13826]), tf.gather(y, idx[:13826])
x_val, y_val = tf.gather(x, idx[-6000:-4500]), tf.gather(y, idx[-6000:-4500])
x_test, y_test = tf.gather(x, idx[-4500:]), tf.gather(y, idx[-4500:])


batchsz = 256
#(x, y), (x_val, y_val) = datasets.mnist.load_data()
#print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dropout(0.5),
                      # layers.Dense(64, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dropout(0.5),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10,activation='relu'),
                      layers.Dense(2)])
network.build(input_shape=(None, 2080))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db, epochs=40, validation_data=ds_val,
            validation_steps=2)

network.evaluate(db_test)

# sample = next(iter(db_test))
# x = sample[0]
# y = sample[1]  # one-hot
# pred = network.predict(x)  # [b, 10]
# # convert back to number
# y = tf.argmax(y, axis=1)
# pred = tf.argmax(pred, axis=1)
#
# print(pred)
# print(y)

y_total=[]
y_pred=[]
for step, (x, y) in enumerate(db_test):
    out = network.predict(x)
    # [b, 10] => [b]
    pred = tf.argmax(out, axis=1)
    # pred = tf.cast(pred, dtype=tf.int32)
    y_pred=tf.concat([y_pred, pred], axis=0)
    y = tf.argmax(y, axis=1)
    y_total=tf.concat([y_total, y], axis=0)

print(y_total.shape, y_pred.shape)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, y_total)
acc= accuracy_score(y_pred, y_total)
print("test data accuracy: ", acc)
print("cm is :", cm)