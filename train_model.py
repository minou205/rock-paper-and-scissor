import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input


DATA_DIR= "data"
MODEL_OUT= "rps_model_v2.keras"
IMG_SIZE= (224, 224)
BATCH= 16
CLASSES= ["rock", "paper", "scissors"]
SEED= 42
EPOCHS_HEAD= 25
LR_HEAD= 1e-3
EPOCHS_FT= 35
LR_FT= 5e-5
UNFREEZE= 30
tf.random.set_seed(SEED)

def load_split(split_name, shuffle=True):
    path = os.path.join(DATA_DIR, split_name)
    if not os.path.isdir(path):
        sys.exit(f"❌  folder not found: {path}")
    ds=tf.keras.utils.image_dataset_from_directory(path,labels="inferred",label_mode="categorical",class_names=CLASSES,image_size=IMG_SIZE,batch_size=BATCH,shuffle=shuffle,seed=SEED,)
    n=sum(1 for _ in ds.unbatch())
    print(f"[{split_name:5s}]  {n} images")
    return ds

print(" loading data...")
raw_train=load_split("train",shuffle=True)
raw_val=load_split("val",shuffle=False)
raw_test=load_split("test",shuffle=False)
print()


@tf.function
def augment(image, label):
    image=tf.cast(image, tf.float32)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_flip_up_down(image)
    image=tf.image.random_brightness(image,0.3)
    image=tf.image.random_contrast(image,0.7,1.3)
    image=tf.clip_by_value(image,0.0,255.0)
    return image,label

@tf.function
def random_rotate(image,label):
    k=tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32)
    image=tf.image.rot90(image,k)
    return image,label

@tf.function
def preprocess_map(image,label):
    image=tf.cast(image,tf.float32)
    image=preprocess_input(image)
    return image,label

AUTOTUNE=tf.data.AUTOTUNE
train_ds=(raw_train.map(augment,num_parallel_calls=AUTOTUNE).map(random_rotate,num_parallel_calls=AUTOTUNE).map(preprocess_map,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
val_ds=raw_val.map(preprocess_map,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds= raw_test.map(preprocess_map,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def build_model(num_classes=3):
    base = MobileNetV2(input_shape=(*IMG_SIZE,3),include_top=False,weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE,3))
    x=base(inputs, training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.50)(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.30)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    return models.Model(inputs,outputs,name="RPS_MobileNetV2"),base

model, base_model=build_model()
model.summary()

def make_callbacks():
    return [
        callbacks.EarlyStopping(monitor="val_accuracy",patience=8,verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.4,patience=4,min_lr=1e-8,verbose=1),
        callbacks.ModelCheckpoint(MODEL_OUT,monitor="val_accuracy",save_best_only=True,verbose=1)
    ]

print("\n" + "─"*55)
print("step 1: Training the Head")
print("─"*55 + "\n")
model.compile(optimizer=tf.keras.optimizers.Adam(LR_HEAD),loss="categorical_crossentropy",metrics=["accuracy"])
h1 = model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS_HEAD,callbacks=make_callbacks(),verbose=1)
print(f"\ndownloading best model from {MODEL_OUT}")
model=tf.keras.models.load_model(MODEL_OUT)
base_model=model.layers[1]
print("\n" + "─"*55)
print(f"step 2: Fine-Tuning")
print("─"*55 + "\n")
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(LR_FT),loss="categorical_crossentropy",metrics=["accuracy"],)
h2 = model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS_FT,callbacks=make_callbacks(),verbose=1,)
print(f"\ndownloading best final model: {MODEL_OUT}")
model = tf.keras.models.load_model(MODEL_OUT)
print("\nevaluating on test data...")
t_loss, t_acc = model.evaluate(test_ds, verbose=1)

def merge(ha,hb,key):
    return ha.history.get(key,[])+hb.history.get(key,[])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RPS MobileNetV2 – Training Report",fontsize=13,fontweight="bold")
ft_start = len(h1.history["accuracy"])

for ax,(tr_k, val_k),title in zip(axes,[("accuracy","val_accuracy"),("loss","val_loss")],["Accuracy","Loss"]):
    tr=merge(h1, h2, tr_k)
    val=merge(h1, h2, val_k)
    ep=range(1, len(tr)+1)
    ax.plot(ep,tr,label="Train",color="steelblue",lw=2)
    ax.plot(ep,val,label="Validation",color="tomato",lw=2)
    ax.axvline(x=ft_start,color="gray", ls="--",alpha=0.7,label="Fine-Tune Start")
    ax.set_title(title);ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_report.png", dpi=130)
plt.show()
print(f"\n{'═'*55}")
print(f"Training completed!")
print(f"Test Accuracy : {t_acc*100:.2f}%")
print(f"Model: {MODEL_OUT}")
print(f"Curves: training_report.png")
print(f"{'═'*55}")
print("\nNext step: python rps_game.py\n")
