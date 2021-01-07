#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
import time # 時間計測用
import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

# proxyの設定 (443 port error 発生の場合こちらを使用する)
proxy_support = urllib.request.ProxyHandler({'http' : 'http://<ID>:<PASS>@<URL>.co.jp:8080',
                                            'https': 'https://<ID>:<PASS>@<URL>.co.jp:8080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

def create_model_three_layer(num_classes, input_shape):
    """
    モデル作成(3層)
    
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : int
        入力画像サイズ
    """
    inputs = x = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes,kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model

def create_model_vgg16(num_classes, input_shape):
    """
    モデル作成(VGG16)
        
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : int
        入力画像サイズ
    """
    base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                                                include_top=False,
                                                input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes)(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model

def create_model_xception(num_classes, input_shape):
    """
    モデル作成(Xception)
        
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : int
        入力画像サイズ
    """
    base_model = keras.applications.xception.Xception(weights='imagenet',
                                                      include_top=False,
                                                      input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes)(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model
    
def create_model_mobilenet(num_classes, input_shape):
    """
    モデル作成(Mobilenet)
        
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : int
        入力画像サイズ
    """
    base_model = keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                      include_top=False,
                                                      input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes)(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model
    
# モデルの前処理
def model_preprocess(num_classes,im_resize,num_data):
    # 学習データ読込(cifar10)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # 大きさが1の次元を削除
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    # 時間短縮のため使用するデータ件数を絞る
    X_train = X_train[:num_data]
    y_train = y_train[:num_data]
    
    # リサイズ
    X_train = [cv2.resize(x, (im_resize, im_resize)) for x in X_train]
    X_test = [cv2.resize(x, (im_resize, im_resize)) for x in X_test]
    
    # 特徴量の正規化(0～255から0～1の実数値に変換)
    X_train = np.asarray(X_train).astype("float32") / 255
    X_test = np.asarray(X_test).astype("float32") / 255
    
    # 正解ラベルを'one hot表現'に変形
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return X_train,y_train,X_test,y_test,num_classes,im_resize

# 学習速度評価プログラム 引数(画像枚数,入力画像サイズ(リサイズ),クラス数,エポック数,バッチサイズ)
def _main(num_data,im_resize,num_classes,epochs,batch_size,flag):
    # 入力画像サイズ
    input_shape = (32, 32, 3)
    
    # モデルの前処理
    X_train,y_train,X_test,y_test,num_classes,im_resize = model_preprocess(num_classes,im_resize,num_data)

    # 学習モデルの選択
    if (flag):
        model = create_model_three_layer(num_classes, input_shape=(im_resize,im_resize,3))# 3層
        #XXXmodel = create_model_vgg16(num_classes, input_shape=(im_resize,im_resize,3))# VGG16 input_shape=(48x48)以上
        #XXXmodel = create_model_xception(num_classes, input_shape=(im_resize,im_resize,3))# Xception input_shape=(71x71)以上
    else:
        model = create_model_mobilenet(num_classes, input_shape=(im_resize,im_resize,3))# mobile-net input_shape=(32x32)以上
    optimizer = tf.keras.optimizers.SGD(
        lr=1e-3 * batch_size, momentum=0.9, nesterov=True
    )

    model.compile(
        loss='categorical_crossentropy', # 損失関数の設定
        optimizer=optimizer, # 最適化法の指定
        metrics=['accuracy'])

    # モデル情報表示
    model.summary()
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    # 学習
    callbacks = []
    
    start = time.time()# 時間計測開始
    model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                )

    # 評価 & 評価結果出力
    score = model.evaluate(X_test, y_test, verbose=1)
    print()
    print('Total Time:', time.time() - start)# 時間計測終了
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    print("mobile-net④エポック50")
    _main(60000,32,10,50,32,False)
    print("3層⑧")
    _main(60000,64,10,5,32,True)
    print("mobile-net⑧")
    _main(60000,64,10,5,32,False)
