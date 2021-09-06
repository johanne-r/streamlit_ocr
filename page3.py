import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from PIL import Image
import tensorflow as tf
from asrtoolkit import cer

def page3():
    st.title("Démonstration")

    @st.cache
    def get_data():
        df = pd.read_csv('top100.csv')
        return df

    path_images = 'raw'

    data = get_data()

    @st.cache
    def get_image(data, path_images):
        X = []
        for img_id in data["image_id"]:
            my_image = cv2.imread(path_images + '/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
            my_image = cv2.resize(my_image, dsize = (65,65), interpolation = cv2.INTER_LINEAR)
            X.append(my_image)
            return X

    X = get_image(data, path_images)

    X = np.array(X)
    X = X.reshape([-1,65,65,1])

    y = data["text"]
    y = np.array(y)
    y = y.reshape(-1,1)

    from sklearn import preprocessing

    enc = preprocessing.OrdinalEncoder(categories='auto')
    enc.fit(y)
    target = enc.transform(y)



    st.write("1. Choisir un modèle dans la liste déroulante.")
    st.write("2. Sélectionner ensuite une image à tester.")
    st.write("3. Enfin, cliquer sur le bouton ci-dessous pour lancer le modèle choisi et afficher les résultats.")

    choix_modele = st.selectbox("Choisissez votre modèle", options = ["LeNet", "Détection d'objet"])

    add_selectbox = st.selectbox("Choisissez une image dans la galerie", sorted(["a01-000u-00-00","a01-000u-00-02","a01-000u-04-02","b01-136-07-00","p03-023-02-02","g01-043-07-02","l04-071-00-04","a01-000u-01-01","a01-000u-02-03","a01-000x-00-06","r06-137-10-00"]))
    img_id = add_selectbox


    click = st.button("Lancer le modèle et afficher les résultats")
    
    if click:
        if choix_modele == "LeNet":
            X_test = []

            my_image = cv2.imread(path_images + '/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
            my_image = cv2.resize(my_image, dsize = (65,65), interpolation = cv2.INTER_LINEAR)
            X_test.append(my_image)

            X_test = np.array(X_test)
            X_test = X_test.reshape([-1,65,65,1])
            
            model = keras.models.load_model('ocr_model_test.h5',custom_objects={'tf': tf})
            y_pred = model.predict(X_test/1.0)
            
            y_pred_class = y_pred.argmax(axis=1)

            target_new = target.reshape([-1])
            y_new = y.reshape([-1])

            index_pred = list(target_new).index(y_pred_class)

            img = Image.open(path_images + '/' + img_id + '.png')
            
            st.write("Vous avez choisi l'image ci-dessous:")
            st.image(img, width = 90)

            st.write("Et voici les résultats de votre modèle :")

            col1, col2 = st.beta_columns(2)
            with col1:
                st.write('**True Label**')
                st.write(str(data[data.image_id == img_id]["text"].values[0]))
            with col2:
                st.write('**Prediction**')
                st.write(str(y_new[index_pred]))
        else:
            X_test = []

            my_image = cv2.imread(path_images + '/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
            my_image = cv2.resize(my_image, dsize = (32,128), interpolation = cv2.INTER_LINEAR)
            X_test.append(my_image)

            X_test = np.array(X_test)

            img = Image.open(path_images + '/' + img_id + '.png')

            st.write("Vous avez choisi l'image ci-dessous:")
            st.image(img, width = 90)

            model = tf.keras.models.load_model('htr_model_cer.h5', custom_objects={'tf': tf})
            
            def loss(labels, logits):
                return tf.reduce_mean(
                        tf.nn.ctc_loss(
                            labels = labels,
                            logits = logits,
                            logit_length = [logits.shape[1]]*logits.shape[0],
                            label_length = None,
                            logits_time_major = False,
                            blank_index=-1
                        )
                    )

            import string
            charList = list(string.ascii_letters)+[' ']

            def encode_labels(labels, charList):
                # Hash Table
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        charList,
                        np.arange(len(charList)),
                        value_dtype=tf.int32
                    ),
                    0,
                    name='char2id'
                )
                return table.lookup(
                tf.compat.v1.string_split(labels, delimiter=''))   


            def decode_codes(codes, charList):
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        np.arange(len(charList)),
                        charList,
                        key_dtype=tf.int32
                    ),
                    '',
                    name='id2char'
                )
                return table.lookup(codes)

            def greedy_decoder(logits):
            # ctc beam search decoder
                predicted_codes, _ = tf.nn.ctc_greedy_decoder(
                    # shape of tensor [max_time x batch_size x  num_classes] 
                    tf.transpose(logits, (1, 0, 2)),
                    [logits.shape[1]]*logits.shape[0]
                )
                
                # convert to int32
                codes = tf.cast(predicted_codes[0], tf.int32)
                
                # Decode the index of caracter
                text = decode_codes(codes, charList)
                
                # Convert a SparseTensor to string
                text = tf.sparse.to_dense(text).numpy().astype(str)
                
                return list(map(lambda x: ''.join(x), text))


            X_test = tf.cast(X_test, dtype = tf.float32)
            l = greedy_decoder(model(np.expand_dims(X_test, -1)))
            
            st.write("Et voici les résultats de votre modèle :")

            col1, col2, col3 = st.beta_columns(3)
            with col1:
                st.write('**True Label**')
                st.write(str(data[data.image_id == img_id]["text"].values[0])) 

            with col2:
                st.write('**Prediction**')
                st.write(str(l[0]))
            
            with col3:
                st.write("**CER**")
                st.write(round(cer(str(data[data.image_id == img_id]["text"].values[0]),str(l[0])),2))
