import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image


def page2():
    st.title("Galerie")

    st.write("Bienvenue dans notre galerie ! Ici vous trouverez un aperçu des images utilisées dans le cadre de ce projet.")
    st.write("Cliquer le bouton ci-dessous pour obtenir un échantillon du jeu de données.")

    click = st.button("Afficher un échantillon du jeu de données")

    @st.cache
    def get_data():
        df = pd.read_csv('top100.csv')
        return df

    data = get_data()

    path_images = 'raw'

    list_img = []
    cv_img = []

    if click:
        for i in np.random.randint(low=0, high=data.shape[0], size=[18]):
            img_id = data.loc[i,"image_id"]
            img1 = Image.open(path_images + '/' + img_id + '.png')
            list_img.append(img1)

        st.image(list_img, width = 90)


