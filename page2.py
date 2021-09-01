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
    list_img_id = ["a01-000u-00-00","a01-000u-00-02","a01-000u-01-01","a01-000u-02-03","a01-000u-04-02","a01-000x-00-06","a01-000x-02-01","b01-136-07-00","f07-032a-05-08","f07-032a-06-03","g01-043-07-02","g06-037k-00-08","g06-037k-01-01","k02-036-03-05","l04-071-00-04","m01-038-06-04","m01-115-04-06","m01-115-04-06","p03-023-02-02","r06-137-10-00","r06-143-00-04"]

    if click:
        for i in range(12):
            img_id = np.random.choice(list_img_id,1)
            img1 = Image.open(path_images + '/' + img_id[0] + '.png')
            list_img.append(img1)

        st.image(list_img, width = 90)


