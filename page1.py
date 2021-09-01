import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def page1():
    st.title("Introduction")
    st.subheader("Description du projet")
    st.write("Le but de ce projet est de mettre en place des méthodes de Deep Learning et de Computer Vision pour transcrire les textes présents sur des images.")
    st.write("Nous avons mis en place deux modèles afin d'atteindre cet objectif.")
  
    st.subheader("Description du jeu de données")
    st.write("Notre jeu de données est un ensemble d'images scannées et labellisées disponible au public sur le lien ci-dessous.")
    st.write("Cliquez [ici](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) pour accéder aux données.")

    st.write("Pour les besoins du projet, nous avons uniquement utilisé les mots isolés et leurs labels.")

    st.subheader("Fonctionnement de l'application")
    st.write("Pour lancer l'application, il vous suffira d'aller dans l'onglet Demonstration et de suivre les instructions..")
    st.write("L'application retournera ensuite le résultat de votre modèle.")
    

