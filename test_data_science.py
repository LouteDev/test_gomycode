import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.sidebar.title("Apprentissage Automatique Supervisé")

pages = ["Contexte du projet", "Exploration des données", 'Analyse des données', 'Modélisation']

page = st.sidebar.radio("Sommaire :", pages)

if page == pages[0]:
    st.title(pages[0])
    st.header("Analyse et Modélisation des Prêts Personnels dans une Banque")


    st.write("Ce projet vise à analyser et à modéliser les prêts personnels dans une banque en utilisant des données clients.")

    st.write("À partir d'un ensemble de données comprenant des informations telles que l'âge, le revenu, l'expérience professionnelle et d'autres variables pertinentes, nous chercherons à comprendre les facteurs qui influent sur la décision des clients de souscrire à un prêt personnel.")

    st.write("Nous utiliserons des techniques d'analyse exploratoire des données pour identifier des tendances et des relations, puis nous développerons des modèles prédictifs pour estimer la probabilité qu'un client accepte une offre de prêt personnel.")
    st.write("L'objectif final est de fournir des informations précieuses à la banque pour mieux cibler ses offres de prêts personnels et améliorer ses taux de conversion.")


    st.image("st_data.png")




if page == pages[1]:
    st.title(pages[1])
    st.header("1. Aperçu des données initiales :")

    data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    data = data.head()
    st.dataframe(data)

    st.header("2. Explication des données :")

    st.markdown("**Age**: L'âge de la personne.")
    st.markdown("**Experience**: Le nombre d'années d'expérience professionnelle de la personne.")
    st.markdown("**Income**: Le revenu de la personne.")
    st.markdown("**Family**: Le nombre de membres de la famille de la personne.")
    st.markdown("**CCAvg**: La moyenne des dépenses sur les cartes de crédit par mois.")
    st.markdown("**Education**: Le niveau d'éducation de la personne, probablement encodé comme une variable catégorielle (par exemple, 1 pour l'éducation de base, 2 pour un diplôme universitaire, etc.).")
    st.markdown("**Mortgage**: Le montant hypothécaire de la personne.")
    st.markdown("**Personal Loan**: Une variable binaire (0 ou 1) indiquant si la personne a pris un prêt personnel.")
    st.markdown("**Securities Account**: Une variable binaire indiquant si la personne possède un compte de valeurs mobilières.")
    st.markdown("**CD Account**: Une variable binaire indiquant si la personne possède un compte à terme (CD).")
    st.markdown("**Online**: Une variable binaire indiquant si la personne utilise les services bancaires en ligne.")
    st.markdown("**CreditCard**: Une variable binaire indiquant si la personne possède une carte de crédit.")


    st.header("3. Traitement des données :")
    st.write("Nous avons éffectué des traitement suivants sur le daataset :")
    st.write("- Recherche des valeurs nulles : Nous avons vérifié s'il y avait des valeurs manquantes dans le dataset en utilisant la méthode isnull() suivie de sum() pour obtenir le total des valeurs manquantes par colonne.")
    st.write("- Suppression des colonnes inutiles : Nous avons supprimé les colonnes 'ID' et 'ZIP Code', qui semblaient ne pas être pertinentes pour notre analyse.")

    st.header("4. Identification de la variable cible :")
    st.write("- Après avoir examiné les données restantes, nous avons identifié la variable cible, qui est 'Personal Loan'. Cette variable indique si un client a souscrit à un prêt personnel (1) ou non (0).")


    st.header("5. Aperçu du Dataset après traitement :")
    data_clean = pd.read_csv('Bank_Personal_Loan_Modelling_clean.csv')
    data_clean = data_clean.head()
    st.dataframe(data_clean)
    st.write("Nous allons maintenant explorer la relation entre cette variable cible et les autres caractéristiques du dataset.")

    




if page == pages[2]:
    st.title(pages[2])
    st.write("1. Tracez des diagrammes en barres montrant la répartition des clients ayant pris un prêt personnel par rapport à différentes variables catégorielles telles que l'éducation, le compte de valeurs mobilières, le compte à terme, etc.")
    st.image("gr1.png")
    st.write("- Ces graphiques montrent la relation entre le niveau d'éducation, le type de compte et le fait d'avoir un prêt personnel.")

    st.markdown("---")
    st.write("2. Tracez des diagrammes en boîte pour comparer la distribution des variables numériques entre les groupes de clients ayant ou non un prêt personnel.")
    st.image("gr2.png")
    st.write("- Les graphiques suggèrent que les clients ayant des caractéristiques telles qu'un revenu élevé, une famille nombreuse et un endettement élevé sont plus susceptibles d'avoir un prêt personnel.")

    st.markdown("---")
    st.write("3. Tracez des histogrammes des variables numériques pour visualiser leur distribution et leur relation avec la cible 'Personal Loan'")
    st.image("gr3.png")
    st.write("- Les variables telles que le revenu, l'endettement et la taille de la famille pourraient jouer un rôle dans l'obtention d'un prêt personnel.")

    st.markdown("---")
    st.header("Sélection Du Modèle")
    st.write("- Étant donné que notre variable cible est binaire ('Personal Loan' : 0 ou 1), il s'agit là d'un problème de classification binaire.")

if page == pages[3]:
    st.title(pages[3])


    model = joblib.load("model_reg_line")
    model_poly = joblib.load("model_poly")


    st.write("Le model choisi est La Regression Linéaire")
    st.write("Après entrainement du modèle, nous avons obtenu un score final de 76%")


    st.subheader("Tester notre modèle avec vos données !!")

    # Age
    Age = st.number_input("L'âge de la personne (Ex.: 38)", step=1.0)

    # Le nombre d'années d'expérience professionnelle de la personne.
    Experience = st.number_input("Le nombre d'années d'expérience professionnelle de la personne : (Ex.: 14)", step=1.0)

    # Income: Le revenu de la personne.
    Income = st.number_input("Le revenu de la personne : (Ex.: 130)", step=1.0)

    # Family: Le nombre de membres de la famille de la personne.
    Family = st.number_input("Le nombre de membres de la famille de la personne : (Ex.: 4)", step=1.0)

    # CCAvg: La moyenne des dépenses sur les cartes de crédit par mois.
    CCAvg = st.number_input("La moyenne des dépenses sur les cartes de crédit par mois : (Ex.: 4.7)", step=1.0)

    # Education: Le niveau d'éducation de la personne
    Education = ['Enseignement Primaire', 'Education Secondaire', 'Education Tertiaire']
    Education_opt = st.selectbox(label = "Le niveau d'éducation de la personne : (Ex.: Education Tertiaire)", options = Education)
    if Education_opt in Education :
        Education_val = Education.index(Education_opt) +1

    # Mortgage: Le montant hypothécaire de la personne.
    Mortgage = st.number_input("Le montant hypothécaire de la personne : (Ex.: 134)", step=1.0)

    # Securities Account: Une variable binaire indiquant si la personne possède un compte de valeurs mobilières.
    Response1 = st.selectbox(label = "Est ce que la personne possède un compte de valeurs mobilières ? (Ex.: Non)", options = ['Non', 'Oui'])
    if Response1 == 'Non' :
        Securities_Account = 0
    elif Response1 == 'Oui' :
        Securities_Account = 1

    # CD Account: Une variable binaire indiquant si la personne possède un compte à terme (CD).
    Response2 = st.selectbox(label = "Est ce que la personne possède un compte à terme (CD) ? (Ex.: Non)", options = ['Non', 'Oui'])
    if Response2 == 'Non' :
        CD_Account = 0
    elif Response2 == 'Oui' :
        CD_Account = 1

    # Online: Une variable binaire indiquant si la personne utilise les services bancaires en ligne.
    Response3 = st.selectbox(label = "Est ce que la personne utilise les services bancaires en ligne ? (Ex.: Non)", options = ['Non', 'Oui'])
    if Response3 == 'Non' :
        Online = 0
    elif Response3 == 'Oui' :
        Online = 1

    # CreditCard: Une variable binaire indiquant si la personne possède une carte de crédit.
    Response4 = st.selectbox(label = "Est ce que la personne possède une carte de crédit ? (Ex.: Non)", options = ['Non', 'Oui'])
    if Response4 == 'Non' :
        CreditCard = 0
    elif Response4 == 'Oui' :
        CreditCard = 1


    X_user = np.array([Age, Experience, Income, Family, CCAvg, Education_val, Mortgage, Securities_Account, CD_Account, Online, CreditCard]).reshape(1,11)
    X_user_poly = model_poly.transform(X_user)

    if(st.button("TESTEZ !")):
        y_pred_user = model.predict(X_user_poly)
        if y_pred_user > 0.5 :
            st.write("Ce profil est plus susceptible de prendre un prêt personnel.")
            st.write("Score :", y_pred_user[0])
        else :
            st.write("Ce profil n'est pas susceptible de prendre un prêt personnel.")
            st.write("Score :", y_pred_user[0])
