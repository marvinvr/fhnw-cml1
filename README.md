<!-- Improved compatibility of back to top link: See: github-link -->
<a name="readme-top"></a>

<!-- logo einfügen mit Ordner namens "images" -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="bilder_plots/immobilienrechner.png" alt="Logo" width="300" height="300">
  </a>

  <h1 align="center">Immobilienrechner</h3>

  <p align="center">
    In diesem Repository wird die Immobilienrechner Challenge des Studienganges BSc Data Science der Fachhochschule Nordwestschweiz berarbeitet.
    <br />
    Ziel dieser Challenge ist es mithilfe von maschinellem lernen den Preis von schweizer Immobilien vorherzusagen.
  </p>
</div>


<!-- TABLE OF CONTENTS -->
  <summary><h2>Inhaltsverzeichnis<h3 /></summary>
  <ol>
    <li>
      <a href="#projektbeschrieb">Projektbeschrieb</a>
      <ul>
        <li><a href="#libraries">Libraries</a></li>
      </ul>
    </li>
    <li>
      <a href="#aufbau">Aufbau</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#modelle">Modelle</a></li>
    <li><a href="#webservice">Webservice</a></li>
    <li><a href="#kontakt">Kontakt</a></li>
  </ol>




<!-- PROJEKTBESCHRIEB -->
### Projektbeschrieb
Ziel dieser Challenge ist es mit gescrapten Daten verschiedene machine learning Modelle zu entwickeln und damit Preis sowie den Typ diverser schweizer Immobilien vorherzusagen.
<br />
Das Regressionsmodell wurde an einer Kaggle-Competition eingereicht.
<br />
Zudem wurde ein Webservice erstellt, mit welchem den Preis einer beliebigen schweizer Immobilie in CHF schätzen lassen kann.
<br />
Die Modelle wurden allesamt in Python mit der Library scikit-learn erstellt.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Libraries

Das Repository wurde mit Python erstellt.
Folgende Libraries wurden dafür verwendet:

* Pandas
* scikit-learn
* Numpy
* Streamlit
* Joblib
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- AUFBAU -->
### Aufbau
Die Jupyter-Notebooks können in chronologischer Reihenfolge ausgeführt werden.

:file_folder: "data" -> Originale Datensätze, Datensätze für die Kaggle-competition, Web Service  <br />
:file_folder: "helpers" -> Diverse Hilfsfunktionen,bei welchen Parameter Global verändert werden können (Pfäde/ Parameter der Modelle).<br />
:file_folder: "linear_regression" -> Einfaches lineares Regressionsmodell zwischen den Variablen Wohnfläche und Preis. <br />
:file_folder: "models" -> Trainiere Modelle als Pickle-Dateien hinterlegt. <br />
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Prerequisites

* joblib==1.2.0
* numpy==1.23.5
* pandas==1.5.2
* scikit_learn==1.2.0
* streamlit==1.16.0

Die genauen Versionen können mit Pip aktualisiert werden.

   ```sh
   pip install "LIBRARY"=="VERSION"
   ```

Falls du noch kein Pip installiert hast findest du [hier](https://hellocoding.de/blog/coding-language/python/pip) eine Anleitung. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MODELLE -->
### Modelle
Um eine exakte Vorhersage zu treffen wurden für jeweils das Regressions- sowie das Klassifikationsproblem mehrere Modelle trainiert und verglichen.


Folgende Modelle wurden verwendet:

<b>Regression:</b>
* Ridge Regression
* Random Forest Regressor
* XGboost Regressor
* MLP Regressor

<b>Klassifikation:</b>
* KNN Klassifikator
* Random Forest Klassifikator
* XGboost Klassifikator
* MLP Klassifikator
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- WEBSERVICE -->
### Webservice
Um den Webservice zu starten muss folgender Befehl in einem Linux-Terminal ausgeführt werden.

   ```sh
   streamlit run 05_0_web_service.py
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- KONTAKT -->
### Kontakt

:bust_in_silhouette: Yannic Lais - [@yalais](https://github.com/yalais) <br />
:bust_in_silhouette: Marvin von Rappard - [@marvinvr](https://github.com/marvinvr) <br />
:bust_in_silhouette: Luca Mazzotta - [@focusedluca](https://github.com/focusedluca)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
