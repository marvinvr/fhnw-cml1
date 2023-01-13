<!-- Improved compatibility of back to top link: See: github-link -->
<a name="readme-top"></a>

<!-- logo einfügen mit Ordner namens "images" -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Immobilienrechner</h3>

  <p align="center">
    In diesem Repository wirddie Immobilienrechner Challange des Studienganges BSc Datascience der Fachhochschule Nordwestschweiz berarbeitet.
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary><b>Inhaltsverzeichnis</b></summary>
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
</details>



<!-- PROJEKTBESCHRIEB -->
### Projektbeschrieb
Ziel dieser Challanges ist es mit gescrapten Daten verschiedene machine learning Modelle zu entwickeln und damit Preis sowie den Immobilientyp verschiedenster schweizer Immobilien vorherzusagen.
<br />
Das Regressionsmodell wurde an einer Kaggle-Competition eingereicht.
<br />
Zudem wurde ein Webservice erstellt, mit welchem man den Preis einer beliebigen Immobilie schätzen lassen kann.
<br />
Die Modelle wurden allesamt in Python mit der Librarie scikit-learn erstellt.


#### Libraries

Das Repository wurde mit Python erstellt.
Folgende Libraries wurden dafür verwendet:

* Pandas
* scikit-learn
* Numpy
* Streamlit
* Joblib

<!-- AUFBAU -->
### Aufbau
Die Jupyter-Notebooks können in Chronologische Reihenfolge ausgeführt werden.

:file_folder: "data" -> Originale Datensätze, Datensätze für die Kaggle-competition, Web Service  <br />
:file_folder: "helpers" -> Diverse Hilfsfunktionen,bei welchen Parameter Global verändert werden können (Pfäde/ Parameter der Modelle).<br />
:file_folder: "linear_regression" -> Einfaches lineares Regressionsmodell zwischen den Variablen Wohnfläche und Preis. <br />
:file_folder: "models" -> Trainiere Modelle als Pickle-Dateien hinterlegt. <br />

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

Yannic Lais - [@yalais](https://github.com/yalais) <br />
Marvin von Rappard - [@marvinvr](https://github.com/marvinvr) <br />
Luca Mazzotta - [@focusedluca](https://github.com/focusedluca)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
