# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [**Angriffserkennung in Honeypot-Logs durch einen Autoencoder mit LSTM**](#toc1_)    
#   - [Einleitung](#toc1_1_)    
#   - [Datensatzbeschreibung](#toc1_2_)    
#     - [Quellen](#toc1_2_1_)    
#   - [Lizenz und Copyright](#toc1_3_)    
#   - [Vorbereitung](#toc1_4_)    
#     - [Importe](#toc1_4_1_)    
#     - [TensorFlow-Version und GPU-Verfügbarkeit](#toc1_4_2_)    
#   - [Datenladen & Basis-Cleaning](#toc1_5_)    
#   - [Visualisierung des Datensatzes](#toc1_6_)    
#     - [Angriffe vs normale Logs](#toc1_6_1_)    
#     - [Fokus auf `http` Service](#toc1_6_2_)    
#       - [Auswahl der HTTP-Daten](#toc1_6_2_1_)    
#     - [Verteilung der Angriffstypen](#toc1_6_3_)    
#   - [Datenauswahl für das Training](#toc1_7_)    
#     - [Korrelation-Heatmap von HTTP-Features: Normalisierte Daten](#toc1_7_1_)    
#   - [Auswahl der PCA-Coverage und Dimensionsreduktion](#toc1_8_)    
#   - [Fensterbildung und zufällige Permutation von Daten](#toc1_9_)    
#       - [Wichtige Parameter und ihre Bedeutung](#toc1_9_1_1_)    
#       - [Alternative Ansätze zur Fensterbildung](#toc1_9_1_2_)    
#       - [Sliding-Window-Ansatz](#toc1_9_1_3_)    
#   - [Implementierung eines LSTM-Autoencoders mit Bayesian Optimization](#toc1_10_)    
#       - [Notationen](#toc1_10_1_1_)    
#       - [Rechenmodelle](#toc1_10_1_2_)    
#       - [Optimierung mit Bayesian Optimization](#toc1_10_1_3_)    
#       - [Ablauf](#toc1_10_1_4_)    
#   - [Trainiere das finale Modell mit den besten Hyperparametern](#toc1_11_)    
#     - [Testen des Autoencoders](#toc1_11_1_)    
#   - [Verarbeitung und Vorbereitung der Testdaten](#toc1_12_)    
#       - [Schritte zur Datenverarbeitung](#toc1_12_1_1_)    
#       - [Anmerkung](#toc1_12_1_2_)    
#   - [Bewertung des LSTM-Autoencoders und Anomalieerkennung](#toc1_13_)    
#       - [Schritte zur Modellbewertung](#toc1_13_1_1_)    
#   - [Verschiedene Metriken](#toc1_14_)    
#   - [Kategoriale Kenngrößen](#toc1_15_)    
#   - [Metriken für Klassifikationsmodelle](#toc1_16_)    
#     - [Accuracy (Standard)](#toc1_16_1_)    
#       - [Precision](#toc1_16_1_1_)    
#       - [Recall](#toc1_16_1_2_)    
#       - [F1-Score](#toc1_16_1_3_)    
#       - [Tabelle: Übersicht der Begriffe](#toc1_16_1_4_)    
#     - [ROC und AUC-Bewertung](#toc1_16_2_)    
#       - [Visualisierung](#toc1_16_2_1_)    
#       - [Wichtige Aspekte](#toc1_16_2_2_)    
#     - [Precision-Recall-Kurve und Average Precision (AP)](#toc1_16_3_)    
#     - [F1-Score-Kurve basierend auf Schwellenwerten](#toc1_16_4_)    
#       - [Schritte zur Berechnung der F1-Score-Kurve](#toc1_16_4_1_)    
#     - [Schwellenwertoptimierung und Bewertung der Anomalieerkennung](#toc1_16_5_)    
#       - [Berechnung der Bewertungsmetriken](#toc1_16_5_1_)    
#   - [Anomalieerkennung im Produktionsdatensatz](#toc1_17_)    
#       - [Schritte zur Datenverarbeitung](#toc1_17_1_1_)    
#       - [Anmerkung](#toc1_17_1_2_)    
#     - [Ausgabe der gefundenen Anormalien](#toc1_17_2_)    
#   - [Ausblick](#toc1_18_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# # <a id='toc1_'></a>[**Angriffserkennung in Honeypot-Logs durch einen Autoencoder mit LSTM**](#toc0_)
# 

# %% [markdown]
# ## <a id='toc1_1_'></a>[Einleitung](#toc0_)
# 
# Im Rahmen des Moduls „Künstliche Intelligenz und Maschinelles Lernen“ wird in diesem Projekt ein Ansatz zur Erkennung von Hackerangriffen mittels eines Autoencoders mit Long Short-Term Memory (LSTM) entwickelt und evaluiert. Ziel ist es, eine Grundlage für die Anwendung maschineller Lernmethoden zur Anomalieerkennung in Netzwerkdaten zu schaffen. Dabei liegt der Fokus auf der Analyse des HTTP-Protokolls als Einstieg in die Entwicklung eines umfassenderen Honeypots.
# 
# Ein Autoencoder ist ein neuronales Netzwerk, das darauf trainiert wird, Eingabedaten auf eine kompakte Repräsentation zu komprimieren und diese anschließend zu rekonstruieren. Anomalien werden erkannt, wenn die rekonstruierten Daten signifikant von den Originaldaten abweichen, da solche Abweichungen auf Muster hinweisen können, die das Modell nicht gelernt hat. Die Integration von LSTM-Schichten erlaubt es, zeitliche Abhängigkeiten in sequentiellen Daten, wie sie bei Netzwerkverkehr auftreten, effektiv zu modellieren.
# 
# Der in diesem Projekt verwendete KDD99-Datensatz ist ein Benchmark-Standard für Intrusion Detection Systeme und bietet eine Vielzahl von Netzwerkverbindungsdaten, die in normale Verbindungen und verschiedene Angriffstypen unterteilt sind. Im Rahmen dieses Projekts wird der Datensatz speziell für das HTTP-Protokoll untersucht. Das Modell wird trainiert, um Muster in normalen Verbindungen zu lernen und Abweichungen, die auf potenzielle Angriffe hinweisen, zu identifizieren.
# 
# Dieses Projekt bildet zugleich den Einstieg in die Vorbereitung einer Bachelorarbeit, die in Zusammenarbeit mit dem Bundesamt für Sicherheit in der Informationstechnik (BSI) durchgeführt wird. Während im aktuellen Projekt nur das HTTP-Protokoll analysiert wird, soll in der Bachelorarbeit ein erweiterter Ansatz entwickelt werden, der sich auf reale Honeypot-Daten des BSI stützt. Honeypots sind speziell gestaltete Systeme, die Cyberangreifer anziehen, um deren Angriffsmethoden zu analysieren und Daten über potenzielle Bedrohungen zu sammeln. Die Bachelorarbeit wird sich auf die Analyse mehrerer Protokolle fokussieren, um die Erkennung von Angriffen zu generalisieren und die Grundlagen für ein umfassenderes, protokollübergreifendes Intrusion Detection System zu schaffen.
# 
# Dieses Projekt im Modul „Künstliche Intelligenz und Maschinelles Lernen“ dient somit als wertvolle Basis, um theoretische Konzepte aus der Vorlesung mit praktischen Anwendungsfällen zu verbinden.
# 
# 
# ---

# %% [markdown]
# ## <a id='toc1_2_'></a>[Datensatzbeschreibung](#toc0_)
# 
# Für die Modellierung der Intrusion-Detection-Aufgabe wird der **KDD99-Datensatz** verwendet, der weithin als Benchmark für IDS gilt. Der KDD99-Datensatz wurde aus dem DARPA1998-Projekt extrahiert und enthält 41 Merkmale, die aus rohen TCP-Paketen abgeleitet wurden. Diese Merkmale sind in vier Kategorien unterteilt:
# 
# 1. **Basic Features**: Grundlegende Merkmale wie die Anzahl der Bytes, Dauer der Verbindung etc.
# 2. **Content Features**: Merkmale, die den Inhalt der Verbindung betreffen, wie die Anzahl der Fehlversuche bei Anmeldungen.
# 3. **Host-based Statistical Features**: Statistische Merkmale, die sich auf den Host beziehen, wie die Anzahl der Prozesse pro Verbindung.
# 4. **Time-based Statistical Features**: Zeitbezogene statistische Merkmale, die Muster in den Verbindungen über die Zeit hinweg erfassen.
# 
# ### <a id='toc1_2_1_'></a>[Quellen](#toc0_)
# 
# - **KDD99 Dataset**: [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/task.html)
# - **Download-Links**:
#     - Trainingsdaten: [kdd_train.csv](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz)
#     - Testdaten: [kdd_test.csv](http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz)
#     - Produktionsdaten: [kdd_prod.csv](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.newtestdata_10_percent_unlabeled.gz)
#     - Spaltennamen: [kddcup.names](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names)
# - <https://lilianweng.github.io/posts/2018-08-12-vae/>\
# - <https://victordibia.com/blog/introducing-anomagram/>
# - https://www.jeremyjordan.me/autoencoders/
# - Song, Y., Hyun, S., & Cheong, Y. G. (2021). Analysis of Autoencoders for Network Intrusion Detection. Sensors (Basel, Switzerland), 21(13), 4294. https://doi.org/10.3390/s21134294
# - Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., Al-Nemrat, A., & Venkatraman, S. (2019). Deep Learning Approach for Intelligent Intrusion Detection System. IEEE Access, 7, 41525-41550. https://ieeexplore.ieee.org/abstract/document/8681044
# - Jason Brownlee https://machinelearningmastery.com/lstm-autoencoders/ und weitere
# - Teile dieses Codes wurden mit ChatGPT-o1 generiert.
# 
# 
# 
# Der Datensatz umfasst sowohl gelabelte als auch ungelabelte Daten, wobei das Ziel darin besteht, ein Modell zu trainieren, das auf Live-Produktionsdaten angewendet werden kann, wo Labels nicht verfügbar sind.
# 
# ---

# %% [markdown]
# ## <a id='toc1_3_'></a>[Lizenz und Copyright](#toc0_)

# %% [markdown]
# MIT License
# 
# Copyright (c) 2024 Lukas Johannes Möller (github@pontis-it.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# %%
dummifiziere_categoricals = False # anstatt sie zu droppen

# %% [markdown]
# ## <a id='toc1_4_'></a>[Vorbereitung](#toc0_)

# %% [markdown]
# ### <a id='toc1_4_1_'></a>[Importe](#toc0_)

# %%
# Basis-Pakete
import os
import csv
import json
import random
import numpy as np
import pandas as pd
import tqdm
import statistics
import math
import scipy.stats as st


# Visualisierung
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from IPython.display import Image
import pydot
import graphviz

# TensorFlow und Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy

# Maschinelles Lernen
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix
)

# Bayesian Optimization
from bayes_opt import BayesianOptimization


# %% [markdown]
# ### <a id='toc1_4_2_'></a>[TensorFlow-Version und GPU-Verfügbarkeit](#toc0_)

# %%
print('TensorFlow-Version:', tf.__version__)
print('Verfügbare GPU-Geräte:', tf.config.list_physical_devices('GPU'))
# `tf.test.is_gpu_available()` ist in neueren TF-Versionen veraltet, aber wir belassen es exemplarisch:
print('Ist eine GPU verfügbar:', tf.test.is_gpu_available())

# %% [markdown]
# ## <a id='toc1_5_'></a>[Datenladen & Basis-Cleaning](#toc0_)

# %% [markdown]
# In diesem Abschnitt werden der Datensatz geladen und grundlegende Bereinigungen durchgeführt. Dazu zählt das Prüfen auf negative Werte in numerischen Spalten.
# 

# %%
# Pfade zum Trainings- und Testdaten
TRAIN_DATEN_PFAD = 'kdd_train.csv'
SPALTENNAMEN_PFAD = 'columns_names.txt'
TEST_DATEN_PFAD = 'kdd_test.csv'
PROD_DATEN_PFAD = 'kdd_prod.csv'

# Einlesen der Spaltennamen
with open(SPALTENNAMEN_PFAD, 'r') as txt_file:
    spaltennamen_liste = txt_file.readlines()

bereinigte_spaltennamen = [
    col.split(':')[0] for col in spaltennamen_liste
    if len(col.split(':')) > 1
]

# Trainingsdaten einlesen
df_train = pd.read_csv(TRAIN_DATEN_PFAD, header=None)
print(f"Ursprungsform der Trainingsdaten: {df_train.shape}")

# Quick-Sanity-Check für die Spaltenanzahl
if len(bereinigte_spaltennamen) > len(df_train.columns):
    bereinigte_spaltennamen = bereinigte_spaltennamen[:len(df_train.columns)]
elif len(bereinigte_spaltennamen) < len(df_train.columns):
    bereinigte_spaltennamen.extend([f"Unnamed_{i}" for i in range(len(bereinigte_spaltennamen), len(df_train.columns))])

df_train.columns = bereinigte_spaltennamen

# Prüfung auf negative Werte in numerischen Spalten
numerische_spalten = df_train.select_dtypes(include=[np.number]).columns
for col in numerische_spalten:
    anzahl_negative = df_train[df_train[col] < 0].shape[0]
    if anzahl_negative > 0:
        print(f"Warnung: {anzahl_negative} negative Werte in {col}.")

df_train.head()

# %% [markdown]
# ## <a id='toc1_6_'></a>[Visualisierung des Datensatzes](#toc0_)

# %% [markdown]
# ### <a id='toc1_6_1_'></a>[Angriffe vs normale Logs](#toc0_)

# %%
pd.value_counts(df_train['result']).plot(kind='bar', figsize=(12, 4))
plt.ylabel('Anzahl Instanzen')
plt.xlabel('Angriffs-/Normal-Typ')
plt.title('Verteilung der Angriffsarten')
plt.show()

# %% [markdown]
# ### <a id='toc1_6_2_'></a>[Fokus auf `http` Service](#toc0_)

# %% [markdown]
# Aus der Analyse der verschieden Dienste stellt sich der HTTP Dienst als gut geeignet heraus diesen genauer zu untersuchen. Das zukünftige Model für das BSI sollte Service übergreifend funktionieren, in diesem Ansatz wird darauf zur Vereinfachung verzichtet.

# %%
# 1) Aggregation pro Service
services = df_train['service'].unique()
service_aggregation = []

for svc in services:
    df_svc = df_train[df_train['service'] == svc]
    total_count   = len(df_svc)
    normal_count  = (df_svc['result'] == 'normal.').sum()
    anomaly_count = total_count - normal_count
    
    # Anteil (%) an Anomalien (0.0, falls total_count = 0)
    pct_anomaly = (anomaly_count / total_count * 100) if total_count > 0 else 0.0
    
    service_aggregation.append({
        'service': svc,
        'total': total_count,
        'normal': normal_count,
        'anomaly': anomaly_count,
        'pct_anomaly': pct_anomaly  # in Prozent
    })

df_services = pd.DataFrame(service_aggregation)

# 2) Nach absteigender total-Anzahl sortieren
df_services.sort_values(by='total', ascending=False, inplace=True)

# 3) Kontrolle
print(df_services)

# ### Visualisierung ###
# 4) Balkendiagramm (Normal vs. Anomaly), sortiert nach total
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
x_positions = range(len(df_services))
ax.bar(x_positions, df_services['normal'], width=bar_width, label='Normal', color='green')
ax.bar(
    [pos + bar_width for pos in x_positions],
    df_services['anomaly'],
    width=bar_width,
    label='Anomaly',
    color='red'
)
ax.set_xticks([pos + bar_width / 2 for pos in x_positions])
ax.set_xticklabels(df_services['service'], rotation=90)
ax.set_ylabel('Anzahl')
ax.set_title('Anzahl Normal vs. Anomaly pro Service (sortiert nach total)')
ax.legend()
plt.tight_layout()
plt.show()

# 5) Anteil Anomalien in %
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(df_services['service'], df_services['pct_anomaly'], color='blue')
ax.set_ylabel('Anomalien in % (von Gesamt)')
ax.set_title('Anteil Anomalien pro Service')
ax.set_xticklabels(df_services['service'], rotation=90)
plt.tight_layout()
plt.show()


# %% [markdown]
# #### <a id='toc1_6_2_1_'></a>[Auswahl der HTTP-Daten](#toc0_)

# %%
df_train_http = df_train[df_train['service'] == 'http']
df_train_http.shape

# %% [markdown]
# ### <a id='toc1_6_3_'></a>[Verteilung der Angriffstypen](#toc0_)

# %%
pd.value_counts(df_train_http['result']).plot(kind='bar', figsize=(12, 4))
plt.ylabel('Anzahl')
plt.xlabel('Angriffs-/Normal-Typ')
plt.title('Verteilung der Angriffsarten im http-Subset')
plt.show()

anzahl_normal = df_train_http[df_train_http['result'] == 'normal.'].shape[0]
anzahl_anomal = df_train_http[df_train_http['result'] != 'normal.'].shape[0]
print(f'Prozentualer Anteil an Anomalien (im HTTP-Subset) = {100*anzahl_anomal/anzahl_normal:.2f}%')

# %% [markdown]
# ## <a id='toc1_7_'></a>[Datenauswahl für das Training](#toc0_)
# Wir wählen hier nur die normalen HTTP-Daten aus, um einen One-Class-Ansatz zu fahren.

# %%
# Nur normale HTTP-Daten
df_train_http_normal = df_train_http[df_train_http['result'] == 'normal.'].copy()
print("Form der ursprünglichen Daten:", df_train_http_normal.shape)

# One-Hot-Encoding für kategoriale Spalten (abhängig von `dummifiziere_categoricals` parameter am Anfang)
kategoriale_spalten = df_train_http_normal.select_dtypes(include=['object', 'category']).columns
if dummifiziere_categoricals:
    print(f"Kategoriale Spalten für One-Hot-Encoding: {list(kategoriale_spalten)}")
    df_train_http_normal = pd.get_dummies(df_train_http_normal, columns=kategoriale_spalten, drop_first=False, dtype=int)
    print(f"Form der Daten nach One-Hot-Encoding: {df_train_http_normal.shape}")
else:
    df_train_http_normal = df_train_http_normal.drop(columns=kategoriale_spalten)

# Spalten ohne Standardabweichung entfernen
spalten_std = df_train_http_normal.std()
spalten_ohne_std = spalten_std[spalten_std == 0].index
df_train_http_normal = df_train_http_normal.drop(columns=spalten_ohne_std)
# aufgrund hoher Korrelationen wird die Spalte 'logged_in' entfernt (leider nur manuell)
df_train_http_normal =df_train_http_normal.drop(columns=['logged_in'])
# Beschreibung der finalen Daten
df_train_http_normal.describe()


# %%
df_train_http_normal.boxplot(figsize=(20, 10))
plt.show()

# %% [markdown]
# ### <a id='toc1_7_1_'></a>[Korrelation-Heatmap von HTTP-Features: Normalisierte Daten](#toc0_)
# Hier betrachten wir nur die normalen HTTP-Daten (nach Normalisierung) und visualisieren eine Korrelation-Heatmap.

# %%
skalierer = StandardScaler()
df_train_http_normal = pd.DataFrame(
    skalierer.fit_transform(df_train_http_normal),
    columns=df_train_http_normal.columns
)

plt.figure(figsize=(15, 10))
sns.heatmap(df_train_http_normal.corr(), cmap='viridis')
plt.title('Korrelation-Heatmap der nur normalen (HTTP) Merkmale')
plt.show()
df_train_http_normal.boxplot(figsize=(20, 10))
plt.show()

# %% [markdown]
# ## <a id='toc1_8_'></a>[Auswahl der PCA-Coverage und Dimensionsreduktion](#toc0_)
# Principal Component Analysis (Hauptkomponentenanalyse) ermöglicht es hochdimensionale Datensätze auf weniger Dimensionen zu reduzieren.
# 
# Siehe: [scikit-Learn Dokumentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html)
# 
# MSE Mean Square Error

# %%
def crossval_pca_coverage(data: pd.DataFrame, coverages=[0.8,0.9,0.95], k=3):
    best_cov = coverages[0]
    best_mse = float('inf')
    data_np = data.to_numpy()
    kf = KFold(n_splits=k, shuffle=False)

    for cov in coverages:
        fold_mses = []
        for train_idx, val_idx in kf.split(data_np):
            X_tr = data_np[train_idx]
            X_val= data_np[val_idx]
            pca_temp = PCA(n_components=cov)
            pca_temp.fit(X_tr)
            X_val_pca = pca_temp.transform(X_val)
            X_val_rec = pca_temp.inverse_transform(X_val_pca)
            mse = np.mean((X_val - X_val_rec)**2)
            fold_mses.append(mse)
        avg_mse = np.mean(fold_mses)
        print(f"Coverage={cov} => avg MSE={avg_mse:.4f}")
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_cov = cov

    return best_cov

coverages = [0.8, 0.9, 0.95, 1]
best_cov = crossval_pca_coverage(df_train_http_normal, coverages=coverages, k=3)
print("Beste Coverage:", best_cov)

# Endgültige PCA antrainieren
pca_modell = PCA(n_components=best_cov)
pca_modell.fit(df_train_http_normal)
pca_dim = pca_modell.n_components_
df_train_pca = pd.DataFrame(
    pca_modell.transform(df_train_http_normal),
    columns=[f'PCA_{i}' for i in range(pca_dim)]
)
print(f"PCA-Komponenten: {pca_dim}")
df_train_pca.head()

# %% [markdown]
# ## <a id='toc1_9_'></a>[Fensterbildung und zufällige Permutation von Daten](#toc0_)
# Hier werden die Daten in überlappende Fenster aufgeteilt und anschließend zufällig permutiert.
# 

# %% [markdown]
# #### <a id='toc1_9_1_1_'></a>[Wichtige Parameter und ihre Bedeutung](#toc0_)
# - **Fenstergröße** (`window_size`)
# - **Schrittweite** (`stride`)
# 

# %% [markdown]
# #### <a id='toc1_9_1_2_'></a>[Alternative Ansätze zur Fensterbildung](#toc0_)
# - Fixed Window, Expanding Window, Tumbling Window etc.
# 

# %% [markdown]
# #### <a id='toc1_9_1_3_'></a>[Sliding-Window-Ansatz](#toc0_)
# 
# In diesem Abschnitt wird eine Funktion zur Fensterbildung implementiert.  
# Die **Fenstergröße** (`fenster_groesse`) legt fest, wie viele aufeinanderfolgende Datenpunkte in einem Fenster enthalten sind. Sie ermöglicht es, zeitliche Muster über einen bestimmten Zeitraum zu erfassen. Eine zu kleine Fenstergröße könnte wichtige Muster verlieren, während eine zu große Fenstergröße Rechenressourcen verschwendet. 
# 
# Die **Schrittweite** (`schritt`) gibt an, wie weit das Fenster in den Daten voranschreitet. Eine kleinere Schrittweite erzeugt mehr überlappende Fenster, was die Modellrobustheit verbessern kann, jedoch die Berechnungszeit erhöht ([Volpe et al., 2024](https://doi.org/10.3390/s24247924)).
# 
# In diesem Beispiel verwenden wir:  
# - `fenster_groesse=10`: Zur Erfassung kurzfristiger Muster.  
# - `schritt=10`: Ohne Überlappung gab es die besten Ergbnisse
# 
# Der Sliding-Window-Ansatz ist hier sinnvoll, weil zeitliche Abhängigkeiten erfasst werden.
# 

# %%
def fenster_bilden(df_eingabe, fenster_groesse, schritt):
    fenster_liste = []
    for i in tqdm.tqdm(range(0, len(df_eingabe) - fenster_groesse + 1, schritt)):
        fenster_liste.append(df_eingabe.iloc[i:i + fenster_groesse, :].to_numpy())
    return np.array(fenster_liste)
fenster_groesse=10
schrittweite=10

train_fenster = fenster_bilden(df_train_pca, fenster_groesse=fenster_groesse, schritt=schrittweite)
print('Form der Fenster:', train_fenster.shape)

# Shuffle
zufalls_indices = np.arange(train_fenster.shape[0])
np.random.shuffle(zufalls_indices)
train_fenster_shuffle = train_fenster[zufalls_indices]
train_fenster_shuffle.shape

# %% [markdown]
# ## <a id='toc1_10_'></a>[Implementierung eines LSTM-Autoencoders mit Bayesian Optimization](#toc0_)
# Im Folgenden wird ein LSTM-Autoencoder definiert. Mithilfe einer Bayesian Optimization werden passende Hyperparameter gesucht.

# %% [markdown]
# #### <a id='toc1_10_1_1_'></a>[Notationen](#toc0_)
# | Symbol | Beschreibung |
# | - | - |
# | $x$ | Eingabesequenz |
# | $z$ | Latente Repräsentation (Encoder-Ausgabe) |
# | $x'$ | Rekonstruktion (Decoder-Ausgabe) |
# | $g_{\phi}(x)$ | Encoder-Funktion (Parameter $\phi$) |
# | $f_{\theta}(z)$ | Decoder-Funktion (Parameter $\theta$) |
# | $\mathcal{L}$ | Verlustfunktion (Rekonstruktionsfehler) |
# 

# %% [markdown]
# #### <a id='toc1_10_1_2_'></a>[Rechenmodelle](#toc0_)
# 1. Encoding: $z = g_{\phi}(x)$
# 2. Decoding: $x' = f_{\theta}(z)$
# 3. Verlust: z. B. Huber Loss, MSE oder MAE.
# 

# %% [markdown]
# #### <a id='toc1_10_1_3_'></a>[Optimierung mit Bayesian Optimization](#toc0_)
# - Hyperparameter-Bereich (Anzahl Neuronen in LSTM-Schichten)
# - Ziel: Minimierung des Validierungsverlusts.
# 
# ![GpParBayesAnimationSmall.gif](GpParBayesAnimationSmall.gif)
# 
# By AnotherSamWilson - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=84842869

# %% [markdown]
# #### <a id='toc1_10_1_4_'></a>[Ablauf](#toc0_)
# 1. Baue LSTM-Autoencoder.
# 2. Trainiere mit gegebenen Parametern.
# 3. Bestimme Validierungsverlust.
# 4. Bayesian Optimization iteriert über Parameter.
# 
# sigmoid activation (Logistische Funktion)
# $
# \sigma(x) = \frac{1}{1 + e^{-x}}
# $ Output Werte zwischen 0 und 1 
# 
# tanh (Tangens-Hyperbolicus-Funktion) activation
# $
# \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
# $ Output Werte zwischen -1 und 1
# 
# ``recurrent_activation`` bezieht sich auf die internen Zustände und die Speichermechansimen des Neurons
# 
# Die Wahl der Iterationen bei der Bayesian Search war nicht trivial. Ein Paper konnte allerdings eine grobe Richtung aufzeigen. [(Yperman et. al. 2016)](https://www.semanticscholar.org/paper/Bayesian-optimization-of-hyper-parameters-in-Yperman-Becker/1d198d646129133a82b876529f066f83878e2f59?utm_source=direct_link) Bei 6 Hyperparametern nahm der Normalisierter Mittlerer Quadratischer Fehler nach 50 Iterationen kaum noch ab.
# 
# Warum sigmoid schlechter als linear?
# 
# Für das Model wurde der functional API Ansatz von Keras gewählt. https://keras.io/guides/functional_api/

# %%

# Funktion zur Aktualisierung und Speicherung der besten Ergebnisse
def aktualisiere_beste_ergebnisse(aktuelle_ergebnisse):
    global beste_ergebnisse_erst
    if (
        beste_ergebnisse_erst is None or
        aktuelle_ergebnisse['score'] > beste_ergebnisse_erst['score']
    ):
        beste_ergebnisse_erst = aktuelle_ergebnisse
        with open(beste_ergebnisse_datei, "w") as f:
            json.dump(beste_ergebnisse_erst, f, indent=2)
        print("Beste Ergebnisse aktualisiert:", beste_ergebnisse_erst)

# Datei zur Speicherung der besten Ergebnisse
beste_ergebnisse_datei = "beste_ergebnisse.json"

# Beste Ergebnisse laden, falls die Datei existiert
if os.path.exists(beste_ergebnisse_datei):
    with open(beste_ergebnisse_datei, "r") as f:
        beste_ergebnisse_erst = json.load(f)
else:
    beste_ergebnisse_erst = None

def baue_autoencoder(
    anzahl_neuronen_1, 
    anzahl_neuronen_2, 
    anzahl_neuronen_3, 
    fenster_groesse, 
    anzahl_merkmale,
    learning_rate,
    verlustfunktion='huber'
):
    """
    Baut ein Autoencoder-Modell basierend auf dem in der Frage
    gezeigten Encoder-Decoder-Aufbau mit gestapelten LSTM-Schichten.
    """
    
    # Input-Layer
    input_layer = Input(shape=(fenster_groesse, anzahl_merkmale), name="input_layer")
    
    # ------------------- Encoder ------------------- 
    # 1. LSTM-Schicht (mit return_sequences=True)
    x = LSTM(
        units=int(anzahl_neuronen_1),
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.0,
        recurrent_dropout=0.0
    )(input_layer)
    
    # 2. LSTM-Schicht (mit return_sequences=True)
    x = LSTM(
        units=int(anzahl_neuronen_2),
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.0,
        recurrent_dropout=0.0
    )(x)
    
    # 3. LSTM-Schicht (latent space, ohne return_sequences)
    latent_space = LSTM(
        units=int(anzahl_neuronen_3),
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.0,
        recurrent_dropout=0.0,
        name='latent_space'
    )(x)
    
    # ------------------- Decoder -------------------
    # Wiederholen des latent-space Vektors für jedes Zeitschritt-Fenster
    x = RepeatVector(fenster_groesse)(latent_space)
    
    # 1. Decoder-LSTM-Schicht
    x = LSTM(
        units=int(anzahl_neuronen_2),
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.0,
        recurrent_dropout=0.0
    )(x)
    
    # 2. Decoder-LSTM-Schicht
    x = LSTM(
        units=int(anzahl_neuronen_1),
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.0,
        recurrent_dropout=0.0
    )(x)
    
    # Ausgabe-Schicht mit TimeDistributed(Dense(...))
    output_layer = TimeDistributed(
        Dense(int(anzahl_merkmale), activation='linear')
    )(x)
    
    # Komplettes Autoencoder-Modell
    autoencoder_ges = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")

    # Optimierer mit dynamischer Lernrate und Gradienten-Clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

    # Modell kompilieren
    autoencoder_ges.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
    return autoencoder_ges


# Zielfunktion für die Optimierung
def trainiere_und_bewerte(
    anzahl_neuronen_1, 
    anzahl_neuronen_2, 
    anzahl_neuronen_3,
    learning_rate,
    batch_size
):
    # Konvertiere Batch-Größe zu Integer
    batch_size = int(batch_size)
    
    # Leere die vorherige Keras-Sitzung
    K.clear_session()
    
    # Bau des Autoencoders mit den aktuellen Hyperparametern
    modell_aktuell = baue_autoencoder(
        anzahl_neuronen_1=int(anzahl_neuronen_1),
        anzahl_neuronen_2=int(anzahl_neuronen_2),
        anzahl_neuronen_3=int(anzahl_neuronen_3),
        fenster_groesse=fenster_groesse,
        anzahl_merkmale=pca_modell.n_components_,
        learning_rate=learning_rate,
        verlustfunktion='huber'
    )
    
    # Define ModelCheckpoint Callback
    temp_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'temp_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0
    )
    
    # Training des Modells
    verlauf = modell_aktuell.fit(
        train_fenster_shuffle,                # Eingabedaten (x)
        train_fenster_shuffle,
        batch_size=batch_size,                # Dynamische Batch-Größe
        validation_split=0.2,
        epochs=10,
        callbacks=[temp_checkpoint],
        verbose=0
    )
    
    # Besten Validierungs-Loss ermitteln
    bester_valverlust = min(verlauf.history['val_loss'])
    
    # Optional: Modell nach dem Training löschen, um Speicher freizugeben
    del modell_aktuell
    tf.keras.backend.clear_session()
    
    # Negativen Validierungs-Loss zurückgeben (da Bayesian Optimization maximieren möchte)
    return -bester_valverlust

# Funktion zur Durchführung der Optimierung mit Zwischenspeicherung
def optimierung_mit_speicherung(bo_optimizer, init_points, n_iter):
    try:
        bo_optimizer.maximize(init_points=init_points, n_iter=n_iter)
    except KeyboardInterrupt:
        print("Optimierung unterbrochen.")
    finally:
        # Verarbeite alle bisherigen Ergebnisse
        for res in bo_optimizer.res:
            # Rundet die Hyperparameter zu den erforderlichen Typen
            aktuelle_hyperparameter = res['params']
            aktuelle_hyperparameter['batch_size'] = int(round(aktuelle_hyperparameter['batch_size']))
            aktuelle_hyperparameter['anzahl_neuronen_1'] = int(round(aktuelle_hyperparameter['anzahl_neuronen_1']))
            aktuelle_hyperparameter['anzahl_neuronen_2'] = int(round(aktuelle_hyperparameter['anzahl_neuronen_2']))
            aktuelle_hyperparameter['anzahl_neuronen_3'] = int(round(aktuelle_hyperparameter['anzahl_neuronen_3']))
            
            aktuelle_ergebnisse = {
                'hyperparameter': aktuelle_hyperparameter,
                'score': res['target']
            }
            
            # Aktualisiere die besten Ergebnisse
            aktualisiere_beste_ergebnisse(aktuelle_ergebnisse)
        
        if beste_ergebnisse_erst is not None:
            print("Beste Ergebnisse nach Abbruch:", beste_ergebnisse_erst)
        else:
            print("Keine Ergebnisse vorhanden.")

# Bayesian Optimization - erster Durchlauf mit erweiterter Hyperparameter-Räume
hyperparameter_raender = {
    'anzahl_neuronen_1': (64, 256),
    'anzahl_neuronen_2': (32, 128),
    'anzahl_neuronen_3': (16, 64),
    'learning_rate': (1e-4, 1e-2),  # Lernrate auf log-Skala
    'batch_size': (32, 128)          # Batch-Größe als diskreter Wert
}

# Initialisiere den Bayesian Optimizer
bo_optimizer = BayesianOptimization(
    f=trainiere_und_bewerte,
    pbounds=hyperparameter_raender,
    random_state=42,
    verbose=2
)

# Anzahl der initialen Punkte und Iterationen festlegen
init_points_zahl = 10
n_iter_zahl = 100

# Starte die Optimierung
optimierung_mit_speicherung(bo_optimizer, init_points_zahl, n_iter_zahl)


# %% [markdown]
# ## <a id='toc1_11_'></a>[Trainiere das finale Modell mit den besten Hyperparametern](#toc0_)
# Wir verwenden die im zweiten Durchlauf ermittelten Hyperparameter. Anschließend stellen wir sicher, dass wir das Modell etwas länger trainieren. Zusätzlich plotten wir den Training-/Validierungsloss.
# 
# Der Versuch das Modell auf erweiterter Hardware zu trainieren verlief leider erfolglos. Tensorflow hatte Probleme die Hardware parallel zu nutzen, der Scheduler scheint optimierungspotential zu haben.
# 
# ![training.png](training.png)

# %%
# Funktion, um den Validation-Loss des gespeicherten Modells zu berechnen
def berechne_val_loss_gespeichertes_modell(gespeichertes_modell, val_data):
    gespeichertes_modell.compile(optimizer='adam', loss='huber')
    val_loss = gespeichertes_modell.evaluate(val_data, val_data, verbose=0)
    return val_loss

# Datei mit den besten Ergebnissen
beste_ergebnisse_datei = "beste_ergebnisse.json"

# Beste Hyperparameter laden
if os.path.exists(beste_ergebnisse_datei):
    with open(beste_ergebnisse_datei, "r") as f:
        gespeicherte_ergebnisse = json.load(f)
        finale_parameter = gespeicherte_ergebnisse.get("hyperparameter", {})
else:
    raise FileNotFoundError("Die Datei mit den besten Ergebnissen wurde nicht gefunden!")

# Sicherstellen, dass die notwendigen Hyperparameter vorhanden sind
required_keys = [
    'anzahl_neuronen_1', 
    'anzahl_neuronen_2', 
    'anzahl_neuronen_3',
    'learning_rate',
    'batch_size'
]
if not all(key in finale_parameter for key in required_keys):
    raise ValueError(
        "Die gespeicherten Hyperparameter sind unvollständig! "
        "Erforderlich: anzahl_neuronen_1, anzahl_neuronen_2, "
        "anzahl_neuronen_3, learning_rate, batch_size"
    )

# Extrahiere Lernrate und Batch-Größe aus den finalen Parametern
learning_rate_final = finale_parameter['learning_rate']
batch_size_final = int(finale_parameter['batch_size'])  # Sicherstellen, dass es ein Integer ist

# Validation-Loss des gespeicherten Modells laden, falls vorhanden
if os.path.exists('autoencoder.h5'):
    try:
        gespeichertes_modell = tf.keras.models.load_model('autoencoder.h5', compile=False)
        gespeichertes_modell_val_loss = berechne_val_loss_gespeichertes_modell(
            gespeichertes_modell,
            val_data=train_fenster_shuffle[int(len(train_fenster_shuffle) * 0.8):]
        )
        print(f"Validation-Loss des gespeicherten Modells: {gespeichertes_modell_val_loss}")
    except Exception as e:
        print(f"Fehler beim Laden des gespeicherten Modells: {e}")
else:
    print("Kein gespeichertes Modell gefunden. Training eines neuen Modells.")

# Neues Modelltraining mit den finalen Hyperparametern
K.clear_session()

print("Verwende finale Hyperparameter:", finale_parameter)

finales_modell = baue_autoencoder(
    anzahl_neuronen_1=finale_parameter['anzahl_neuronen_1'],
    anzahl_neuronen_2=finale_parameter['anzahl_neuronen_2'],
    anzahl_neuronen_3=finale_parameter['anzahl_neuronen_3'],
    fenster_groesse=fenster_groesse,
    anzahl_merkmale=pca_modell.n_components_,
    learning_rate=learning_rate_final,  # Lernrate übergeben
    verlustfunktion='huber'
)

# Callbacks definieren
checkpoint_final = tf.keras.callbacks.ModelCheckpoint(
    'autoencoder_temp.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,              # Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
    restore_best_weights=True,  # Die besten Gewichte werden nach dem Stoppen wiederhergestellt
    verbose=1
)

# Trainings-Verlauf starten mit der geladenen Batch-Größe
trainings_verlauf = finales_modell.fit(
    train_fenster_shuffle,                # Eingabedaten (x)
    train_fenster_shuffle,
    batch_size=batch_size_final,          # Geladene Batch-Größe verwenden
    validation_split=0.2,
    epochs=100,
    callbacks=[checkpoint_final, early_stop],  # EarlyStopping 
    verbose=1
)

# Validation-Loss des neu trainierten Modells berechnen
neu_val_loss = min(trainings_verlauf.history['val_loss'])
print(f"Validation-Loss des neu trainierten Modells: {neu_val_loss}")

if 'gespeichertes_modell_val_loss' in locals():
    if neu_val_loss < gespeichertes_modell_val_loss:
        print("Das neu trainierte Modell ist besser. Es wird als 'autoencoder.h5' gespeichert.")
        if os.path.exists('autoencoder.h5'):
            os.remove('autoencoder.h5')  # Existierende Datei entfernen
        os.replace('autoencoder_temp.h5', 'autoencoder.h5')  # Datei umbenennen und überschreiben
    else:
        print("Das neu trainierte Modell ist nicht besser. Das gespeicherte Modell bleibt erhalten.")
        if os.path.exists('autoencoder_temp.h5'):
            os.remove('autoencoder_temp.h5')  # Temporäre Datei entfernen
else:
    # Wenn kein gespeichertes Modell existierte oder das Laden fehlgeschlagen ist
    print("Kein gespeichertes Modell vorhanden oder Laden fehlgeschlagen. Das neu trainierte Modell wird gespeichert als 'autoencoder.h5'.")
    os.replace('autoencoder_temp.h5', 'autoencoder.h5')

# Plot: Trainings- und Validierungs-Loss
plt.figure(figsize=(8, 6))
plt.plot(trainings_verlauf.history['loss'], label='Trainings-Loss')
plt.plot(trainings_verlauf.history['val_loss'], label='Validierungs-Loss')
plt.title('Trainings- und Validierungs-Loss (Autoencoder)')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# ### <a id='toc1_11_1_'></a>[Testen des Autoencoders](#toc0_)
# Die Funktion cross_validate_autoencoder führt eine K-Fold-Cross-Validation durch, um die Leistung eines Autoencoder-Modells zu bewerten. Das Modell wird auf mehreren Daten-Splits trainiert, und die Validierungsverluste werden gesammelt, um die Stabilität und Generalisierungsfähigkeit zu analysieren.

# %%
# Funktion zur Cross-Validation mit geladenem Modell
def cross_validate_autoencoder(n_splits=3, epochs=5, batch_size=batch_size_final):
    """
    Führt eine K-Fold-Cross-Validation auf 'train_fenster_shuffle' durch.
    
    Args:
        n_splits (int): Anzahl der Folds für KFold.
        epochs (int): Anzahl Trainings-Epochen pro Fold.
        batch_size (int): Batchgröße beim Training.
        
    Returns:
        List[float]: Liste der Validierungs-Verluste pro Fold.
    """
    # Validierungs-Verluste initialisieren
    val_losses = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Beste Hyperparameter und gespeichertes Modell laden
    beste_ergebnisse_datei = "beste_ergebnisse.json"
    if os.path.exists(beste_ergebnisse_datei):
        with open(beste_ergebnisse_datei, "r") as f:
            gespeicherte_ergebnisse = json.load(f)
            finale_parameter = gespeicherte_ergebnisse.get("hyperparameter", {})
    else:
        raise FileNotFoundError("Die Datei mit den besten Ergebnissen wurde nicht gefunden!")

    # Sicherstellen, dass die Hyperparameter vollständig sind
    required_keys = ['anzahl_neuronen_1', 'anzahl_neuronen_2', 'anzahl_neuronen_3']
    if not all(key in finale_parameter for key in required_keys):
        raise ValueError("Die gespeicherten Hyperparameter sind unvollständig!")

    # Cross-Validation-Schleife
    for train_index, val_index in kf.split(train_fenster_shuffle):
        X_train_fold = train_fenster_shuffle[train_index]
        X_val_fold = train_fenster_shuffle[val_index]

        # Neues Modell pro Fold bauen
        K.clear_session()
        model_fold = baue_autoencoder(
            anzahl_neuronen_1=finale_parameter['anzahl_neuronen_1'],
            anzahl_neuronen_2=finale_parameter['anzahl_neuronen_2'],
            anzahl_neuronen_3=finale_parameter['anzahl_neuronen_3'],
            fenster_groesse=fenster_groesse,
            anzahl_merkmale=pca_modell.n_components_,
            learning_rate=learning_rate_final,
            verlustfunktion='huber'
        )

        # Training für diesen Fold
        history_fold = model_fold.fit(
            X_train_fold,
            X_train_fold,
            epochs=100, 
            batch_size=batch_size,
            validation_data=(X_val_fold, X_val_fold),
            verbose=0
        )

        # Bester Validation-Loss in diesem Fold
        best_val_loss_fold = min(history_fold.history['val_loss'])
        val_losses.append(best_val_loss_fold)

    return val_losses


# Beispiel-Aufruf der Funktion
val_losses_cv = cross_validate_autoencoder(n_splits=3, epochs=5, batch_size=64)
mean_val_loss_cv = statistics.mean(val_losses_cv)
std_val_loss_cv = statistics.pstdev(val_losses_cv)  # Populations-Streuung oder stdev()

# 95%-Konfidenzintervall berechnen
confidence_95 = 1.96 * (std_val_loss_cv / math.sqrt(len(val_losses_cv)))

# Ergebnisse ausgeben
print("Cross Validation Ergebnisse:")
print("Fold-wise Validation Losses:", val_losses_cv)
print(f"Mean Val Loss = {mean_val_loss_cv:.5f}")
print(f"Std Val Loss  = {std_val_loss_cv:.5f}")
print(f"95%-Konfidenzintervall: ["
      f"{mean_val_loss_cv - confidence_95:.5f}, {mean_val_loss_cv + confidence_95:.5f}]")


# %% [markdown]
# ## <a id='toc1_12_'></a>[Verarbeitung und Vorbereitung der Testdaten](#toc0_)
# 

# %% [markdown]
# #### <a id='toc1_12_1_1_'></a>[Schritte zur Datenverarbeitung](#toc0_)
# 1. Laden der Testdaten
# 2. Filtern auf HTTP
# 3. Label-Erzeugung (0=normal, 1=anomal)
# 4. Fensterbildung mit zugehörigen Labels
# 

# %% [markdown]
# #### <a id='toc1_12_1_2_'></a>[Anmerkung](#toc0_)
# Durch diese Schritte liegen die Testdaten im gleichen Format vor wie die Trainingsdaten.

# %%
df_test = pd.read_csv(TEST_DATEN_PFAD, header=None, names=bereinigte_spaltennamen)
df_test_http = df_test[df_test['service']=='http'].copy()

zeilen_label = (df_test_http['result']!='normal.').astype(int)
test_fenster_labels = []
for i in range(0, len(zeilen_label)-fenster_groesse+1, schrittweite):
    slice_sum = sum(zeilen_label[i:i+fenster_groesse])
    test_fenster_labels.append(1 if slice_sum>0 else 0)

# Kategoriale spalten löschen/one hot
cat_cols_test = df_test_http.select_dtypes(include=['object','category']).columns
if dummifiziere_categoricals:
    df_test_http = pd.get_dummies(df_test_http, columns=cat_cols_test, drop_first=False, dtype=int)
else:
    df_test_http.drop(columns=cat_cols_test, inplace=True)

std_test = df_test_http.std()
no_var_test = std_test[std_test==0].index
df_test_http.drop(columns=no_var_test, inplace=True)

train_final_cols = df_train_http_normal.columns
df_test_http = df_test_http.reindex(columns=train_final_cols, fill_value=0)

# Skalierung
df_test_http = pd.DataFrame(
    skalierer.transform(df_test_http),
    columns=df_test_http.columns
)

# PCA
df_test_http_pca = pd.DataFrame(
    pca_modell.transform(df_test_http),
    columns=[f'PCA_{i}' for i in range(pca_dim)]
)

test_fenster_daten = fenster_bilden(df_test_http_pca, fenster_groesse, schrittweite)
test_fenster_daten.shape, len(test_fenster_labels)

# %% [markdown]
# ## <a id='toc1_13_'></a>[Bewertung des LSTM-Autoencoders und Anomalieerkennung](#toc0_)
# Wir laden das beste Modell (`autoencoder.h5`), berechnen Rekonstruktionsfehler und skalieren diese zu Anomalie-Scores.
# 

# %% [markdown]
# #### <a id='toc1_13_1_1_'></a>[Schritte zur Modellbewertung](#toc0_)
# 1. Laden des Modells
# 2. Vorhersage (Rekonstruktion)
# 3. Rekonstruktionsfehler
# 4. Anomaliescore-Berechnung
# 5. Visualisierung
# 

# %%
# bestes Modell laden
autoencoder_geladen = tf.keras.models.load_model('autoencoder.h5', compile=False)

test_fenster_vorhersage = autoencoder_geladen.predict(test_fenster_daten)

def berechne_rekon_fehler(fenster_original, fenster_pred, fenster_size, anzahl_dims):
    fehler_liste = []
    for i in range(fenster_original.shape[0]):
        diff = fenster_original[i] - fenster_pred[i]
        # Mittelwert der Norm über die Zeitachse
        re = np.mean(np.linalg.norm(diff, axis=1))
        fehler_liste.append(re)
    return np.array(fehler_liste)

rekonstruktions_fehlertest = berechne_rekon_fehler(
    test_fenster_daten,
    test_fenster_vorhersage,
    fenster_groesse,
    pca_modell.n_components_
)
rekonstruktions_fehlertest = rekonstruktions_fehlertest.reshape(-1, 1)

minmax_scaler = MinMaxScaler()
anomalie_scores_test = minmax_scaler.fit_transform(rekonstruktions_fehlertest).flatten()

# Plot
plt.figure(figsize=(16, 6))
plt.plot(test_fenster_labels, c='blue', label='Wahre Labels (0=normal,1=anomal)')
plt.plot(anomalie_scores_test, c='red', label='Anomalie-Score [0..1]')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Test-Fenster')
plt.ylabel('Anomalie-Score')
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# ## <a id='toc1_14_'></a>[Verschiedene Metriken](#toc0_)

# %% [markdown]
# ## <a id='toc1_15_'></a>[Kategoriale Kenngrößen](#toc0_)

# %% [markdown]
# ## <a id='toc1_16_'></a>[Metriken für Klassifikationsmodelle](#toc0_)
# 
# ### <a id='toc1_16_1_'></a>[Accuracy (Standard)](#toc0_)
# Im einfachsten Fall wird die Summe der Übereinstimmungen $ \hat{Y} = Y $ durch die insgesamte Anzahl an Datensätzen dividiert. Dadurch wird der Prozentsatz der Übereinstimmungen ermittelt. Diese Metrik eignet sich bei Single-Label-Klassifikationen (pro Datensatz gibt es nur eine einzige Ausprägung von $ Y $). Die Accuracy ist in diesem Datensatz mit Vorsicht zu genießen, die Datenanalyse am Anfang hat gezeigt, dass nur 0,48% der Daten Anormalien sind, bei einer Accuracy von 0,995 wäre es also möglich, dass keine einzige Anormalie erkannt wurde, trotzdem sieht die Accuracy erstmal gut aus.
# 
# $
# \text{Accuracy} = \frac{\text{Summe der Übereinstimmungen}}{\text{Gesamtanzahl der Datensätze}}
# $
# 
# #### <a id='toc1_16_1_1_'></a>[Precision](#toc0_)
# Diese Metrik errechnet sich gemäß:
# 
# $
# \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
# $
# 
# ---
# 
# #### <a id='toc1_16_1_2_'></a>[Recall](#toc0_)
# Diese Metrik errechnet sich gemäß:
# 
# $
# \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
# $
# 
# ---
# 
# #### <a id='toc1_16_1_3_'></a>[F1-Score](#toc0_)
# Der F1-Score wird als harmonisches Mittel von Precision und Recall berechnet:
# 
# $
# \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
# $
# 
# ---
# 
# #### <a id='toc1_16_1_4_'></a>[Tabelle: Übersicht der Begriffe](#toc0_)
# | Tatsächlich/Vorhersage | **1** (Positiv) | **0** (Negativ) |
# |-------------------------|-----------------|-----------------|
# | **1** (Positiv)         | TPR (True Positive) | FNR (False Negative) |
# | **0** (Negativ)         | FPR (False Positive) | TNR (True Negative) |
# 
# 

# %% [markdown]
# ### <a id='toc1_16_2_'></a>[ROC und AUC-Bewertung](#toc0_)

# %% [markdown]
# #### <a id='toc1_16_2_1_'></a>[Visualisierung](#toc0_)
# Wir plotten FPR gegen TPR und geben den AUC-Wert an.
# 

# %% [markdown]
# #### <a id='toc1_16_2_2_'></a>[Wichtige Aspekte](#toc0_)
# - Kurve nahe (0,1) deutet auf sehr gute Trennung hin
# - Diagonale = zufällige Klassifikation
# 

# %%
fpr_roc, tpr_roc, thresholds_roc = roc_curve(test_fenster_labels, anomalie_scores_test)
auc_roc = roc_auc_score(test_fenster_labels, anomalie_scores_test)
print('AUC-Wert:', auc_roc)

plt.figure(figsize=(6, 4))
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.plot(fpr_roc, tpr_roc, label=f'AUC={auc_roc:.4f}')
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC-Kurve')
plt.show()

# %% [markdown]
# ### <a id='toc1_16_3_'></a>[Precision-Recall-Kurve und Average Precision (AP)](#toc0_)

# %%
precisions, recalls, schwellen_pr = precision_recall_curve(test_fenster_labels, anomalie_scores_test)
ap_wert = average_precision_score(test_fenster_labels, anomalie_scores_test)
print('Average Precision (AP):', ap_wert)

plt.figure(figsize=(6, 4))
plt.plot(recalls, precisions, label=f'AP={ap_wert:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall-Kurve')
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# ### <a id='toc1_16_4_'></a>[F1-Score-Kurve basierend auf Schwellenwerten](#toc0_)

# %% [markdown]
# #### <a id='toc1_16_4_1_'></a>[Schritte zur Berechnung der F1-Score-Kurve](#toc0_)
# 1. Durchiterieren verschiedener Schwellen.
# 2. Klassifikation (0/1) je nach Score.
# 3. F1-Score berechnen.
# 

# %%
alle_schwellen = np.sort(anomalie_scores_test)
f1_werte = []
for schwelle in alle_schwellen:
    prädiktion = (anomalie_scores_test > schwelle).astype(np.int32)
    f1_aktuell = f1_score(test_fenster_labels, prädiktion)
    f1_werte.append(f1_aktuell)

plt.figure(figsize=(10, 5))
plt.plot(alle_schwellen, f1_werte)
plt.grid()
plt.xlabel('Schwellenwerte')
plt.ylabel('F1-Score')
plt.title('F1-Score vs. Schwellenwerte')
plt.show()

# %% [markdown]
# ### <a id='toc1_16_5_'></a>[Schwellenwertoptimierung und Bewertung der Anomalieerkennung](#toc0_)

# %% [markdown]
# #### <a id='toc1_16_5_1_'></a>[Berechnung der Bewertungsmetriken](#toc0_)
# - Konfusionsmatrix
# - Precision, Recall, F1, Accuracy
# 

# %%
# Schwellenwertoptimierung
schwellen_alle = np.sort(anomalie_scores_test)
bester_f1 = 0
bester_schwellenwert = 0

for schwelle in schwellen_alle:
    pred_schwell = (anomalie_scores_test > schwelle).astype(int)
    akt_f1 = f1_score(test_fenster_labels, pred_schwell)
    if akt_f1 > bester_f1:
        bester_f1 = akt_f1
        bester_schwellenwert = schwelle

print(f'Bester Schwellenwert auf Basis F1={bester_schwellenwert:.4f}, F1={bester_f1:.4f}')

final_pred = (anomalie_scores_test > bester_schwellenwert).astype(int)
konf_matrix = confusion_matrix(test_fenster_labels, final_pred)

# Verbesserte Darstellung der Konfusionsmatrix
labels = ['Normal', 'Anomal']
plt.figure(figsize=(8, 6))
sns.heatmap(konf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Konfusionsmatrix')
plt.xlabel('Vorhergesagte Labels')
plt.ylabel('Wahre Labels')
plt.show()

precision_erg = precision_score(test_fenster_labels, final_pred)
recall_erg = recall_score(test_fenster_labels, final_pred)
f1_erg = f1_score(test_fenster_labels, final_pred)
accuracy_erg = accuracy_score(test_fenster_labels, final_pred)

print('Endgültige Klassifikationsmetriken:\n',
      f'Precision = {precision_erg:.4f}\n',
      f'Recall    = {recall_erg:.4f}\n',
      f'F1        = {f1_erg:.4f}\n',
      f'Accuracy  = {accuracy_erg:.4f}\n')

# Visualisierung
plt.figure(figsize=(12, 5))
sns.scatterplot(
    x=np.arange(0, len(anomalie_scores_test)), 
    y=anomalie_scores_test,
    hue=['normal' if x==0 else 'anomal' for x in final_pred],
    palette=['blue', 'red'], legend='full'
)
plt.axhline(y=bester_schwellenwert, linestyle='--', label='Bester Schwellenwert')
plt.xlabel('Index Testfenster')
plt.ylabel('Anomalie-Score')
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# ## <a id='toc1_17_'></a>[Anomalieerkennung im Produktionsdatensatz](#toc0_)
# Abschließend wenden wir das trainierte Modell auf Produktionsdaten an (unbekannte Daten).

# %% [markdown]
# #### <a id='toc1_17_1_1_'></a>[Schritte zur Datenverarbeitung](#toc0_)
# - Laden
# - HTTP filtern
# - Cleaning/Skalierung/PCA
# - Fensterbildung
# - Rekonstruktionsfehler & Scores
# - Klassifikation über best. Schwellenwert

# %% [markdown]
# #### <a id='toc1_17_1_2_'></a>[Anmerkung](#toc0_)
# Dies dient als Demonstration, wie ein Modell in Produktionsumgebungen Abweichungen erkennen kann.
# 

# %%
df_prod = pd.read_csv(PROD_DATEN_PFAD, header=None, names=bereinigte_spaltennamen)

# nur http
df_prod_http = df_prod[df_prod['service'] == 'http'].copy()
print('Form des Produktionsdatensatzes (HTTP):', df_prod_http.shape)

# Cleaning
df_prod_http.drop([
    'protocol_type', 'service', 'flag', 'land',
    'is_host_login', 'is_guest_login', 'result'
], axis=1, inplace=True)
df_prod_http.drop([
    'wrong_fragment', 'urgent', 'num_failed_logins', 'su_attempted',
    'num_file_creations', 'num_outbound_cmds'
], axis=1, inplace=True)
# aufgrund hoher Korrelationen wird die Spalte 'logged_in' entfernt (leide nur manuell)
df_prod_http =df_prod_http.drop(columns=['logged_in'])
# Skalierung
df_prod_http = pd.DataFrame(
    skalierer.transform(df_prod_http),
    columns=df_prod_http.columns
)

# PCA
df_prod_http_pca = pd.DataFrame(
    pca_modell.transform(df_prod_http),
    columns=df_train_pca.columns
)

prod_fenster = fenster_bilden(df_prod_http_pca, fenster_groesse=fenster_groesse, schritt=schrittweite)
prod_fenster_pred = autoencoder_geladen.predict(prod_fenster)

prod_fehler = berechne_rekon_fehler(
    prod_fenster,
    prod_fenster_pred,
    fenster_groesse,
    pca_modell.n_components_
)
prod_fehler = prod_fehler.reshape(-1, 1)

# Normalisierung
scaler_prod = MinMaxScaler()
prod_scores = scaler_prod.fit_transform(prod_fehler).flatten()

prod_anomal = (prod_scores > bester_schwellenwert).astype(np.int32)

plt.figure(figsize=(20,6))
sns.scatterplot(
    x=np.arange(0, len(prod_scores)),
    y=prod_scores,
    hue=['normal' if x==0 else 'anomal' for x in prod_anomal],
    palette=['blue', 'red'], legend='full'
)
plt.axhline(y=bester_schwellenwert, linestyle='--', label='Schwellenwert', color='gray')
plt.title('Anomalieerkennung im Produktionsdatensatz')
plt.xlabel('Index Produktions-Fenster')
plt.ylabel('Anomalie-Score')
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# ### <a id='toc1_17_2_'></a>[Ausgabe der gefundenen Anormalien](#toc0_)

# %%
# Extraktion der Anomalie-Fenster
anomalie_df_liste = []

for i, (score, ind) in enumerate(zip(prod_scores, prod_anomal)):
    if ind == 1:
        start_idx = i * schrittweite
        end_idx = start_idx + fenster_groesse
        fenster_subset = df_prod_http.iloc[start_idx:end_idx].copy()
        fenster_subset['anomalie_score'] = score
        fenster_subset['fenster_index'] = i
        anomalie_df_liste.append(fenster_subset)

anomalien_prod = pd.concat(anomalie_df_liste, axis=0)
anomalien_prod = anomalien_prod.sort_values(by='anomalie_score', ascending=False)

print("Gefundene Anomalien (sortiert nach Score):")
print(anomalien_prod)

# Optional als CSV
anomalien_prod.to_csv("anomalien_produktiv.csv", index=False)
print("Die Tabelle der Anomalien wurde in 'anomalien_produktiv.csv' gespeichert.")

# %% [markdown]
# ## <a id='toc1_18_'></a>[Ausblick](#toc0_)
# 
# Das Modell liefert noch keine perfekten Ergebnisse, weitere Anpassungen auf Basis neuerster Referenz Entwicklungen könnte sinnvoll sein.
# 
# Um den Wert der Ergebnisse zu erhöhen sollte in Zukunft das Modell um eine Klassifizierung der Angriffe erweiterte werden.
# 
# Es ist zu beachten, dass der KDD mittlerweile veraltet ist, daher könnten die Logdaten in der Bachelorthesis, welche vom BSI bereitgestellt werden, neue Chancen aber auch Probleme bergen.
# Die Logs des BSI werden nicht gelabelt sein, was den bisherigen Ansatz unbrauchbar macht. Ziel könnte es sein einen Weg zu finden ohne ein manuelles Labeling die Daten nutzbar zu machen.
# 
# Das Modell sollte im besten Fall in der Lage sein nahezun in Echtzeit Daten zu bewerten.
# 
# Konsistente Datenbereinigung über Funktion wäre sinnvoll.
# 


