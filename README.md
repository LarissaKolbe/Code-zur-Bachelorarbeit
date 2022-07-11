# NER in structured Documents - Python-Code

## Datensatz- und Labelverarbeitung

`CVwiseToLineWise` ... Konvertiert CV-weisen doccano-Datensatz in zeilenweises Format. Kann mit `JsonToSpaCy_Lines` in spaCys Format umgewandelt werden. Habe ich genutzt, um den zeilenweises Datensatz des Divided Skills Schemas zu erstellen.

`EditLabeledData` ... Passt wenn nötig automatisch einzelne Label in einer doccano json-Datei an und exportiert die bearbeitete Datei. Zur Vereinheitlichung der Label genutzt. Passt Label nur an oder löscht sie nach bestimmten Regeln, kann aber keine neuen erstellen. Ist auf Divided Skills Labelschema ausgelegt. Die ausgegebene Datei kann mit "JsonToSpaCy_Edited.ipynb" in spaCys Format umgewandelt werden.

`EditLabeledDataAndGetInfos` ... Wie `EditLabeledData` mit dem Zusatz, dass weiter unten noch genauere Informationen zu den verteilten Labeln berechnet werden.


`JsonToSpaCy` ... Wandelt einen unbearbeiteten doccano-Datensatz von json in eine spaCys Datei um, mit der ein Modell trainiert werden kann.

`JsonToSpaCy_Edited` ... Wandelt einen mit `EditLabeledData` angepassten Datensatz in spaCys Format um.

`JsonToSpaCy_Lines` ... Konvertiert zeilenweisen json-Datensatz in spaCys Format. Nimmt als Input einen mit `CVwiseToLineWise` erstellten Datensatz.


## Versuchsreihen und Auswertungen

`EvalDataExperiment` ... Der Code mit dem das Evaluationsdatenmengen-Experiment durchgeführt wurde.

`EvalDataExperiment_Auswertung` ... Zur Auswertung des Evaluationsdatenmengen-Experiments verwendet.


`TrainDataExperiment` ... Der Code mit dem das Trainingsdatenmengen-Experiment durchgeführt wurde.

`TrainDataExperiment_Auswertung` ... Zur Auswertung des Trainingsdatenmengen-Experiments verwendet.


`TrainDauerExperiment` ... Der Code mit dem das Trainingsdauer-Experiment durchgeführt wurde.

`TrainDauerExperiment_Auswertung` ... Zur Auswertung des Trainingsdauer-Experiments verwendet.


`TrainModellsOnLabelschemas` ... Zum Bestimmen der Performance der einzelnen Labelschemen verwendet.

`TrainModellsOnLabelschemas_Auswertung` ... Zur Auswertung der Performance der einzelnen Labelschemen verwendet.


`Labelstudie_Auswertung` ... Zur Auswertung der Umfrage zur Labelkonsistenz verwendet.


## Weiteres

`TestModels` ... Wendet ein trainiertes Modelle auf einen CV an. Stellt den gelabelten CV anschließend grafisch dar.
