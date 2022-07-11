import json
import random
import datetime
import spacy
from spacy.tokens import DocBin
from spacy.tokens import Doc
from spacy.cli.train import train
from spacy.training.example import Example


# ----------------------------------
# ------------ Options: ------------
# ----------------------------------

modelsToTrain = 10 # Anzahl der Modelle, die trainiert werden sollen
evalSteps = 10     # Anzahl an verschiedenen Evaluationsdatensatzgrößen 
evalsPerStep = 25  # Anzahl an Evaluationen pro Modell und Datensatzgröße

anzTrainData = 25  # Größe des Trainingsdatensatzes
anzDevData = 20    # Größe des Entwicklungsdatensatzes (development data)
anzEvalDataStart = 2    # Startgröße des Evaluationsdatensatzes
evalDataIncrementer = 2 # wird nach jedem Durchgang auf die Evaluationsdatensatzgröße addiert

# maximale Anzahl von Wörtern um die Spangrenzen (Labelgrenzen) verschoben werden können.
# Wird benötigt, um Fehler zu vermeiden
maxShift = 2

modelNumber = "3_10"       # Nummer des Datensatzes, der verwendet werden soll
datasetName = "editedData" # Name der Datei mit dem vollständigen gelabelten Datensatz


# -----------------------------------
# ----------- Funktionen: -----------
# -----------------------------------

"""
Lädt gelabelte Daten des Labelschemas mit der angegebenen Nummer (modelNumber)
"""
def loadCVs(modelNumber, datasetName):
    with open('./corpus/Modell_{}/{}.jsonl'.format(modelNumber, datasetName), 'r', encoding="utf-8") as f:
        for cv in f:
            cvs = json.loads(cv) 
    return cvs['CVs']

"""
Erstellt Datensatz der Länge 'rangeLen' mit zufälligen CVs aus 'cvs'.
Achtet dabei darauf keine Elemente doppelt oder aus 'alreadyChoosen' hinzuzufügen.
"""
def createDataset(rangeLen, cvs, alreadyChoosen):
    dataset = []
    for m in range(rangeLen):
        randomCV = random.choice(cvs)
        while randomCV in dataset or randomCV in alreadyChoosen:
            randomCV = random.choice(cvs)
        dataset.append(randomCV)
    return dataset

"""
Konvertiert den übergebenen Datensatz ins spacy-Format und speichert es unter dem angegebenen Namen.
"""
def convertForSpacy(dataset, datasetName, modelNumber, nlp, maxShift):
    db = DocBin()
    for cv in dataset:
        text = cv['text']
        annotations = cv['label']
        doc = nlp(text)
        ents = []
        for annotation in annotations:
            span = doc.char_span(annotation[0], annotation[1], label=annotation[2])
            # Brauche die Loop für Fälle die sonst als None gespeichert werden und Errors erzeugen
            # Solche Fälle werden hier abgefangen und die Labelgrenze um bis zu 'maxShift' Stellen nach rechts verschoben
            # Wird der Span trotzdem nicht akzeptiert, wird das Label verworfen
            # Beispielfälle: 
              # Data: "... in Release 1905." -> nicht akzeptiert: "Release 1905", akzeptiert: "Release 1905." 
              # Data: "QM&EAM", nicht akzeptiert: "QM" und "EAM", akzeptiert: "QM&EAM" 
            canSave = False
            start = int(annotation[0])
            end = int(annotation[1])
            counter = 0
            #verschiebt die Grenze nach hinten, solange der span ungültig ist
            while span == None and counter <= maxShift:
                counter = counter + 1
                end = end + 1
                span = doc.char_span(start, end, label=annotation[2])

            if not span == None:
                canSave = True
            # geht hier rein, falls der span immer noch ungültig ist
            else:
                end = int(annotation[1])
                counter = 0;
                # setzt die Spangrenze zurück und verschiebt sie diesmal nach vorne, solange der span ungültig ist
                while span == None and counter <= maxShift:
                    counter = counter + 1
                    start = start - 1
                    span = doc.char_span(start, end, label=annotation[2])
                if not span == None:
                    canSave = True
            # Speichert Label, falls kein Fehler aufgetreten ist
            if canSave:
                ents.append(span)
                
        # Speichert die Label in spaCys Format
        try:
            doc.ents = ents
        except:
            print("ERROR")
        db.add(doc)
    # Speichert Daten unter dem Namen aus 'datasetName'
    db.to_disk("./corpus/Modell_{}/{}.spacy".format(modelNumber, datasetName))

"""
Evaluiert das trainierte Modell und gibt die ermittelte Performance zurück
"""
def evaluateModel(model, dataset):
    data = []
    for cv in evalData:
        spacy.training.offsets_to_biluo_tags(model.make_doc(cv['text']), cv['label'])
        data.append(Example.from_dict(model.make_doc(cv['text']), {"entities": cv['label']}))
    scores_model = model.evaluate(data)
    return {
        "f_score": scores_model["ents_f"],
        "precision": scores_model["ents_p"],
        "recall": scores_model["ents_r"]
    }
    
"""
Generiert Übersicht über das Modell und seine Performance
"""
def generateOverview(modell, evaluationScores, evalSize, durationTrain, durationEval, modelNr, step, run):
    return {
        "modelNr": modelNr,
        "step": step,
        "run": run,
        "evalSize": evalSize,
        "durationTrain": durationTrain,
        "durationEval": durationEval,
        "performance": {
            "f-score": modell.meta["performance"]['ents_f'],
            "precision": modell.meta["performance"]['ents_p'],
            "recall": modell.meta["performance"]['ents_r'],
            "tok2vec_loss": modell.meta["performance"]["tok2vec_loss"],
            "ner_loss": modell.meta["performance"]["ner_loss"]
        },
        "evaluation": evaluationScores
    }

"""
Exportiert die Übersicht über alle Ergebnisse 
"""
def exportModellOverview(overview, name, doIndent):
    overviewDict = {"Modelle": overview}
    fileName = './EvalModels/ModellOverview_{}.jsonl'.format(name)
    with open(fileName, 'w', encoding='utf-8') as f:
        if doIndent:
            json.dump(overviewDict, f, ensure_ascii=False, indent=3)
        else:
            json.dump(overviewDict, f, ensure_ascii=False)


# ----------------------------------
# ------------- Start: -------------
# ----------------------------------
    
#leeres Modell importieren
nlp = spacy.blank("de")
cvs = loadCVs(modelNumber, datasetName)

# Hier werden die Informationen zu allen trainierten und evaluierten Modellen gespeichert. 
# Wird am Ende als json-Datein exportiert
modelOverviews = []

# trainiert modelsToTrain-viele Modelle auf zufälligen Datensätzen
for modelNr in range(modelsToTrain):
    anzEvalData = anzEvalDataStart
    
    #erstellt zufälligen Trainings- & Entwicklungsdatensatz
    trainData = createDataset(anzTrainData, cvs, [])
    devData = createDataset(anzDevData, cvs, trainData)

    #wandelt Datensets in spaCy-Format um
    convertForSpacy(trainData, "train", modelNumber, nlp, maxShift)
    convertForSpacy(devData, "dev", modelNumber, nlp, maxShift)

    print("----------------- Train -----------------")
    #trainiert und lädt Modell auf ausgewähltem Datenset
    durationTrain = datetime.datetime.now()
    train("./EvalModels/config.cfg", "./EvalModels/TempModell")
    modell = spacy.load("./EvalModels/TempModell/model-best/")
    durationTrain = datetime.datetime.now() - durationTrain
    
    # evaluiert jedes Modell auf evalSteps-vielen verschiedenen Evaluierungsdatenmengen
    for step in range(evalSteps):
        # evaluiert jedes Modell auf jeder Datenmenge evalsPerStep-mal
        for run in range(evalsPerStep):    
            durationEval = datetime.datetime.now()
            
            print("---------------- Evaluate ---------------")
            #evaluiert Modell mit zufälligen Daten
            evalData = createDataset(anzEvalData, cvs, trainData)
            evaluationScores = evaluateModel(modell, evalData)
            durationEval = datetime.datetime.now() - durationEval

            #Erstellt Übersicht über die Performance des aktuellen Modells
            modelOverviews.append(generateOverview(modell, evaluationScores, anzEvalData, str(durationTrain), str(durationEval), modelNr, step, run))
            #Exportiert Übersicht über die Performance aller bisherigen Modelle,
            # um einen Überblick über den Fortschritt zu bekommen
            exportModellOverview(modelOverviews, "currentStatus", True)
            
        # erhöht Evaluationsdatensatzgröße
        anzEvalData = anzEvalData + evalDataIncrementer

print("---------------- Finished ---------------")
#Exportiert Übersicht über die Performance aller Modelle
exportModellOverview(modelOverviews, "finished", False)
