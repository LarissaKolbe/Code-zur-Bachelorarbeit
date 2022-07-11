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

nrDiffTrainDataLen = 9 # Anzahl an verschiedenen Trainingsdatensatzgrößen
runsPerStep = 100      # Anzahl zu trainierender Modelle, pro Trainingsdatensatzgröße

initialAnzTrainData = 0  # Startgröße des Trainingsdatensatzes. Wird direkt zu beginn mit trainDataIncrementer erhöht
trainDataIncrementer = 4 # Schrittgröße in der die Trainingsdatensatzgröße (anzTrainData) erhöht wird
anzEvalData = 12         # Größe des Evaluations- und Entwicklungsdatensatzes

maxShift = 2      # Maximale Anzahl von Wörtern um die Spangrenzen (Labelgrenzen) verschoben werden können. Wird benötigt, um Fehler zu vermeiden
maxOccurence = 2  # Anzahl wie oft jedes einzelne Datenset (bspw Trainingsdatenset) sich wiederholen darf 

modelNumber = "3_10"       # Nummer des Datensatzes, der verwendet werden soll
datasetName = "editedData" # Name der Datei mit dem vollständigen gelabelten Datensatz


# -----------------------------------
# ----------- Funktionen: -----------
# -----------------------------------

"""
Lädt gelabelte Daten
"""
def loadCVs(modelNumber, datasetName):
    with open('./corpus/Modell_{}/{}.jsonl'.format(modelNumber, datasetName), 'r', encoding="utf-8") as f:
        for cv in f:
            cvs = json.loads(cv) 
    return cvs['CVs']

"""
Erstellt Datensatz der Länge 'rangeLen' mit zufälligen CVs aus 'cvs'.
Achtet dabei darauf keine Elemente doppelt oder aus 'alreadyChoosen' hinzuzufügen. 
Jeder Datensatz darf nur 'maxOccurence'-mal verwendet werden
"""
def createDataset(rangeLen, cvs, alreadyChoosen, alreadyUsedDatasets, maxOccurence):
    # erstellt neuen Datensatz, falls aktueller bereits mehrmals (> maxOccurence) in alreadyUsedDatasets ist
    # soll verhindern, dass ständig auf demselben Dataset trainiert wird
    while True:
        dataset = []
        indexes = []
        for m in range(rangeLen):
            randomCV = random.choice(cvs)
            index = cvs.index(randomCV)
            while index in indexes or index in alreadyChoosen:
                if index + 1 < len(cvs):
                    index = index + 1
                else:
                    index = 0
                randomCV = cvs[index]
            dataset.append(randomCV)
            indexes.append(index)
        indexes.sort()
        counter = 0
        # zählt wie oft dieser Datensatz bereits vorkam
        for element in (element for element in alreadyUsedDatasets if str(indexes) == element):
            counter = counter + 1
        # wenn dar Datensatz noch nicht so oft vorkam wird er gespeichert und verwendet, 
        # anderenfalls wird ein neuer erstellt
        if counter < maxOccurence:
            alreadyUsedDatasets.append(str(indexes))
            break
    return dataset, indexes, alreadyUsedDatasets

"""
Konvertiert das übergebene Dataset in spacy-Format und speichert es unter dem angegebenen Namen.
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
            # Solche Fälle werden hier abgefangen und die Labelgrenze um bis zu 5 Stellen nach rechts verschoben
            # Wird der Span trotzdem nicht akzeptiert, wird das Label verworfen
            # Beispielfälle: 
              # Data: "... in Release 1905.", gelabelt abr nicht akzeptiert: "Release 1905", akzeptierter Span: "Release 1905." 
              # Data: "QM&EAM", gelabelt abr nicht akzeptiert: "QM" und "EAM", akzeptierter Span: "QM&EAM" 
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
Evaluiert das trainierte Modell
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
def generateOverview(modell, evaluationScores, trainSize, evalSize, durationTrain, durationEval, step, run, trainIndexes, devIndexes, evalIndexes):
    return {
        "step": step,
        "run": run,
        "trainSize": trainSize,
        "evalSize": evalSize,
        "durationTrain": durationTrain,
        "durationEval": durationEval,
        "trainIndexes": str(trainIndexes), 
        "devIndexes": str(devIndexes), 
        "evalIndexes": str(evalIndexes),
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
Exportiert den aktuellen Stand aller Ergebnisse mit Einrückungen
"""
def exportCurrentStatusModellOverview(overview, name):
    overviewDict = {"Modelle": overview}
    fileName = './TrainModels/ModellOverview_{}.jsonl'.format(name)
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(overviewDict, f, ensure_ascii=False, indent=3)
            
"""
Exportiert die komplette Übersicht über die Ergebnisse ohne Einrückungen
"""
def exportFinishedModellOverview(overview, nrCVs, nrSteps):
    overviewDict = {"Modelle": overview}
    fileName = './TrainModels/ModellOverview_{}CVs_{}Steps.jsonl'.format(nrCVs, nrSteps)
    with open(fileName, 'w', encoding='utf-8') as f:
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

anzTrainData = initialAnzTrainData

# trainiert nrDiffTrainDataLen-viele Modelle auf zufälligen Datensätzen
for step in range(nrDiffTrainDataLen):
    anzTrainData = anzTrainData + trainDataIncrementer  
    
    # speichert bereits verwendete Datensätze, damit diese nicht nochmal vorkommen
    alreadyUsedDatasetsTrain = []
    alreadyUsedDatasetsDev = []
    alreadyUsedDatasetsEval = []
    alreadyUsedDatasets = []

    # trainiert runsPerStep-viele Modelle pro Trainingsdatensatzgröße
    for run in range(runsPerStep):
        #erstellt zufälligen Trainings-, Entwicklungs- und Evaluierungsdatensatz
        # geht sicher, dass keine Kombination doppelt verwendet wird
        didNotRunYet = True
        while didNotRunYet or [trainIndexes, devIndexes, evalIndexes] in alreadyUsedDatasets:
            trainData, trainIndexes, alreadyUsedDatasetsTrain = createDataset(anzTrainData, cvs, [], alreadyUsedDatasetsTrain, maxOccurence)
            devData, devIndexes, alreadyUsedDatasetsDev = createDataset(anzEvalData, cvs, trainIndexes, alreadyUsedDatasetsDev, maxOccurence)
            evalData, evalIndexes, alreadyUsedDatasetsEval = createDataset(anzEvalData, cvs, trainIndexes, alreadyUsedDatasetsEval, maxOccurence)
            didNotRunYet = False
        
        alreadyUsedDatasets.append([trainIndexes, devIndexes, evalIndexes])
        
        #wandelt Datensets in spaCy-Format um
        convertForSpacy(trainData, "train", modelNumber, nlp, maxShift)
        convertForSpacy(devData, "dev", modelNumber, nlp, maxShift)

        print("----------------- Train -----------------")
        #trainiert und lädt Modell auf ausgewähltem Datenset
        durationTrain = datetime.datetime.now()
        train("./TrainModels/config.cfg", "./TrainModels/TempModell")
        durationTrain = datetime.datetime.now() - durationTrain
        modell = spacy.load("./TrainModels/TempModell/model-best/")

        print("---------------- Evaluate ---------------")
        #evaluiert Modell mit zufälligen Daten
        durationEval = datetime.datetime.now()
        evaluationScores = evaluateModel(modell, evalData)
        durationEval = datetime.datetime.now() - durationEval

        #Erstellt Übersicht über die Performance des aktuellen Modells
        modelOverviews.append(generateOverview(modell, evaluationScores, anzTrainData, anzEvalData, str(durationTrain), str(durationEval), step, run, trainIndexes, devIndexes, evalIndexes))
        #Exportiert Übersicht über die Performance aller bisherigen Modelle,
        # um einen Überblick über den Fortschritt zu bekommen
        exportCurrentStatusModellOverview(modelOverviews, "currentStatus")

print("---------------- Finished ---------------")
#Exportiert Übersicht über die Performance aller Modelle
exportFinishedModellOverview(modelOverviews, len(cvs), nrDiffTrainDataLen)
