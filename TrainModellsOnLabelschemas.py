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

runs = 100          # Anzahl zu trainierender Modelle, pro Labelschema
numberOfSchemas = 4 # Trainiert auf den Labelschemen 1 bis 'numberOfSchemas'. Könnte besser gelöst werden, war aber hier nicht nötig

ratioTrain = 9 / 20 # Verhältnis der Trainingsdatensatzgröße zum Gesamtdatensatz
ratioDev   = 6 / 20 # Verhältnis der Entwicklungsdatensatzgröße zum Gesamtdatensatz
ratioEval  = 5 / 20 # Verhältnis der Evaluationsdatensatzgröße zum Gesamtdatensatz

maxShift = 2     # Maximale Anzahl von Wörtern um die Spangrenzen (Labelgrenzen) verschoben werden können. Wird benötigt, um Fehler zu vermeiden
maxOccurence = 2 # Anzahl wie oft jeder einzelne Datensatz (bspw Trainingsdatenset) sich wiederholen darf 

datasetName = "editedData" # Name der Datei mit dem vollständigen gelabelten Datensatz


# -----------------------------------
# ----------- Funktionen: -----------
# -----------------------------------

"""
Lädt gelabelte Daten
"""
def loadCVs(schemaNumber, datasetName):
    with open('./corpus/Modell_{}/{}.jsonl'.format(schemaNumber, datasetName), 'r', encoding="utf-8") as f:
        for cv in f:
            cvs = json.loads(cv) 
    return cvs['lines'] #['CVs']

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
Konvertiert den übergebenen Datensatz ins spacy-Format und speichert ihn unter dem angegebenen Namen.
"""
def convertForSpacy(dataset, datasetName, schemaNumber, nlp, maxShift):
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
    db.to_disk("./corpus/Modell_{}/{}.spacy".format(schemaNumber, datasetName))

"""
Evaluiert das trainierte Modell
"""
def evaluateModel(modell, dataset):
    data = []
    for cv in evalData:
        spacy.training.offsets_to_biluo_tags(modell.make_doc(cv['text']), cv['label'])
        data.append(Example.from_dict(modell.make_doc(cv['text']), {"entities": cv['label']}))
    scores_model = modell.evaluate(data)
    return {
        "f_score": scores_model["ents_f"],
        "precision": scores_model["ents_p"],
        "recall": scores_model["ents_r"],
        "perLabel": scores_model['ents_per_type']
    }
    
"""
Generiert Übersicht über das Modell und seine Performance
"""
def generateOverview(modell, evaluationScores, trainSize, devSize, evalSize, durationTrain, durationEval, durationModel, run, trainIndexes, devIndexes, evalIndexes):
    return {
        "run": run,
        "trainSize": trainSize,
        "devSize": devSize,
        "evalSize": evalSize,
        "durationTillNow": durationModel,
        "durationTrain": durationTrain,
        "durationEval": durationEval,
        "trainIndexes": str(trainIndexes), 
        "devIndexes": str(devIndexes), 
        "evalIndexes": str(evalIndexes),
        "performance": {
            "f-score": modell.meta["performance"]['ents_f'],
            "precision": modell.meta["performance"]['ents_p'],
            "recall": modell.meta["performance"]['ents_r'],
            "perLabel": modell.meta["performance"]['ents_per_type'],
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
    fileName = './TestModelle/ModellOverview_{}.jsonl'.format(name)
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(overviewDict, f, ensure_ascii=False, indent=3)
            
"""
Exportiert die komplette Übersicht über die Ergebnisse ohne Einrückungen 
"""
def exportFinishedModellOverview(overview, name):
    overviewDict = {"Modelle": overview}
    fileName = './TestModelle/ModellOverview_{}.jsonl'.format(name)
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(overviewDict, f, ensure_ascii=False)
        

# ----------------------------------
# ------------- Start: -------------
# ----------------------------------
    
for i in range(numberOfSchemas):   
    # zählt, wie lange das Programm pro Labelschema braucht
    durationModelStart = datetime.datetime.now()
    schemaNumber = str(i+1)
    
    #leeres Modell importieren
    nlp = spacy.blank("de")
    cvs = loadCVs(schemaNumber, datasetName)
    # Hier werden die Informationen zu allen trainierten und evaluierten Modellen gespeichert. 
    # Wird am Ende als json-Datein exportiert
    modelOverviews = []
    
    datasetLen = len(cvs)
    anzTrainData = (round(datasetLen * ratioTrain)) # Größe des Trainingsdatensatzes
    anzDevData   = (round(datasetLen * ratioDev))   # Größe des Entwicklungsdatensatzes (development data)
    anzEvalData  = (round(datasetLen * ratioEval))  # Größe des Evaluierungsdatensatzes

    # speichert bereits verwendete Datensätze, damit diese nicht nochmal vorkommen
    alreadyUsedDatasetsTrain = []
    alreadyUsedDatasetsDev = []
    alreadyUsedDatasetsEval = []
    alreadyUsedDatasets = []

    # trainiert und evaluiert auf jedem Labelschema-Datensatz runs-viele Modelle
    for run in range(runs):
        print("------------ Run Number:", run,"-------------")
        #erstellt zufälligen Trainings-, Entwicklungs- und Evaluierungsdatensatz
        # geht sicher, dass keine Kombination doppelt verwendet wird
        didNotRunYet = True
        while didNotRunYet or [trainIndexes, devIndexes, evalIndexes] in alreadyUsedDatasets:
            trainData, trainIndexes, alreadyUsedDatasetsTrain = createDataset(anzTrainData, cvs, [], alreadyUsedDatasetsTrain, maxOccurence)
            devData, devIndexes, alreadyUsedDatasetsDev       = createDataset(anzDevData, cvs, trainIndexes, alreadyUsedDatasetsDev, maxOccurence)
            evalData, evalIndexes, alreadyUsedDatasetsEval    = createDataset(anzEvalData, cvs, trainIndexes, alreadyUsedDatasetsEval, maxOccurence)
            didNotRunYet = False

        alreadyUsedDatasets.append([trainIndexes, devIndexes, evalIndexes])

        #wandelt Datensets in spaCy-Format um
        convertForSpacy(trainData, "train", schemaNumber, nlp, maxShift)
        convertForSpacy(devData, "dev", schemaNumber, nlp, maxShift)

        print("----------------- Train -----------------")
        #trainiert und lädt Modell auf ausgewähltem Datenset
        durationTrain = datetime.datetime.now()
        train("./TestModelle/config_{}.cfg".format(schemaNumber), "./TestModelle/TempModell")
        durationTrain = datetime.datetime.now() - durationTrain
        modell = spacy.load("./TestModelle/TempModell/model-best/")

        print("---------------- Evaluate ---------------")
        #evaluiert Modell mit zufälligen Daten
        durationEval = datetime.datetime.now()
        evaluationScores = evaluateModel(modell, evalData)
        durationEval = datetime.datetime.now() - durationEval

        #Erstellt Übersicht über die Performance des aktuellen Modells
        durationModel = datetime.datetime.now() - durationModelStart
        modelOverviews.append(generateOverview(modell, evaluationScores, anzTrainData, anzDevData, anzEvalData, str(durationTrain), str(durationEval), str(durationModel), run, trainIndexes, devIndexes, evalIndexes))
        #Exportiert Übersicht über die Performance aller bisherigen Modelle,
        # um einen Überblick über den Fortschritt zu bekommen
        exportCurrentStatusModellOverview(modelOverviews, "currentStatus")

    print("---------------- Finished ---------------")
    #Exportiert Übersicht über die Performance aller Modelle
    exportFinishedModellOverview(modelOverviews, "Modell_{}".format(schemaNumber))
