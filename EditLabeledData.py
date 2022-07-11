# Bearbeitet doccanos CV-weisen Datensatz und passt Label an, um Einheitlichkeit zu gewährleisten.
# Auf Divided Skills Labelschema ausgelegt.
# Die ausgegebene Datei kann mit "JsonToSpaCy_Edited.ipynb" in spacy-Format umgewandelt werden.


# doccano.json einlesen
import json
import spacy
from spacy.tokens import DocBin

# ----------------------------------
# ------------ Options: ------------
# ----------------------------------

# sortiert Label anhand der shouldBe-Listen wenn nötig um
doRearrange = True
# exportiert die entstandenen Daten als jsonl-Datei
doExport = False
# zeigt im searchText-Bereich auch den Kontext an, in dem die gesuchten Begriffe stehen
showContext = False
# printet Labelanzahlen 
printCounter = False
# maximale Anzahl von Wörtern um die Spangrenzen (Labelgrenzen) verschoben werden können
# Wird benötigt, um Fehler zu vermeiden
maxCounter = 30

# Nummer des Datensatzes, der bearbeitet werden soll
modelNumber = "3"    

# muss komplett in kleinbuchstaben sein
searchTexts = [
        "TextToSearch",
    ] 


# ------------------------------------
# ---- Listen zur Labelanpassung: ----
# ------------------------------------

shouldBeSkill = [
    'alv', 'applikation management', 'berichte', 'co', 
    'organisationsmanagement', 'reporting', 'report', 'reports', 'abrechnungssteuerung', 
    'aufbauorganisation', 'datenmodellierung', 'golive support', 
    'qualitätssicherung', 'qualitätsqueck', 'prototyping', 'servicemanagement', 
    'opportunity- und vertragsmanagement', 'veranstaltungsmanagement', 'testmanagement', 
    'testkonzepterstellung', 'testkonzeption', 'testmanagement', 'abnahmetests', 
    'aufwandsschätzung', 'aufwandsschätzungen', 'abrechnungsvergleich', 
    'anforderungsanalyse', 'anforderungserhebung', 'business process integration', 
    'finanzreporting', 'geschäftspartnerkonsolidierung', 'personalabrechnungsverfahren', 
    'ressourcenplanung', 'controlling', 'reklamationsmanagement', 'besuchsmanagement',
    'geschäftsprozessanalyse', 'anforderungsaufnahme', 'anwendungsentwicklung',
    'datenmigration', 'vertriebsprozesse','go-live-support', 'planungs–systems',
    'sap-systeme', 'datenbankdesign', 'powershell-skripting', 'powershell-/json-skripting',
    'proof-of-concept', 'arbeitsprozessanalyse', 'fachkonzept', 'produktivsetzung',
    'schnittstellen', 'userstories', 'fehleranalyse', 'project \nmanagement', 'sepa-umstellung',
    'sepa-überweisung', 'einführungsprojekt', 'exchange', 'feinkonzept', 'release dokumentation',
    'vertriebsplanung', 'kostenplanung', 'kosten-planungslösung', 'besuchsplanung und -durchführung',
    'planungsfunktionen', 'migrationsplan', 'migrationsplane', 'rolloutplanung',
    'test-planung, -durchführung, -dokumentation', 'technische konzepterstellung', 
    'reporterstellung', 'konzepterstellung', 'testunterstützung', 'anwendungsunterstützung',
    'technischen 	vertriebsunterstützung', 'entwicklungsunterstützung', 'programmierunterstützung',
    'backups', 'program management', "software auswahl","software-updates",
    "softwareanalysen","softwareauswahl","softwaretests","softwareupdates",
    "ticketaufbereitung","ticketbearbeitung","applikationsentwicklung","auditbegleitung",
    "bu übergabe","bu übergaben","data warehouse design","datenaustausch","entwicklungsbegleitung",
    "going live","kostenstellenverantwortung","lohnkonten- und stammdatenmigration","lösungsdesign",
    "lösungs-design","parallelabrechnung","patch-level upgrades","patch-level upgrades",
    "roll out management","rolloutsteuerung","sap bw plattform migration","sw- und produktentwicklungen",
    "sales-reporting","stabilisierungsmaßnahmen","strategieberatung","strategieempfehlungen",
    "transitionmangement","wissenskonsolidierung","wissenstransfer","workflowimplementierung",
    "multidimensionalen und relationalen analyse- und berichtsstrukturen","lieferantenbetreuung",
    "produktivexport","produktivmigration","produktivbetreuung","produktivbegleitung",
    "produktivbetriebe","programmleitung", 'produktivbetriebes', 'active directory',
    'grobkonzepten', 'sap bw', 'performance optimierung'
]

shouldBeSkillIfContains = [
    "project", "projekt", "prozess", "process", "system", "level support","level-support"
]

shouldBeActivity = [
    'administration', 'auswertungen', 'analyse', 'test', 'testing', 'agile testing', 'design', 
    'simulation','betreuung', 'hosting', 'umsetzung', 'erweiterungen', 'support prozesses', 
    'inbetriebnahme', 'konsolidierung', 'vertrieb', 'vertriebs', 'vorstudie', 'programmierkenntnisse', 
    'verschlüsselung', 'consulting', 'planung', 'planungen', 'strategische planung',
    'operative und strategische planung', 'einplanung', 'geplant', "planen", "lncentive planung",
    "softwareentwicklung", 'ablösung', 'anbindung', 'dashboarding', 'recoverys', 'aufbau',
    'rollierende planung', 'training', 'untersuchungen', 'software entwicklung', 'enhancements',
    'it strategie', 'it strategien', 'it-strategien', 'monitorings'
]

shouldBeBranche = [
    "öffentliche verwaltung"
]


shouldHaveNoLabel = [
    'datenqualität', 'datenqualität check', 'portal', 'programmiersprachen', 'prüfzeugnis', 
    'genehmigung', 'tickets', ')', 'project', 'projektdokumente', 'projektdokumenten',
    'prozessfehlern', 'dokumenten', 'briefschreibung','entwicklungsumfeld','erfahrungsaustausch',
    'führungserfahrung', 'einbeziehung'
]


# -----------------------------------
# ----------- Funktionen: -----------
# -----------------------------------

def importData(modelNumber):
    cvList = []
    with open("./corpus/Modell_{}/data.jsonl".format(modelNumber), 'r') as f:
        for cv in f:
            currentCV = json.loads(cv)
            cvList.append(currentCV)   
    return cvList
  

def exportData(dicts, modelNumber, cvDicts):
    try:
        for cv in cvDicts:
            spacy.training.offsets_to_biluo_tags(nlp.make_doc(cv['text']), cv['label'])
        fileName = './corpus/Modell_{}/editedData.jsonl'.format(modelNumber)
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(dicts, f, ensure_ascii=False)
    except:
        print("ERROR WHILE EXPORTING")
        

"""
Benötigt um Span-/Labelgrenzen anzupassen, sodass sie den spaCy-Standards entsprechen.
Wird das nicht gemacht, kann es später zu Fehlern kommen.
Möglicherweise fallen hierdurch Label weg. 
Bspw. kann "Java-Programmierung" in spaCy nur als ganzes gelabelt werden.
"""
def adjustSpanBoundaries(start, end, label, maxCounter):
    span = doc.char_span(start, end, label=label)
    # wird True gesetzt, wenn es trotz verschieben Labelgrenzen zu Fehlern kommt
    boundaryError = False

    newEnd = end
    counter = 0;
    #verschiebt die Grenze nach hinten, solange der span ungültig ist
    while span == None and not counter > maxCounter:
        counter = counter + 1
        newEnd = newEnd + 1
        span = doc.char_span(start, newEnd, label=label)

    # geht hier rein, falls der span immer noch ungültig ist
    if span == None:
        newEnd = end
        counter = 0;
        # setzt die Spangrenze zurück und verschiebt sie diesmal nach vorne, solange der span ungültig ist
        while span == None and not counter > maxCounter:
            counter = counter + 1
            start = start - 1
            span = doc.char_span(start, end, label=label)
        # gibt Fehler zurück, sollte der span immer noch ungültig sein
        if span == None:
            boundaryError = True

    return boundaryError, span, start, newEnd 
                
"""
Passt Label basierend auf den entsprechenden Listen an, um Einheitlichkeit zu gewährleisten.
"""
def rearrangeLabels(span, spanStart, spanEnd, label):
    spanText = span.text.lower()
    deleteThisLabel = False
    if spanText in shouldHaveNoLabel:
        #print("---- DELETED LABEL: ", span.text, " ----") 
        deleteThisLabel = True
    elif label == "Tätigkeit" and spanText in shouldBeSkill:
        #print("---- CHANGED LABEL TO Skill: ", span.text, " ----") 
        span = doc.char_span(spanStart, spanEnd, label="Skill")
        label = "Skill"                 
    elif label == "Skill" and spanText in shouldBeActivity:
        #print("---- CHANGED LABEL TO Tätigkeit: ", span.text, " ----") 
        span = doc.char_span(spanStart, spanEnd, label="Tätigkeit")
        label = "Tätigkeit"
    elif not label == "Branche" and spanText in shouldBeBranche:
        #print("---- CHANGED LABEL TO Branche: ", span.text, " ----") 
        span = doc.char_span(spanStart, spanEnd, label="Branche")
        label = "Branche"
    elif label == "Tätigkeit":
        for element in (element for element in shouldBeSkillIfContains if element in spanText):
            #print("---- CHANGED LABEL TO Skill: ", span.text, " ----") 
            span = doc.char_span(spanStart, spanEnd, label="Skill")
            label = "Skill"  
            
    return span, label, deleteThisLabel
                        
"""
Prüft, ob der aktuelle Span sich mit keinem anderen überlappt, da spaCy damit nicht umgehen kann.
Gibt einen entsprechenden boolean zurück.
"""      
def checkForOverlap(spanStart, spanEnd, labels):
    foundOverlap = False
    for otherStart, otherEnd, _ in labels:
        if otherStart == spanStart and otherEnd == spanEnd:
            foundOverlap = True
        elif spanStart < otherEnd and spanEnd > otherStart:
            foundOverlap = True
    return foundOverlap

"""
Printet alle gefundenen Spans, in denen einer der übergebenen searchTexts vorkommt
"""
def searchForTokens(searchTexts, span, allreadyFound, showContext):
    spanText = span.text.lower()
    for searchText in searchTexts: 
        if searchText in spanText and not ("{} - LABEL: {}".format(spanText, span.label_) in allreadyFound):
            allreadyFound.append("{} - LABEL: {}".format(spanText, span.label_))
            if showContext:
                print("---- INCLUDES", searchText,": ", span.text, "; LABEL: ", label, " ----")
                contextLen = 1
                if (span.start - contextLen >= 0):
                    span.start = span.start - contextLen
                else:
                    span.start = 0                           
                if (span.end + contextLen <= len(doc)):
                    span.end = span.end + contextLen
                else:
                    span.end = len(doc)
                print("    ---- CONTEXT: ", span.text, " ----")
                span.start = span.start + contextLen
                span.end = span.end - contextLen  
            else:
                print("---- INCLUDES", searchText,": ", span.text, "\n       LABEL: ", label, " ----")   


# -----------------------------------
# -------------- Main: --------------
# -----------------------------------

# leeres neues Modell erstellen
nlp = spacy.blank("de")
cvList = importData(modelNumber)

# Arrays mit allen vergebenen Labeln
activities = []
skills = []
rollen = []
branchen = []

allreadyFound = []
# zählt wie oft die einzelnen Label vorkommen
labelCounter = []

# speichert die CVs mit den bearbeiteten Labels
cvDicts = []
dicts = {"CVs": cvDicts}

# geht jedes Datenelement (hier CV) durch und überprüft alle Label
for cv in cvList:
    try:
        text = cv['data']
    except:
        text = cv['text']
    annotations = cv['label']
    
    doc = nlp(text)
    
    counterSkills = 0
    counterActivities = 0
    counterRolle = 0
    counterBranche = 0
    
    # packt angepasste Daten in ein dictionary, das aufgebaut ist, wie die zu exportierende json-Datei
    labels = []
    cvDict = {
      "text": text,
      "label": labels
    }
    
    # geht alle Label durch und passt sie wenn nötig an
    for annotation in annotations:
        label = annotation[2]
        boundaryError, span, spanStart, spanEnd = adjustSpanBoundaries(int(annotation[0]), int(annotation[1]), label, maxCounter)
           
        # bei boundaryError wird das Label übersprungen und fällt damit weg
        if not boundaryError:
            if doRearrange:
                span, label, deleteThisLabel = rearrangeLabels(span, spanStart, spanEnd, label)
                if deleteThisLabel:
                    continue
                      
            searchForTokens(searchTexts, span, allreadyFound, showContext)
            
            # sortiert sich überlappende Label aus  
            foundOverlap = checkForOverlap(spanStart, spanEnd, labels)
                
            # speichert Label, sofern es sich mit keinem anderen überschneidet
            if not foundOverlap:
                labels.append([spanStart, spanEnd, label])
                # zählt counter hoch
                if label == "Tätigkeit":
                    activities.append(span.text)
                    counterActivities = counterActivities + 1
                elif label == "Skill":
                    skills.append(span.text)  
                    counterSkills = counterSkills + 1
                elif label == "Branche": 
                    branchen.append(span.text)
                    counterBranche = counterBranche + 1
                elif label == "Rolle": 
                    rollen.append(span.text)
                    counterRolle = counterRolle + 1
                
    cvDicts.append(cvDict)
    labelCounter.append({
        "Tätigkeiten": counterActivities, 
        "Skills": counterSkills, 
        "Branchen": counterBranche, 
        "Rollen": counterRolle, 
        "Gesamt": (counterActivities+counterSkills+counterBranche+counterRolle)
    })
    
if doExport:
    exportData(dicts, modelNumber, cvDicts)
