from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm

from utils import (
    tokenize,
    tokenize_fn,
    convert_ner_data_format,
    word2features,
    sent2features,
    sent2labels,
    sent2tokens,
    bio_classification_report,
)

import re
import pycrfsuite

def train_intent_entity_model(file_path: str):
    """
    file_path: dataset file path(rasa nlu.md format)
    """
    current_intent_focus = ""

    # intent dataset definition
    X = []
    y = []

    entity_dataset = []

    with open(file_path, encoding="utf-8") as dataFile:
        for line in tqdm(dataFile.readlines(), desc="collecting intent entity data..."):
            if "## intent" in line:
                current_intent_focus = line.split(":")[1].strip()
            elif line.strip() == "":
                current_intent_focus = ""
            else:
                if current_intent_focus != "":
                    utterance = line[2:].strip()

                    entity_dataset.append(
                        convert_ner_data_format(utterance, tokenize_fn)
                    )

                    X.append(utterance)
                    y.append(current_intent_focus)

    # split intent dataset by train & val
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=88)

    svc = make_pipeline(CountVectorizer(analyzer="word", tokenizer=tokenize), SVC())
    print("intent model training(with SVC)")
    svc.fit(X_train, y_train)
    print("intent model train done, validation reporting")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    # split entity dataset by train & val
    X_train, X_test, y_train, y_test = train_test_split(
        [sent2features(s) for s in entity_dataset],
        [sent2labels(s) for s in entity_dataset],
        random_state=88,
    )

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": 1.0,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
            # minimum frequency
            #'feature.minfreq': 5
        }
    )

    print("entity model training(with CRF)")
    trainer.train("morphine-entity-model.crfsuite")
    print("entity model train done, validation reporting")
    tagger = pycrfsuite.Tagger()
    tagger.open("morphine-entity-model.crfsuite")

    y_pred = []
    for test_feature in X_test:
        y_pred.append(tagger.tag(test_feature))

    print(bio_classification_report(y_test, y_pred))
