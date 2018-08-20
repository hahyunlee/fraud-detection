import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, classification_report


def rf_run(df):
    X = df[['body_length', 'event_length', 'previous_payouts', 'channels', 'delivery_method',
            'name_length',
            # 'num_order',
            # 'num_payouts',
            # 'sale_duration',
            'user_age', 'user_type']]
    y = df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # test_y_pred = logit.predict(X_test)
    # test_y_pred_proba = logit.predict_proba(X_test)

    # print('accuracy: ',accuracy_score(y_test, test_y_pred))
    # print('precision: ',precision_score(y_test, test_y_pred))
    # print('recall: ',recall_score(y_test, test_y_pred))
    # print('f1: ',f1_score(y_test, test_y_pred))
    # print('auc: ',roc_auc_score(y_test, test_y_pred))

    random_forest = RandomForestClassifier(n_estimators=100)

    model_rf = random_forest.fit(X_train, y_train)
    print ("Model score: ", model_rf.score(X_test, y_test))
    print ("ROC AUC score: ", roc_auc_score(y_test, model_rf.predict(X_test)))
    print('precision: ', precision_score(y_test, model_rf.predict(X_test)))
    print('recall: ', recall_score(y_test, model_rf.predict(X_test)))
    print('f1: ', f1_score(y_test, model_rf.predict(X_test)))

    return model_rf
