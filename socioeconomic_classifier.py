import numpy
import sys
import pandas
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import minmax_scale,scale,StandardScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn import metrics,tree,svm
import math
import matplotlib.pyplot as plt
from io import BytesIO  
from IPython.display import Image  
import pydotplus
from bidi import algorithm as bidialg
import warnings
import random
from tensorflow import keras, math
from keras import backend,models,layers,optimizers,losses,callbacks,saving
from scikeras.wrappers import KerasClassifier
from sec_descriptive_stats import explore_data_stats
from sec_optimizers import optimize_decision_tree,optimize_ordinal_regression,optimize_support_vector_machine,\
     optimize_random_forest,optimize_neural_network 
import datetime
from numpy import unravel_index
import joblib
import shutil
import os


# Utilities

SEI_DIRECTORY = "SEC"
DTC_FILENAME = "SEC\\dtc_model.joblib"
OMC_FILENAME = "SEC\\omc_model.joblib"
SVC_FILENAME = "SEC\\svc_model.joblib"
RFC_FILENAME = "SEC\\rfc_model.joblib"
NNC_FILENAME = "SEC\\nnc_model.tf"

party_lists = {
    "AMT": {"title": "אמת", "color": "#FA8072"},
    "B": {"title": "ב", "color": "#87CEEB"},
    "G": {"title": "ג", "color": "#A020F0"},
    "D": {"title": "ד", "color": "#FFA500"},
    "WM": {"title": "ום", "color": "#FF0000"},
    "T": {"title": "ט", "color": "#008B8B"},
    "KN": {"title": "כן", "color": "#808000"},
    "L": {"title": "ל", "color": "#B03060"},
    "MHL": {"title": "מחל", "color": "#0000FF"},
    "MRC": {"title": "מרצ", "color": "#00FF7F"},
    "AM": {"title": "עם", "color": "#228B22"},
    "PH": {"title": "פה", "color": "#F4A460"},
    "SS": {"title": "שס", "color": "#DAA520"},
    "others": {"title": "אחרות", "color": "#BEBEBE"}
    }

basic_feature_set = ["prop_voters","AMT","B","G","D","WM","T","KN","L","MHL","MRC","AM","PH","SS","others"]
extended_feature_set = ["prop_voters","AMT","B","G","D","WM","T","KN","L","MHL","MRC","AM","PH","SS","others","is_coop","is_jew","is_pal"]

numpy.set_printoptions(precision=3,threshold=sys.maxsize)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.float_format', '{:7,.3f}'.format)
pandas.set_option('display.width', 120)

def perform_safe_exit(output_file, stdout_backup, to_exit=True):
    output_file.close()
    sys.stdout = stdout_backup
    if to_exit:
        exit()

# Preprocessing data
def preprocess_data(df):
    kalpi_data = df[["id","place_name","jew","pal","coop","cluster"]].copy()
    kalpi_data.insert(len(kalpi_data.columns),"is_coop", list(map(lambda s: 1.0 if s==1 else 0.0, df["coop"])))
    kalpi_data.insert(len(kalpi_data.columns),"is_jew", list(map(lambda s: 1.0 if s==1 else 0.0, df["jew"])))
    kalpi_data.insert(len(kalpi_data.columns),"is_pal", list(map(lambda s: 1.0 if s==1 else 0.0, df["pal"])))
    #kalpi_data["prop_voters"] = df["legal"]/df["elligible"]
    kalpi_data.insert(len(kalpi_data.columns),"prop_voters", list(df["legal"]/df["elligible"]))
    for pl in party_lists.keys():
        #kalpi_data[pl] = df[pl]/df["legal"]
        kalpi_data.insert(len(kalpi_data.columns),pl,list(df[pl]/df["legal"]))
    return kalpi_data

# Preparing data for classification
def prepare_data_for_classification(kalpi_data):
    clustered_kalpi_data = kalpi_data[kalpi_data["cluster"].notna()]
    data_for_classification = clustered_kalpi_data[["id","cluster"]+extended_feature_set]
    return data_for_classification

# Printing information about train and test data
def print_train_test_info(X_train, X_test):
    train_index_distribution = X_train["cluster"].value_counts().sort_index()
    jewish_train_distribution = X_train[(X_train["is_jew"]==1) & (X_train["is_pal"]==0)]["cluster"].value_counts().sort_index()
    arab_train_distribution = X_train[(X_train["is_jew"]==0) & (X_train["is_pal"]==1)]["cluster"].value_counts().sort_index()
    mixed_train_distribution = X_train[(X_train["is_jew"]==1) & (X_train["is_pal"]==1)]["cluster"].value_counts().sort_index()
    druze_train_distribution = X_train[(X_train["is_jew"]==0) & (X_train["is_pal"]==0)]["cluster"].value_counts().sort_index()
    test_index_distribution = X_test["cluster"].value_counts().sort_index()
    jewish_test_distribution = X_test[(X_test["is_jew"]==1) & (X_test["is_pal"]==0)]["cluster"].value_counts().sort_index()
    arab_test_distribution = X_test[(X_test["is_jew"]==0) & (X_test["is_pal"]==1)]["cluster"].value_counts().sort_index()
    mixed_test_distribution = X_test[(X_test["is_jew"]==1) & (X_test["is_pal"]==1)]["cluster"].value_counts().sort_index()
    druze_test_distribution = X_test[(X_test["is_jew"]==0) & (X_test["is_pal"]==0)]["cluster"].value_counts().sort_index()
    
    print("Train SEI distribution:")
    print("SEI","kalpiot","Jewish","Arab","Mixed","Druze", sep="\t")
    for i in range(10):
        j = jewish_train_distribution[i] if i in jewish_train_distribution.index else 0
        a = arab_train_distribution[i] if i in arab_train_distribution.index else 0
        m = mixed_train_distribution[i] if i in mixed_train_distribution.index else 0
        d = druze_train_distribution[i] if i in druze_train_distribution.index else 0
        print (int(train_index_distribution.index[i])+1, train_index_distribution.values[i], j, a, m, d, sep="\t")
    print("\nTest SEI distribution:")
    print("SEI","kalpiot","Jewish","Arab","Mixed","Druze", sep="\t")
    for i in range(10):
        j = jewish_test_distribution[i] if i in jewish_test_distribution.index else 0
        a = arab_test_distribution[i] if i in arab_test_distribution.index else 0
        m = mixed_test_distribution[i] if i in mixed_test_distribution.index else 0
        d = druze_test_distribution[i] if i in druze_test_distribution.index else 0
        print (int(test_index_distribution.index[i])+1, test_index_distribution.values[i], j, a, m, d, sep="\t")

# Creating sample weights
def create_sample_weights (y_train):
    # Counting the instances for each SEI in the train data
    values = y_train.reshape(-1).astype(int)
    sorted_values = numpy.sort(values)
    indices, counts = numpy.unique(sorted_values, return_counts=True)
    class_weights = [0.0]*10
    print("Class weights:")
    # Iterating over SEIs
    for i in range(len(indices)):
        # SEI potential weight is the number miultiplying by which would equalize the count as the average of the counts of the flanking SEIs
        prv = counts[i-1] if i>0 else 0
        nxt = counts[i+1] if i<9 else 0
        pw = (prv+nxt)/(2*counts[i])
        # The potential weight is assigned only if it is greater than 1. Otherwise it's 1
        class_weights[i] = pw if pw>1 else 1.0
        print(i+1, class_weights[i], sep="\t")
    sample_weights = numpy.array(list(map(lambda s: class_weights[s], values)))
    # Sample weights are returned
    return sample_weights

# Evaluating test predictions of a classifier
def evaluate_predictions(X_test_kalpiot, y_test, y_pred, classifier_name):

    print("\n\nEvaluating predictions of", classifier_name)
    acc = metrics.accuracy_score(y_test, y_pred)

    print("\nAccuracy:", round(acc,3))

    near_acc_array = numpy.array(list(map(lambda s,t: 1 if abs(s-t)<=1 else 0, y_test, y_pred)))
    near_acc = numpy.mean(near_acc_array)
    print("Near-accuracy (diff<=1):", round(near_acc, 3))

    print("Accuracy by sector:")
    jk = X_test_kalpiot["is_jew"].astype(bool) & ~(X_test_kalpiot["is_pal"].astype(bool))
    ak = ~(X_test_kalpiot["is_jew"].astype(bool)) & X_test_kalpiot["is_pal"].astype(bool)
    mk = X_test_kalpiot["is_jew"].astype(bool) & X_test_kalpiot["is_pal"].astype(bool)
    dk = ~(X_test_kalpiot["is_jew"].astype(bool)) & ~(X_test_kalpiot["is_pal"].astype(bool))
    print("Jewish:", round(metrics.accuracy_score(y_test[jk],y_pred[jk]),3))
    print("Arab:", round(metrics.accuracy_score(y_test[ak],y_pred[ak]),3))
    print("Mixed:", round(metrics.accuracy_score(y_test[mk],y_pred[mk]),3))
    print("Druze:", round(metrics.accuracy_score(y_test[dk],y_pred[dk]),3))
    
    macro_f1 = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0.0)
    print("SEI-average ('macro-') F1 score:", round(macro_f1, 3))

    precision_by_SEI = metrics.precision_score(y_test, y_pred, average=None, zero_division=numpy.nan)
    print("\nPrecision by SEI:")
    print(precision_by_SEI)
          
    recall_by_SEI = metrics.recall_score(y_test, y_pred, average=None, zero_division=numpy.nan)
    print("\nRecall by SEI:")
    print(recall_by_SEI)

    print("\nConfusion matrix:")
    print(metrics.confusion_matrix(y_test,y_pred,normalize='true'))

    custom_macro_recall = numpy.prod(recall_by_SEI+0.1) ** 0.1

    custom_score = near_acc * custom_macro_recall
    print("\nCustomized classifier score (near-accuracy * customized cross-SEI recall product):", custom_score)

    return custom_score
    
   
# Flow
def run_classification (input_file, output_file, explore_data=True, load_models=False, save_models=False, create_guesser=True):
    # Set up output file and retain shell output to be resumed later
    opf = open(output_file, mode='w', encoding="utf-8")
    stdout_backup = sys.stdout
    sys.stdout = opf
    os.makedirs(SEI_DIRECTORY,exist_ok=True)
    # Reading data
    try:
        entire_data = pandas.read_csv(input_file, sep=",")
    except:
        print ("Problem using data file", input_file, ". Exiting.")
        perform_safe_exit(opf, stdout_backup)

    # Preprocessing data
    try:
        # Getting data as proportions rather than absolute numbers
        kalpi_data = preprocess_data(entire_data)
        print("Preprocessed data")

        # Exploring the data and creating graphs
        if explore_data:
            explore_data_stats(opf, kalpi_data, party_lists, True)

        # Preparing data for classification
        data_for_classification = prepare_data_for_classification(kalpi_data)
        print("Prepared data for classification")

        # Obtaining SE-indices for classification
        indices = numpy.array(data_for_classification["cluster"]).reshape(-1,1)
        print("Obtained SEIs")

        # Splitting data to train and test
        rn = int(datetime.datetime.now().timestamp())%1000
        #X_train_kalpiot, X_test_kalpiot, y_train, y_test = train_test_split(data_for_classification, indices, test_size = 500, random_state = rn)
        X_train_kalpiot, X_test_kalpiot, y_train, y_test = train_test_split(data_for_classification, indices, test_size = 0.2, random_state = 156)
        print("Split train and test")

        # Printing information about train and test data
        print_train_test_info(X_train_kalpiot, X_test_kalpiot)

        # The actual data used for training and testing should contain the features only. These are unscaled (non-standardized) data
        X_train_unscaled_extended = X_train_kalpiot[extended_feature_set]
        X_test_unscaled_extended = X_test_kalpiot[extended_feature_set]
        X_train_unscaled_basic = X_train_kalpiot[basic_feature_set]
        X_test_unscaled_basic = X_test_kalpiot[basic_feature_set]
        print("Extracted actual data for ML processes")

        # Scaling the train data
        X_train_means = X_train_unscaled_basic.mean()
        X_train_stds = X_train_unscaled_basic.std()

        X_train_data = scale(X_train_unscaled_basic, axis=0)
        X_train_basic = pandas.DataFrame(X_train_data, columns=basic_feature_set)

        # Scaling the test data
        X_test_basic = pandas.DataFrame()
        for bf in basic_feature_set:
            X_test_basic[bf] = (X_test_unscaled_basic[bf]-X_train_means[bf])/X_train_stds[bf]

        print("Scaled the data")
        
        # Creating extended version of scaled train and test
        X_train_extended = X_train_basic
        X_test_extended = X_test_basic
        for bf in ["is_coop","is_jew","is_pal"]:
            X_train_extended.insert(len(X_train_extended.columns), bf, X_train_unscaled_extended[bf].to_numpy())
            X_test_extended.insert(len(X_test_extended.columns), bf, X_test_unscaled_extended[bf].to_numpy())
                
        # Creating sample weights
        train_sample_weights = create_sample_weights(y_train)
        print("Created sample weights")
    except:
        print ("Problem preprocessing data. Exiting")
        perform_safe_exit(opf, stdout_backup)

    # Creating optimal decision tree classifier
    loaded = False
    if load_models:
        try:
            dtc = joblib.load(DTC_FILENAME)
            print("\nDecision tree model loaded successfully. Skipping training")
            loaded = True
        except:
            print("Failed to load decision tree model. Training.")
    if not loaded:
        try:
            print("\nOptimizing decision tree")
            dtc = optimize_decision_tree (X_train_unscaled_extended, y_train, train_sample_weights, True)
        except:
            print ("Problem with decision tree classifier. Exiting")
            perform_safe_exit(opf, stdout_backup)
    if save_models:
        try:
            joblib.dump(dtc, DTC_FILENAME)
            print("Decision tree model saved.")
        except:
            print("Failed to save decision tree model.")

    # Creating optimal ordinal regression classifier
    # OrderedModel cannot be loaded from file as it needs the data as part of construction
    #loaded = False
    #if load_models:
    #    try:
    #        omc = joblib.load(OMC_FILENAME)
    #        print("Ordinal logistic regression model loaded successfully. Skipping training")
    #        loaded = True
    #    except:
    #        print("Failed to load ordinal logistic regression model. Training.")
    #if not loaded:
    try:
        print("\nOptimizing ordinal logistic regression")
        omc, dst, mthd = optimize_ordinal_regression (X_train_extended, y_train, train_sample_weights, True) 
    except:
        print ("Problem with ordered model classifier. Exiting")
        perform_safe_exit(opf, stdout_backup)
    #if save_models:
    #    try:
    #        joblib.dump(omc, OMC_FILENAME)
    #        print("Ordinal logistic regression model saved.")
    #    except:
    #        print("Failed to save ordinal logistic regression model.")

    # Creating optimal support vector classifier
    loaded = False
    if load_models:
        try:
            svc = joblib.load(SVC_FILENAME)
            print("\nSupport vector machine model loaded successfully. Skipping training")
            loaded = True
        except:
            print("Failed to load support vector machine model. Training.")
    if not loaded:
        try:
            print("\nOptimizing support vector machine")
            svc = optimize_support_vector_machine (X_train_unscaled_extended, y_train, train_sample_weights)
        except:
            print ("Problem with C-supported vector classifier. Exiting")
            perform_safe_exit(opf, stdout_backup)
    if save_models:
        try:
            # SVC model is large, so compressing it.
            joblib.dump(svc, SVC_FILENAME, 2)
            print("Support vector machine model saved.")
        except:
            print("Failed to save support vector machine model.")
        
    # Creating random forest classifier
    loaded = False
    if load_models:
        try:
            rfc = joblib.load(RFC_FILENAME)
            print("\nRandom forest model loaded successfully. Skipping training")
            loaded = True
        except:
            print("Failed to load random forest model. Training.")
    if not loaded:
        try:
            print("\nOptimizing random forest")
            rfc = optimize_random_forest (X_train_unscaled_extended, y_train, train_sample_weights, True)
        except:
            print ("Problem with random forest classifier. Exiting")
            perform_safe_exit(opf, stdout_backup)
    if save_models:
        try:
            # Random forest model is gigantic, so compressing it substantially.
            joblib.dump(rfc, RFC_FILENAME, 5)
            print("Random forest model saved.")
        except:
            print("Failed to save random forest model.")
    
    # Creating neural network classifier
    loaded = False
    if load_models:
        try:
            nnc = models.load_model(NNC_FILENAME)
            print("\nNeural network model loaded successfully. Skipping training")
            loaded = True
        except:
            print("Failed to load neural network model. Training.")
    if not loaded:
        try:
            print ("\nOptimizing neural network")
            nnc = optimize_neural_network (X_train_extended, y_train, train_sample_weights)
        except:
            print ("Problem with neural network classifier. Exiting")
            perform_safe_exit(opf, stdout_backup)
    if save_models:
        try: 
            nnc.save(NNC_FILENAME, True, "tf")
            print("Neural network model saved.")
        except:
            print("Failed to save neural network model.")


    # Making predictions over test data and evaluating classifiers
    try:
        dtc_y_pred = dtc.predict(X_test_unscaled_extended)
        dtc_score = evaluate_predictions(X_test_kalpiot, y_test, dtc_y_pred, "Decision tree")
        omc_y_pred = numpy.array([numpy.argmax(x) for x in omc.model.predict(omc.params, X_test_extended)])
        omc_score = evaluate_predictions(X_test_kalpiot, y_test, omc_y_pred, "Ordinal logistic regression")
        svc_y_pred = svc.predict(X_test_unscaled_extended)
        svc_score = evaluate_predictions(X_test_kalpiot, y_test, svc_y_pred, "Support vector machine")
        rfc_y_pred = rfc.predict(X_test_unscaled_extended)
        rfc_score = evaluate_predictions(X_test_kalpiot, y_test, rfc_y_pred, "Random forest")
        nnc_y_pred = numpy.array([numpy.argmax(x) for x in nnc.predict(X_test_extended, verbose=0)])
        nnc_score = evaluate_predictions(X_test_kalpiot, y_test, nnc_y_pred, "Neural network")
    except:
        print ("Problem evaluating classifiers. Exiting")
        perform_safe_exit(opf, stdout_backup)

    try:

        # Weighted-score meta classifier:
        classifier_scores = numpy.array([dtc_score,omc_score,svc_score,rfc_score,nnc_score])
        mean_score = numpy.mean(classifier_scores)
        classifier_weights = classifier_scores/mean_score
        test_weights = numpy.tile(classifier_weights, (len(y_test),1))
        test_preds = numpy.transpose(numpy.array([dtc_y_pred,omc_y_pred,svc_y_pred,rfc_y_pred,nnc_y_pred]))
        weighted_preds = numpy.round(numpy.mean(test_weights * test_preds, axis=1)).astype(int)
        wsm_score = evaluate_predictions(X_test_kalpiot, y_test, weighted_preds, "Weighted score meta classifier") 
    

        # Preparing data for meta_classifier
        print("Preparing train and test data for meta-SVC...")
        dtc_train_pred = dtc.predict(X_train_unscaled_extended)
        omc_train_pred = numpy.array([numpy.argmax(x) for x in omc.model.predict(omc.params, X_train_extended)])
        svc_train_pred = svc.predict(X_train_unscaled_extended)
        rfc_train_pred = rfc.predict(X_train_unscaled_extended)
        nnc_train_pred = numpy.array([numpy.argmax(x) for x in nnc.predict(X_train_extended, verbose=0)])

        meta_X_train = X_train_extended[["is_jew","is_pal","is_coop"]]
        meta_X_train.insert(len(meta_X_train.columns),"dtc",dtc_train_pred)
        meta_X_train.insert(len(meta_X_train.columns),"omc",omc_train_pred)
        meta_X_train.insert(len(meta_X_train.columns),"svc",svc_train_pred)
        meta_X_train.insert(len(meta_X_train.columns),"rfc",rfc_train_pred)
        meta_X_train.insert(len(meta_X_train.columns),"nnc",nnc_train_pred)
    
        meta_X_test = X_test_extended[["is_jew","is_pal","is_coop"]]
        meta_X_test.insert(len(meta_X_test.columns),"dtc",dtc_y_pred)
        meta_X_test.insert(len(meta_X_test.columns),"omc",omc_y_pred)
        meta_X_test.insert(len(meta_X_test.columns),"svc",svc_y_pred)
        meta_X_test.insert(len(meta_X_test.columns),"rfc",rfc_y_pred)
        meta_X_test.insert(len(meta_X_test.columns),"nnc",nnc_y_pred)

        # Creating and optimizing meta_classifier
        print("Optimizing meta-SVC")
        meta_svc = optimize_support_vector_machine (meta_X_train, y_train, train_sample_weights)
        meta_svc_y_pred = meta_svc.predict(meta_X_test)
        meta_svc_score = evaluate_predictions(X_test_kalpiot, y_test, meta_svc_y_pred, "Meta-SVC")
    except:
        print ("Problem with meta classifiers. Exiting")
        perform_safe_exit(opf, stdout_backup)

    # Creating guesser for interactive kalpi SEI guessing
    if create_guesser:
        perform_safe_exit(opf, stdout_backup, False)
        print("Preparing guesser...")

        # Retraining all the models on the entire dataset
        try:
            # Dataset for training
            X_train_unscaled = data_for_classification[extended_feature_set]

            # Standardizing features for ordinal logistic regression and neural network
            X_train_data = scale(X_train_unscaled[basic_feature_set], axis=0)
            X_train_scaled = pandas.DataFrame(X_train_data, columns=basic_feature_set)
            for bf in ["is_coop","is_jew","is_pal"]:
                X_train_scaled.insert(len(X_train_scaled.columns), bf, X_train_unscaled[bf].to_numpy())
            print(X_train_unscaled)

            # Sample weights based on entire dataset
            sample_weights = create_sample_weights(indices)


            # Training models
            dtc.fit(X_train_unscaled, indices, sample_weights)
            
            # (Ordinal logistic regression model needs to be reconstructed using the distr and method parameters obtained via optimization)
            omc = OrderedModel(indices, X_train_scaled, distr=dst)
            omc_res_log = omc.fit(method=mthd,max_iter=500,skip_hessian=True,disp=False)
            
            svc.fit(X_train_unscaled, indices.ravel(), sample_weights)

            rfc.fit(X_train_unscaled, indices.ravel(), sample_weights)
            
            # Callbacks for optimizing neural network
            cb = callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=10, start_from_epoch=20)
            mc = callbacks.ModelCheckpoint("entire_data_nnc.tf", monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_format="tf")
            # Training neural network
            history = nnc.fit(X_train_scaled, indices, epochs=200, sample_weight=sample_weights, validation_split=0.15, verbose=0, callbacks=[cb,mc])
            nnc = models.load_model("entire_data_nnc.tf")
            shutil.rmtree("entire_data_nnc.tf")

            # Obtaining train predictions to create meta SVC
            dtc_train_pred = dtc.predict(X_train_unscaled)
            omc_train_pred = numpy.array([numpy.argmax(x) for x in omc_res_log.model.predict(omc_res_log.params, X_train_scaled)])
            svc_train_pred = svc.predict(X_train_unscaled)
            rfc_train_pred = rfc.predict(X_train_unscaled)
            nnc_train_pred = numpy.array([numpy.argmax(x) for x in nnc.predict(X_train_scaled, verbose=0)])

            meta_X_train = X_train_unscaled[["is_jew","is_pal","is_coop"]]
            meta_X_train.insert(len(meta_X_train.columns),"dtc",dtc_train_pred)
            meta_X_train.insert(len(meta_X_train.columns),"omc",omc_train_pred)
            meta_X_train.insert(len(meta_X_train.columns),"svc",svc_train_pred)
            meta_X_train.insert(len(meta_X_train.columns),"rfc",rfc_train_pred)
            meta_X_train.insert(len(meta_X_train.columns),"nnc",nnc_train_pred)

            # Creating and optimizing meta_classifier
            meta_svc = optimize_support_vector_machine (meta_X_train, indices, sample_weights)

            
        except:
            print("Problem training the models on the entire data. Exiting.")
            exit()

        # Means and Stds for scaling individual kalpi features for neural network
        X_train_means = X_train_unscaled.mean()
        X_train_stds = X_train_unscaled.std()

        # Interacting with the user
        while True:
            # Asking input of kalpi
            print("Enter kalpi id (0 to exit):")
            kalpi_id = int(input())
            if kalpi_id:
                try:
                    # Finding kalpi
                    kalpi = kalpi_data.loc[kalpi_data["id"]==kalpi_id]
                    print("Kalpi data:\n", kalpi)
                    kalpi_unscaled = kalpi[extended_feature_set]
                    kalpi_scaled = pandas.DataFrame()
                    for f in extended_feature_set:
                        if f in ["is_jew","is_pal","is_coop"]:
                            kalpi_scaled.insert(len(kalpi_scaled.columns), f, kalpi_unscaled[f].to_numpy())
                        else:
                            kalpi_scaled.insert(len(kalpi_scaled.columns), f, (kalpi_unscaled[f] - X_train_means[f])/X_train_stds[f])
                    print("SEI predictions:")
                    dtc_kalpi_pred = dtc.predict(kalpi_unscaled)
                    print("Decision tree:", int(dtc_kalpi_pred[0])+1)
                    omc_kalpi_pred = numpy.array([numpy.argmax(x) for x in omc_res_log.model.predict(omc_res_log.params,kalpi_scaled)])
                    print("Ordinal logistic regression:", int(omc_kalpi_pred[0])+1)
                    svc_kalpi_pred = svc.predict(kalpi_unscaled)
                    print("Support vector machine:", int(svc_kalpi_pred[0])+1)
                    rfc_kalpi_pred = rfc.predict(kalpi_unscaled)
                    print("Random forest:", int(rfc_kalpi_pred[0])+1)
                    nnc_kalpi_pred = numpy.array([numpy.argmax(x) for x in nnc.predict(kalpi_scaled, verbose=0)])
                    print("Neural network:", int(nnc_kalpi_pred[0])+1)
                    all_preds = numpy.array([dtc_kalpi_pred,omc_kalpi_pred,svc_kalpi_pred,rfc_kalpi_pred,nnc_kalpi_pred])
                    weighted_kalpi_pred = numpy.round(numpy.mean(classifier_weights * all_preds)).astype(int)
                    print("Weighted score meta classifier:", weighted_kalpi_pred + 1)
                    meta_kalpi_data = kalpi_unscaled[["is_jew","is_pal","is_coop"]]
                    meta_kalpi_data.insert(len(meta_kalpi_data.columns),"dtc",dtc_kalpi_pred)
                    meta_kalpi_data.insert(len(meta_kalpi_data.columns),"omc",omc_kalpi_pred)
                    meta_kalpi_data.insert(len(meta_kalpi_data.columns),"svc",svc_kalpi_pred)
                    meta_kalpi_data.insert(len(meta_kalpi_data.columns),"rfc",rfc_kalpi_pred)
                    meta_kalpi_data.insert(len(meta_kalpi_data.columns),"nnc",nnc_kalpi_pred)
                    meta_svc_kalpi_pred = meta_svc.predict(meta_kalpi_data)
                    print("Meta-SVC:", int(meta_svc_kalpi_pred[0])+1)
                    
                except:
                    print("Kalpi", kalpi_id, "doesn't exist.")
            else:
                print("Exiting.")
                exit()
                    
                            
                                       

      



# Support vector machine

# Random forest

# Neural network

# Ensemble classifier



