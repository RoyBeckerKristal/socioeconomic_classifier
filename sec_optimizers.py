import numpy
import pandas
import tensorflow as tf
from sklearn.preprocessing import minmax_scale,scale,StandardScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn import metrics,tree,svm
import math
import warnings
import random
from tensorflow import keras, math
from keras import backend,models,layers,optimizers,losses,callbacks,saving
from scikeras.wrappers import KerasClassifier
import datetime
from numpy import unravel_index
import shutil


# Optimizing a decision tree classifier
def optimize_decision_tree (X_train, y_train, sample_weights=None, report_features=False):

    # Hyperparameter ranges and grid
    max_depth = numpy.arange(4,12)
    min_samples_split = numpy.arange(2,11,2)
    min_samples_leaf = numpy.arange(1,6)
    dtc_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

    # Creating the DT model
    dtc = DecisionTreeClassifier()

    try:
        # Creating the optimization grid
        dtc_model = GridSearchCV(estimator=dtc, param_grid=dtc_grid, verbose=3)
        # Tuning
        dtc_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        print("Best decision tree classifier achieved with the following hyperparameters:\n", dtc_model.best_params_)
        # Printing feature importance
        if report_features:
            ftr_imp = list(dtc_model.best_estimator_.feature_importances_)
            feature_set = X_train.columns
            tree_feature_importances = [(ftr, round(imp, 3)) for ftr, imp in zip(feature_set,ftr_imp)]
            tree_feature_importances = sorted(tree_feature_importances, key = lambda x: x[1], reverse=True)
            print("Tree feature importances:")
            for f in tree_feature_importances:
                print("Feature:", f[0], ", Importance:", f[1])
        # Returning the best DT variant
        return dtc_model.best_estimator_
    except:
        print("Problem in method 'optimize_decision_tree'")
        return None            

# Optimizing ordinal regression classifier
def optimize_ordinal_regression (X_train, y_train, sample_weights=None, report_features=False):
    # Hyperparameter ranges:
    distributions = ['logit','probit']
    methods = ['nm','bfgs','powell','cg','ncg','minimize']

    # OrderedModel isn't compatible with GridSearchCV, so creating one from scratch, as follows:
    # 1. Creating a shuffled list of indices (0-4) for 5 folds for all samples
    n_samples = X_train.shape[0]
    cv_indices = numpy.zeros(n_samples,dtype=int)
    sample_indices = numpy.arange(n_samples)
    ind_set = numpy.arange(5)
    numpy.put(cv_indices, sample_indices, ind_set)
    numpy.random.shuffle(cv_indices)
    # 2. Creating validation fold-flag arrays to be applied on all samples
    cv_flags = numpy.zeros((n_samples,5),dtype=bool)
    for i in range(5):
        cv_flags[:,i] = cv_indices==i
    # 3. Creating train fold-flag arrays (inverse of the validation fold-flag arrays)
    cv_inv_flags = numpy.zeros((n_samples,5),dtype=bool)
    cv_inv_flags = numpy.invert(cv_flags)
    # 4. Creating train and validation folds using the fold-flag arrays
    Xts = []
    yts = []
    Xvs = []
    yvs = []
    for i in range(5):
        Xt = X_train[cv_inv_flags[:,i]]
        yt = y_train[cv_inv_flags[:,i]]
        Xv = X_train[cv_flags[:,i]]
        yv = y_train[cv_flags[:,i]]
        Xts.append(Xt)
        yts.append(yt)
        Xvs.append(Xv)
        yvs.append(yv)

    # Accuracy values for each distr-method combination
    acc_values = numpy.zeros((len(distributions),len(methods)))

    # Iterating over distr-method combinations

    for d in range(len(distributions)):
        for m in range(len(methods)):
            # Fold accuray values
            accs = numpy.zeros(5)
            # Iterating over folds
            for i in range(5):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Creating model with fold train data and distr value
                    om = OrderedModel(yts[i],Xts[i],distr=distributions[d])
                    # Fitting model with method value
                    res_log = om.fit(method=methods[m],max_iter=200,skip_hessian=True,disp=False)
                    # Obtaining predictions for fold validation data
                    preds = res_log.model.predict(res_log.params, Xvs[i])
                # Calculating fold accuracy
                yvs_pred = [numpy.argmax(x) for x in preds]
                accs[i] = metrics.accuracy_score(yvs[i],yvs_pred)
                print(distributions[d], methods[m], i, ". Accuracy:", accs[i])
            # Calculating distr-method combination accuracy
            acc_values[d,m] = accs.mean()
            print("distr:", distributions[d], "; method:", methods[m], ". CV-accuracy:", acc_values[d,m])

    # Finding optimal distr-method combination according to best accuracy        
    max_acc = numpy.max(acc_values)
    dm = unravel_index(acc_values.argmax(), acc_values.shape)
    best_d = distributions[dm[0]]
    best_m = methods[dm[1]]
    print("Optimal OrderedModel settings: distr =", best_d, "; method =", best_m)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Creating optimal model with the entire train data and best distr
        best_om = OrderedModel(y_train, X_train, distr=best_d)
        # Fitting optimal model with best method
        res_log = best_om.fit(method=best_m,max_iter=500,skip_hessian=True,dsip=False)
        # Printing summary of fitted model
        if report_features:
            print(res_log.summary())
    # Returning fitted model
    return res_log, best_d, best_m
    #return res_log

# Optimizing a c-supported vector classifier
def optimize_support_vector_machine (X_train, y_train, sample_weights=None):
    # Hyperparameter ranges and grid
    c_reg = [0.1,0.3,1,3,10,30,100,300]
    kernel = ['linear','poly','rbf','sigmoid']
    break_ties = [True,False]
    svc_grid = {'C': c_reg, 'kernel': kernel, 'break_ties': break_ties}

    # Creating the SVC model
    rs = int(datetime.datetime.now().timestamp())%1000
    svc = svm.SVC(random_state=rs)
    try:
        # Creating the optimization grid
        svc_model = GridSearchCV(estimator=svc, param_grid=svc_grid, verbose=3)
        # Tuning
        svc_model.fit(X_train, y_train.ravel(), sample_weight=sample_weights)
        
        print("Best C-supported vector classifier achieved with the following hyperparameters:\n", svc_model.best_params_)
        # Returning the best SVC variant
        return svc_model.best_estimator_
    except:
        print("Problem in method 'optimize_support_vectore_machine'")
        return None

# Optimizing a random forest classifier
def optimize_random_forest (X_train, y_train, sample_weights=None, report_features=False):

    # Hyperparameter ranges and grid
    coarse_n_estimators = numpy.arange(100,1500,100)
    coarse_max_depth = numpy.arange(4,11)
    coarse_min_samples_split = numpy.arange(4,11)
    coarse_min_samples_leaf = numpy.arange(2,6)
    bootstrap = [True,False]
    coarse_rfc_grid = {'n_estimators': coarse_n_estimators, 'max_depth': coarse_max_depth,\
                       'min_samples_split': coarse_min_samples_split, 'min_samples_leaf': coarse_min_samples_leaf,\
                       'bootstrap': bootstrap}
    # Creating the RFC model
    rfc = RandomForestClassifier()

    try:
        
        # Creating the coarse search mechanism
        rs = int(datetime.datetime.now().timestamp())%1000
        coarse_rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=coarse_rfc_grid, n_iter=50,\
                                               scoring='accuracy', verbose=3, random_state=rs)
        # Coarse tuning
        coarse_rfc_random.fit(X_train, y_train.ravel(), sample_weight=sample_weights)            

        # Saving best coarse variant
        best_params = coarse_rfc_random.best_params_

        # Deriving new sets of fine hyper-parameter value ranges around those of the best coarse variant
        fine_n_estimators = numpy.arange(best_params['n_estimators']-50,best_params['n_estimators']+60,10)
        fine_max_depth = numpy.arange(best_params['max_depth']-1,best_params['max_depth']+2)
        fine_min_samples_split = numpy.arange(best_params['min_samples_split']-1,best_params['min_samples_split']+2)
        fine_min_samples_leaf = numpy.arange(best_params['min_samples_leaf']-1,best_params['min_samples_leaf']+2)

        # Creating the grid for fine tuning
        fine_rfc_grid = {'n_estimators': fine_n_estimators, 'max_depth': fine_max_depth,\
                         'min_samples_split': fine_min_samples_split, 'min_samples_leaf': fine_min_samples_leaf,\
                         'bootstrap': bootstrap}

        # Creating the fine search mechanism
        rs = int(datetime.datetime.now().timestamp())%1000
        fine_rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=fine_rfc_grid, n_iter=50,\
                                             scoring='accuracy', verbose=3, random_state=rs)

        # Fine tuning
        fine_rfc_random.fit(X_train, y_train.ravel(), sample_weight=sample_weights)            

        # Selecting the best variant (just in case the best coarse variant wasn't included
        # in the second search and is better than the best fine variant)
        best_model = fine_rfc_random if fine_rfc_random.best_score_ > coarse_rfc_random.best_score_\
                     else coarse_rfc_random
        
        print("Best random forest classifier achieved with the following hyperparameters:\n", best_model.best_params_)
        
        # Printing feature importance
        if report_features:
            ftr_imp = list(best_model.best_estimator_.feature_importances_)
            feature_set = X_train.columns
            rf_feature_importances = [(ftr, round(imp, 3)) for ftr, imp in zip(feature_set,ftr_imp)]
            rf_feature_importances = sorted(rf_feature_importances, key = lambda x: x[1], reverse=True)
            print("Random forest feature importances:")
            for f in rf_feature_importances:
                print("Feature:", f[0], ", Importance:", f[1])
        # Returning the best DT variant
        return best_model.best_estimator_
    except:
        print("Problem in method 'optimize_random_forest'")
        return None            


# Contstruction method to be used internally with NN optimizer grid
def create_nn_variant(layer_nodes, input_nodes):
    # Constructing the network with input parameters
    model = keras.models.Sequential()
    model.add(layers.Dense(layer_nodes, input_shape=(input_nodes,), activation="relu"))
    model.add(layers.Dense(layer_nodes, activation="relu"))
    model.add(layers.Dense(10))
    # Loss function (may be parameterized later
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    # Optimizer
    optim_fn = optimizers.legacy.Adam()
    # Compiling the model
    model.compile(optimizer=optim_fn,loss=loss_fn,metrics=["accuracy"],weighted_metrics=["accuracy"])
    # Returning the compiled model to the optimizer grid
    return model    

# Optimizing a neural network classifier
def optimize_neural_network (X_train, y_train, sample_weights=None):
    # Hyperparameter ranges and grid
    coarse_learning_rates = [0.05,0.005,0.0005]
    coarse_batch_sizes = numpy.arange(15,60,5)
    coarse_layer_nodes = numpy.arange(10,31,3)
    input_nodes = X_train.shape[1]
    coarse_nn_grid = {"model__layer_nodes": coarse_layer_nodes, "model__input_nodes": [input_nodes],\
                      "optimizer__learning_rate" : coarse_learning_rates, "batch_size": coarse_batch_sizes}

    try:
        # Creating the NN model
        nn_model = KerasClassifier(model=create_nn_variant, verbose=3)
        rs = int(datetime.datetime.now().timestamp())%1000
        # Creating the coarse optimization grid
        coarse_nn_random = RandomizedSearchCV(estimator=nn_model, param_distributions=coarse_nn_grid, n_iter=50,\
                                              scoring='accuracy', verbose=3, random_state=rs, error_score='raise')
        # Coarse tuning
        coarse_nn_random.fit(X_train, y_train, epochs=80, sample_weight=sample_weights, verbose=0)

        best_params = coarse_nn_random.best_params_

        # Deriving new sets of fine hyper-parameter value ranges around those of the best coarse variant
        fine_learning_rates = [best_params['optimizer__learning_rate']*2,best_params['optimizer__learning_rate'],best_params['optimizer__learning_rate']/2]
        fine_batch_sizes = [best_params['batch_size']-5,best_params['batch_size'],best_params['batch_size']+5]
        fine_layer_nodes = [best_params['model__layer_nodes']-3,best_params['model__layer_nodes'],best_params['model__layer_nodes']+3]

        # Creating the grid for fine tuning
        fine_nn_grid = {"model__layer_nodes": fine_layer_nodes, "model__input_nodes": [input_nodes],\
                        "optimizer__learning_rate" : fine_learning_rates, "batch_size": fine_batch_sizes}

        # Creating the fine search mechanism
        fine_nn_model = GridSearchCV(estimator=nn_model, param_grid=fine_nn_grid, scoring='accuracy', verbose=3)

        # Fine tuning
        fine_nn_model.fit(X_train, y_train, epochs=120, sample_weight=sample_weights, verbose=0)          

        best_params = fine_nn_model.best_params_
        print("Best neural network classifier achieved with the following hyperparameters:\n", best_params)

        # Now reconstructing and compiling the fine-tuned model
        best_model = keras.models.Sequential([layers.Dense(best_params['model__layer_nodes'], input_shape=(best_params['model__input_nodes'],), activation="relu"),\
                                              layers.Dense(best_params['model__layer_nodes'], activation="relu"), layers.Dense(10)])
        optim_fn = optimizers.legacy.Adam()
        optim_fn.learning_rate.assign(best_params['optimizer__learning_rate'])
        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
        best_model.compile(optimizer=optim_fn,loss=loss_fn,metrics=["accuracy"],weighted_metrics=["accuracy"])
        # Creating callbacks for early stopping at minimal validation loss
        cb = callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=10, start_from_epoch=20)
        mc = callbacks.ModelCheckpoint("best_nn.tf", monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_format="tf")
        # Optimizing the model
        history = best_model.fit(X_train, y_train, epochs=200, sample_weight=sample_weights, batch_size=best_params['batch_size'],\
                                 validation_split=0.15, verbose=0, callbacks=[cb,mc])
        # Reloading the saved optimal variant
        saved_model = models.load_model("best_nn.tf")
        shutil.rmtree("best_nn.tf")

        # Returning the best NN variant
        return saved_model 
    except:
        print("Problem in method 'optimize_neural_network'")
        return None
