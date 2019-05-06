## step wise logistical regression
## 2019/5/3 9:55
## Shanghaitech University SIST 1A406
## wangke
 import os
 import pandas as pd
 import statsmodels.api as sm
 from sklearn.metrics import auc,roc_curve
 import matplotlib.pyplot as plt
 
 def stepwise_selection(X, y, initial_list=[], threshold_in=0.02, threshold_out = 0.05, verbose = True):
     """ Perform a forward-backward feature selection
     based on p-value from statsmodels.api.OLS
     Arguments:
         X - pandas.DataFrame with candidate features
         y - list-like with the target
         initial_list - list of features to start with (column names of X)
         threshold_in - include a feature if its p-value < threshold_in
         threshold_out - exclude a feature if its p-value > threshold_out
         verbose - whether to print the sequence of inclusions and exclusions
     Returns: list of selected features
     Always set threshold_in < threshold_out to avoid infinite looping.
     See https://en.wikipedia.org/wiki/Stepwise_regression for the details
     """
     included = list(initial_list)
     while True:
         changed=False
         # forward step
         excluded = list(set(X.columns)-set(included))
         new_pval = pd.Series(index=excluded)
         for new_column in excluded:
             model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit(method='bfgs')
             new_pval[new_column] = model.pvalues[new_column]
         best_pval = new_pval.min()
         if best_pval < threshold_in:
             best_feature = new_pval.argmin()
             included.append(best_feature)
             changed=True
             if verbose:
                 print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
         # backward step
         model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(method='bfgs')
         # use all coefs except intercept
         pvalues = model.pvalues.iloc[1:]
         worst_pval = pvalues.max() # null if pvalues is empty
         if worst_pval > threshold_out:
             changed=True
             worst_feature = pvalues.argmax()
             included.remove(worst_feature)
             if verbose:
                 print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
         if not changed:
             break
     return included
 result = stepwise_selection(X, y)
 print('resulting features:')
 print(result)
 lr = sm.Logit(y,sm.add_constant(X[result]))
 rst = lr.fit()
 print(rst.summary2())
 y_predicted = rst.predict(sm.add_constant(X[result]))
 fpr, tpr, thresholds = roc_curve(y,y_predicted, pos_label=1)
 auc_score = auc(fpr,tpr)
 w = tpr - fpr
 ks_score = w.max()
 ks_x = fpr[w.argmax()]
 ks_y = tpr[w.argmax()]
 fig,ax = plt.subplots()
 ax.plot(fpr,tpr,label='AUC=%.5f'%auc_score)
 ax.set_title('Receiver Operating Characteristic')
 ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
 ax.plot([ks_x,ks_x], [ks_x,ks_y], '--', color='red')
 ax.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)
 ax.legend()
 fig.show()


