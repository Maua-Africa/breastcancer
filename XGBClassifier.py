xgb =XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xgb.fit(x_train, y_train)



y_pred=xgb.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Training Score: ",xgb.score(x_train,y_train)*100)
print(xgb.score(x_test,y_test))


print("Training Score: ",xgb.score(x_train,y_train)*100)

data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data
