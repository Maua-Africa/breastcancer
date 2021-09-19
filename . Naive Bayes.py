from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print("Training Score: ",gnb.score(x_train,y_train)*100)
print(gnb.score(x_test,y_test))


data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data
