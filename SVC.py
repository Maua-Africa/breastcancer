from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)



y_pred=svc.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Training Score: ",svc.score(x_train,y_train)*100)
print(svc.score(x_test,y_test))



print("Training Score: ",svc.score(x_train,y_train)*100)
