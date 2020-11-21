# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


#creating Class for diabetes detection

class bloodpressureChecker():

    def __init__(self):
        #creating classifier
        self.classifier = RandomForestClassifier(n_estimators=2000,random_state=42) #best value 2000

    def preprocess(self):
        # Importing the dataset
        self.dataset = pd.read_csv('C:/Users/JANET MIGIRO/Downloads/data.csv')
        self.X = self.dataset.iloc[:,:8].values
        self.y = self.dataset.iloc[:, 8].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.20, random_state = 42)

        # Feature Scaling
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)


        ''''# Applying Kernel PCA  without this 74% with this 70%
        from sklearn.decomposition import KernelPCA
        self.kpca = KernelPCA(n_components=2, kernel='rbf')
        self.X_train = self.kpca.fit_transform(self.X_train)
        self.X_test = self.kpca.transform(self.X_test)'''


    def train(self):
        self.preprocess()
        # Fitting Random Forest Classification to the Training set
        self.classifier.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)

        #accuracy
        acc = round(100*self.classifier.score(self.X_test,self.y_test),3)
        print(str(acc)+' %')
        #print(self.classifier.feature_importances_)

        pickle.dump(self.classifier,open('Saved Classifier '+ str(acc)+'% .sav', 'wb'))


    def load(self):
        self.classifier=pickle.load(open('Saved Classifier 69.481% .sav', 'rb'))
        self.preprocess()
        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)

        #accuracy
        acc = round(100*self.classifier.score(self.X_test,self.y_test),3)
        print(str(acc)+' %')

    def visualise(self):
        from matplotlib.patches import Rectangle
        #prediction histogram Test set
        # create legend
        labels = ["Predicted","Actual"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['#EF7215','#411F15']]
        plt.legend(handles, labels)
        #histogram
        ticks = [0.17,0.835]
        label1 = ['Normal','Diabetic']
        plt.hist([self.y_test,self.y_pred],bins=3,color=['#EF7215','#411F15'])
        plt.xticks(ticks,label1)
        plt.title('Machine predicted V/S Actual Diabetes Patient')
        plt.xlabel('Status')
        plt.ylabel('Frequency')
        plt.show()


        '''#prediction histogram Train set
        # create legend
        labels = ["Predicted","Actual"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['orange','blue']]
        plt.legend(handles, labels)
        #histogram
        ticks = [0.17,0.835]
        label1 = ['Normal','Diabetic']
        plt.hist([self.y_train,self.classifier.predict(self.X_train)],bins=3)
        plt.xticks(ticks,label1)
        plt.title('Machine predicted V/S Actual Diabetes Patient\n(Train Set)')
        plt.xlabel('Status')
        plt.ylabel('Frequency')
        plt.show()'''

        # Adjusting the threshold to 0.1
        from sklearn.preprocessing import binarize

        self.y_pred_class = binarize([self.y_pred], 0.5)
        # Data Visualisation of ROC curve :
        # equilibrium between True (tpr) and False (fpr) Positive Rate
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
        line = np.linspace(0, 1, 2)
        plt.plot(line, 'r')
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC Curve for Diabetes Classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.show()




if __name__=="__main__" :
    c = DiabetesChecker()
    c.train()
    c.visualise()