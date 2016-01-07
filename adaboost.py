import numpy as np

class Adaboost():
    def __init__(self,NumofClassifier):
        self.NumofClassifier = NumofClassifier

    def AdaboostTrain(self,wk_list, x, y):
        NumofData = x.shape[0]
        g_list = []
        total_min =[]
        weight_array = np.zeros((NumofData,1))
        weight_array[:,0] = 1.0/NumofData #init

        for t in range(self.NumofClassifier):
            min_err_rate = 1
            for w in wk_list:
                (stump, pos) = w[0] #the threshold and sign of decision stump
                dim = w[1]          #the dimension of decision stump
                #Get the decision result of decision stump
                decision_result = self.DecisionStump(stump,pos,x[:,dim:dim+1])
                err_rate = sum(weight_array[decision_result!=y])
                if err_rate < min_err_rate:
                    min_err_rate = err_rate
                    best_decision = decision_result
                    g = [(stump,pos),dim,0]          #weak classifier with minimize error
            min_err_rate = min_err_rate / sum(weight_array)
            if min_err_rate  > 0.5:        #If the error of weak classifer > 0.5, the algorithm stop
                print "error rate is larget than 0.5"
                return g_list

            total_min.append(min_err_rate)
            weight_update = np.sqrt((1-min_err_rate)/min_err_rate)
            alpha = np.log(weight_update)

      #update weight for each data samples
            weight_array = self.UpdWeight(best_decision,y,weight_array,weight_update)
            g[2] = alpha          #record the alpha value
            g_list.append(g)      #generate weak classifier of this iteration

        return g_list            #return the strong classification

    def AdaboostTest(self,g_list,x):
        N = x.shape[0]
        label_array = np.zeros((N,1))
        alpha_array = np.zeros((N,1))
        for g in g_list:
            (stump,pos) = g[0]
            dim = g[1]
            alpha = g[2]
            err_result = self.DecisionStump(stump,pos,x[:,dim:dim+1])
            alpha_array = alpha_array+err_result*alpha

        label_array[alpha_array > 0] = 1
        label_array[alpha_array <= 0] = -1

        return label_array

    def InitClassifier(self,stump,pos,dim):
            wkclassifier_list = []
            for i in range(len(stump)):
                wkclassifier = []
                feature = (stump[i],pos) #(stump,pos)
                wkclassifier.append(feature)
                wkclassifier.append(dim)
                wkclassifier.append(0) #init alpha
                wkclassifier_list.append(wkclassifier)
            return wkclassifier_list

    def DecisionStump(self,stump,pos,x):
        N = x.shape[0]
        err = np.zeros((N,1))
        if pos == 1:
            err[x > stump] = 1
            err[x <= stump] = -1
        elif pos == 0:
            err[x > stump] = -1
            err[x <= stump] = 1
        return err

    def UpdWeight(self,err,y,weight_array,weight_update):
        weight_array[err==y] = weight_array[err==y] / weight_update
        weight_array[err!=y] = weight_array[err!=y] * weight_update

        return weight_array

    #Get the median value of two data points
    def GetMedian(self,a,stump_constraint):
        a = np.sort(a)
        N = a.shape[0]
        stump = []
        for i in range(0,N-1):
            if (a[i+1] - a[i]) > stump_constraint:
                stump.append(float(a[i]+a[i+1])/2)
        return stump

    def GenWeakClassifier(self,x,stump_constraint):
        wkclassifier_list = []
        dim=x.shape[1]
        for i in range((dim-2)):
            stump = self.GetMedian(x[:,i],stump_constraint)
            wkclassifier_list=wkclassifier_list+(self.InitClassifier(stump,1,i)+ \
            self.InitClassifier(stump,0,i))
        return wkclassifier_list
