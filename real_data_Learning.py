import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


class Perceptron():

    def __init__(self, n_iter=1000):
        self.n_iter = n_iter

    def perceptronLearning(self, x, w):
        check_break = 0
        self.w = w

        for i in range(8143) :
            X_normalized = x[i][:4] / np.linalg.norm(x[i][:4])
            check = np.dot(self.w, x[i][:4])

            if x[i, 4] == 1 :
                if check <= 0 :
                    self.w += X_normalized
                    check_break +=1

            elif x[i, 4] == 0 : 
                if check > 0 :
                    self.w -= X_normalized
                    check_break +=1

        return w, check_break

    def predict(self, x, w):
        pred_class_list = []
        self.test_data = test_data

        for i in range(len(self.test_data)):
            check = np.dot(x[i, :4], w)
        
            if check > 0 :
                pred_class = 1.0
                pred_class_list.append(float(pred_class))
                
            else :
                pred_class = 0.0
                pred_class_list.append(float(pred_class))

        return pred_class_list


    def accuracy(self, x, y):
        train_data = x
        test_data = y
        train_data = np.insert(train_data, 3, 1, axis=1)
        check_break_list = []
        w = np.random.randn(4)

        for i in range (1000):
             w, check_break = self.perceptronLearning(train_data, w)

             if check_break == 0 :
                 break
             
             else :
                 if i < 30 : 
                    check_break_list.append(check_break)
                 check_break = 0

        plt.xlabel('Iteration')
        plt.ylabel('제대로 분류되지 않은 데이터 개수')
        plt.xticks(range(1, 32))
        plt.grid(True)
        plt.plot(check_break_list)
        plt.show()


        test_data = np.insert(test_data, 3, 1, axis=1)

        preds = self.predict(test_data, w)
        label = test_data[:, 4].tolist()

        accuracy_score = 0.0
        class_one = 0
        class_zero = 0
        for i in range (len(label)):
            if preds[i] == label[i]:
                accuracy_score +=1.0
                if(label[i]) == 1 :
                    class_one +=1
                else :
                    class_zero +=1


        print('class 1로 제대로분류 된', class_one)
        print('class 0로 제대로 분류된', class_zero) 

        accuracy = accuracy_score / float(len(preds))
        return accuracy
    


train_data = pd.read_table('datatraining.txt',sep=',')
test_data = pd.read_table('datatset2.txt',sep=',')

train_data = train_data[['Temperature','Light','CO2','Occupancy']]
test_data = test_data[['Temperature','Light','CO2','Occupancy']]

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# aa = test_data[:, -1]
# one = 0
# zero = 0
# for i in range (len(aa)) :
#     if (aa[i] == 1) :
#         one +=1
#     else :
#         zero +=1



# print(one)
# print(zero)
# exit()
test = Perceptron()

accuracy = test.accuracy(train_data, test_data)
print('Accuracy : ', accuracy)
