import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

# Learning이 없는 Perceptron() 클래스 생성
class no_learn_Perceptron():

    #트레이닝 과정(weight update)이 없기 때문에, weight와 각 feature vector를 내적하여
    #그 값을 test 조건으로 그대로 사용
    def no_learn_predict(self, x):
        no_learn_pred_class_list = []
        # w에 랜덤한 값으로 initialize.
        w = np.random.randn(4)
        test_data = x

        # Condition에 따라 해당 label 체크
        for i in range(len(test_data)):
            check = np.dot(x[i, :4], w)
        
            if check > 0 :
                pred_class = 1.0
                no_learn_pred_class_list.append(float(pred_class))
                
            else :
                pred_class = 0.0
                no_learn_pred_class_list.append(float(pred_class))

        return no_learn_pred_class_list
    
    # accuracy()함수와 동일한 작동
    def no_learn_accuracy(self, x):
        preds = self.no_learn_predict(x)
        label = x[:, 4].tolist()

        class_one = 0
        class_zero = 0
        accuracy_score = 0.0    
   
        for i in range (len(label)):
            if preds[i] == label[i]:
                accuracy_score +=1.0
                if(label[i]) == 1 :
                    class_one +=1
                else :
                    class_zero +=1

        #print('class 1로 제대로분류 된', class_one)
        #print('class 0로 제대로 분류된', class_zero)       


        accuracy = accuracy_score / float(len(preds))
        return accuracy


#test_data만 사용하기 때문에, test_data만 불러옴
test_data = np.loadtxt('synthetic_data_test.txt', delimiter=',')
#bias를 고려햐여 test_data에 dummy data '1' 삽입
test_data = np.insert(test_data, 3, 1, axis=1)

test = no_learn_Perceptron()

no_train_accuracy = test.no_learn_accuracy(test_data)
print('no_learn_Accuracy : ', no_train_accuracy)


