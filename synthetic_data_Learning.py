# synthetic : weight Training 수행을 하는 Perceptron

#필요한 라이브러리 설치
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#퍼셉트론 클래스 구현
class Perceptron():

    def __init__(self, n_iter=1000):
        self.n_iter = n_iter

    #과제에 주어진 알고리즘에 맞게 트레이닝 하는 함수
    # 해당 함수 1번 호출 = 1번 트레이닝
    # 트레이닝이 1번 완료될 때마다, weight와 잘못 분류된 갯수 반환
    def perceptronLearning(self, x, w):
        check_break = 0
        self.w = w

        for i in range(2000) :
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

    #Model이 예측한 label 값을 반환해주는 함수
    #Model이 예측한 label 값을 pred_class_list를 만들어, 넣어줌.
    def predict(self, x, w):
        pred_class_list = []

        for i in range(len(x)):
            check = np.dot(x[i, :4], w)
        
            if check > 0 :
                pred_class = 1.0
                pred_class_list.append(float(pred_class))
                
            else :
                pred_class = 0.0
                pred_class_list.append(float(pred_class))

        return pred_class_list

    # 정확도 출력 함수
    def accuracy(self, x, y):
        train_data = x
        test_data = y
        # bias를 고려하기 위한 dummy data '1' 삽입.
        train_data = np.insert(train_data, 3, 1, axis=1)
        #iteration 30회 출력을 위한 리스트
        check_break_list = []
        # weight initialize
        w = np.random.randn(4)

        # 최대 1000번의 학습을 돌리고, 트레이닝 횟수마다 잘못 분류된 갯수를 반환는다.
        # 만약 잘못 분류된 것이 하나도 없다면 braek로 training 종료.
        
        for i in range (1000):
             w, check_break = self.perceptronLearning(train_data, w)

             if check_break == 0 :
                 break
             
             else :
                 # 1 ~ 30번까지의 오류 갯수는 출력을 위해 check_breeak_list에 삽입 
                 if i < 30 : 
                    check_break_list.append(check_break)
                 check_break = 0

        plt.xlabel('Iteration')
        plt.ylabel('제대로 분류되지 않은 데이터 개수')
        plt.grid(True)
        plt.plot(check_break_list)
        plt.show()


        # 테스트 데이터에도 bias를 고려하여 dummy data '1' 삽입.
        test_data = np.insert(test_data, 3, 1, axis=1)
        preds = self.predict(test_data, w)
        # test_data의 label만 새로운 list에 삽입
        label = test_data[:, 4].tolist()

        accuracy_score = 0.0
        class_one = 0
        class_zero = 0
        # 루프를 돌며 label과 예측값이 맞는지 확인.
        # 맞다면 accuracy_score + 1.
        for i in range (len(label)):
            if preds[i] == label[i]:
                accuracy_score +=1.0
                if(label[i]) == 1 :
                    class_one +=1
                else :
                    class_zero +=1


        print('class 1로 제대로분류 된', class_one)
        print('class 0로 제대로 분류된', class_zero)        
        # 최종적으로 accuracy 출력
        # '모델이 맞춘 정답 갯수' / '전체 데이터(test_data) 갯수'
        accuracy = accuracy_score / float(len(preds))
        return accuracy

# train_data, test_data 불러옴
train_data = np.loadtxt('synthetic_data_train.txt', delimiter=',')
test_data = np.loadtxt('synthetic_data_test.txt', delimiter=',')

# Perceptron() 클래스 객체 생성
test = Perceptron()

#해당 accuracy 반환
accuracy = test.accuracy(train_data, test_data)
print('Accuracy : ', accuracy)
