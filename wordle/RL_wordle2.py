import time
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import words
from utils import Web
np.random.seed(0)
class Wordle:
    def __init__(self,epoch=200,eta=0.003,max_generate=20000):
        lst = words.words()
        self.lst5 = [word for word in lst if (len(word) == 5 and ord(word[0]) >= 97)]
        self.history = []
        self.phi = [0.1, 0.1, 0.1]
        self.phi_history = [self.phi]
        # self.used_time=[]
        self.epoch = epoch
        self.eta = eta
        self.max_generate = max_generate
        self.theta_0 = np.ones((5, 26))
        self.pi_0 = self.softmax(self.theta_0)
    def init_theta(self):
        theta_0 = np.zeros((5,26))
        for i in range(5):
            for j in self.lst5:
                theta_0[i,ord(j[i])-97] += 1
            theta_0[i,:]/=max(theta_0[i])
        theta_0[np.where(theta_0 == 0)]=np.nan
        return theta_0
    def plot_bar(self,theta_0):
        x = [chr(i) for i in range(97, 97 + 26)]
        for i in range(5):
            plt.title(f'The {i+1} letter frequency')
            plt.bar(x, theta_0[i])
            plt.show()
    def softmax(self,theta):
        [m, n] = theta.shape
        pi=np.zeros((m, n))
        exp_theta = np.exp(theta)
        for i in range(m):
            pi[i, :] = exp_theta[i, :]/np.nansum(exp_theta[i, :])
        pi = np.nan_to_num(pi)
        return pi
    def sigmoid(self,theta):
        return 1/(1+np.exp(-theta))
    def alphabet_action(self,pi, state):
        return np.random.choice([i for i in range(26)], p = pi[state])
    def result(self,theta,word,ans):
        #obs = [None for _ in range(5)]
        for i in range(5):
            if word[i] == ans[i]:
                theta[i,:] = np.nan
                theta[i,word[i]] = 1
            elif word[i] in ans:
                pass
            else:
                theta[:,word[i]] = np.nan
        return theta
    def word_to_number(self,word):
        num = []
        for alphabet in word:
            num.append(ord(alphabet)-97)
        return np.array(num)
    def number_to_word(self,num):
        word = []
        en_word = ''
        for n in num:
            word.append(chr(n + 97))
        for alphabet in word:
            en_word += alphabet
        return en_word

    def result2(self,theta,word,ans,phi):
        #obs = [None for _ in range(5)]
        #delta=phi[0] k1=phi[1]
        state = [0,0,0]#一共有幾個 全對、只有字母對、全錯
        obs=[None for _ in range(5)]#幾A幾B的意思
        for i in range(5):
            if word[i] == ans[i]:
                theta[i, :word[i]] -= phi[0]
                theta[i, word[i]+1:] -= phi[0]
                theta[i,word[i]] += phi[0]
                # theta[:i, word[i]] -= phi[0]*phi[1]#update
                # theta[i+1:, word[i]] -= phi[0]*phi[1]#update

                state[0]+=1
                obs[i]=1
            elif word[i] in ans:
                theta[:i, word[i]] += phi[1]
                theta[(i+1):, word[i]] += phi[1]

                theta[i, word[i]] -= phi[1]
                theta[i, :word[i]] += phi[1]#update
                theta[i, word[i]+1:] += phi[1]#update

                state[1] += 1
                obs[i]=2
            else:
                theta[:,word[i]] -= phi[2]

                state[2] += 1
                obs[i]=3
        print('--')
        print(self.number_to_word(word))
        print(obs)
        return theta,state

    def update_phi(self,phi, t, state):
        phi=np.array(phi)
        r1=state[0]
        r2=state[1]
        r3=state[2]
        delta = np.array([r1/t,r2/t,r3/t])
        return phi + self.eta*delta

    def play(self,phi=None):
        #theta_0 = np.ones((5, 26))
        theta_0 = self.init_theta()
        #pi_0 = self.softmax(theta_0)
        en_ans = np.random.choice(self.lst5).lower()
        print('ans:',en_ans)
        ans = self.word_to_number(en_ans)
        t=1
        testtime=1
        state=[]
        #initword = np.random.choice(self.lst5).lower()
        initword = 'soare'
        initword = self.word_to_number(initword)
        theta = self.result2(theta_0, initword, ans, self.phi)[0]
        pi = self.softmax(theta)

        word = [0 for _ in range(5)]
        en_word = self.number_to_word(word)
        while en_word != en_ans:
            testtime+=1
            if testtime >= self.max_generate:
                # print(f'pi:{pi}')
                # print(f'theta:{theta}')
                break
            for alphabet in range(5):
                word[alphabet] = self.alphabet_action(pi, alphabet)
            en_word = self.number_to_word(word)
            #print(en_word)
            if en_word in self.lst5:
                #print(en_word)
                t += 1
                theta, s = self.result2(theta, word, ans, phi)
                state.append(s)
                pi = self.softmax(theta)
                if en_word == en_ans:
                    print(t,en_word)

            else:
                pass
                # print('do not exist')
        return t,np.array(state),testtime

    def train(self):
        for i in range(self.epoch):
            print('---------------------\ntesttime:', i)
            t, epoch_state, usedtime = self.play(self.phi)
            if t == 1:
                continue
            self.history.append(t)
            r1 = sum([i for i in epoch_state.T[0] if i > 0])
            r2 = sum([i for i in epoch_state.T[1] if i > 0])
            r3 = sum([i for i in epoch_state.T[2] if i > 0])

            self.phi = self.update_phi(self.phi, t, [r1,r2,r3])
            self.phi_history.append(self.phi)
            # used_time.append(usedtime)
        # print(sum(history)/len(history))

        phi_history = np.array(self.phi_history)
        # print(phi_history)
        np.save('out', phi_history[-1])

        plt.title('guess times')
        plt.axhline(y=6, xmin=0, xmax=self.epoch)
        plt.plot(self.history, 'g')
        plt.show()
    def run(self):
        web = Web()
        phi=np.load('out.npy')
        theta_0 = self.init_theta()
        initword = 'soare'
        web.answer('soare')
        result = web.result()
        initword = self.word_to_number(initword)
        theta = self.web_result(theta_0, initword, phi, result)[0]
        pi = self.softmax(theta)

        word = [0 for _ in range(5)]
        while True:
            for alphabet in range(5):
                word[alphabet] = self.alphabet_action(pi, alphabet)
            en_word = self.number_to_word(word)
            if en_word in self.lst5:
                print('answering:',en_word)
                web.closeweb()
                web = Web()
                web.answer(en_word)
                try:
                    result = web.result()
                except Exception as e:
                    print(en_word,'not in word list!')
                    continue
                if result == [1, 1, 1, 1, 1]:
                    print('find ans!!!!!!!!!!!')
                    break
                theta= self.web_result(theta_0, word, phi, result)[0]
                pi = self.softmax(theta)
        return en_word
    def web_result(self,theta,word,phi,result):
        #obs = [None for _ in range(5)]
        #delta=phi[0] k1=phi[1]
        state = [0,0,0]#一共有幾個 全對、只有字母對、全錯
        obs=[None for _ in range(5)]#幾A幾B的意思
        for i in range(5):
            if result[i] == 1:
                theta[i, :word[i]] -= phi[0]
                theta[i, word[i]+1:] -= phi[0]
                theta[i,word[i]] += phi[0]
                # theta[:i, word[i]] -= phi[0]*phi[1]#update
                # theta[i+1:, word[i]] -= phi[0]*phi[1]#update

                state[0]+=1
                obs[i]=1
            elif result[i] == 2:
                theta[:i, word[i]] += phi[1]
                theta[(i+1):, word[i]] += phi[1]

                theta[i, word[i]] -= phi[1]
                theta[i, :word[i]] += phi[1]#update
                theta[i, word[i]+1:] += phi[1]#update

                state[1] += 1
                obs[i]=2
            elif result[i] == 3:
                theta[:,word[i]] -= phi[2]
                state[2] += 1
                obs[i]=3

        print(self.number_to_word(word))
        print(obs)
        print('--')
        return theta,state
if __name__=='__main__':
    t1=time.time()
    wordle=Wordle(eta=0.01,max_generate=50000,epoch=200)
    #wordle.train()
    ans = wordle.run()
    #find answer
    web = Web(invisiable=False)
    web.answer(ans)
    t2=time.time()
    print('used time:',t2-t1)