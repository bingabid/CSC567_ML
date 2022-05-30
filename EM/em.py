import json
import requests
import matplotlib.pyplot as plt

class EM_Algorithm:
    # initialize class varibale: api url, #draws, #epochs 
    def __init__(self, url = "https://24zl01u3ff.execute-api.us-west-1.amazonaws.com/beta", draws = 30, epochs = 10, data = "data_sample.txt", clean_data = "clean_data.txt", model = "em_model.json"):
        self.url = url
        self.data = data
        self.draws = draws
        self.model = model
        self.theta_a = 0.94 # assign any random value between 0 to 1
        self.theta_b = 0.16 # assign any random value between 0 to 1, make sure self.theta_a != self.theta_b
        self.epochs = epochs
        self.sample_data = []
        self.clean_data = clean_data

    # parse the data retrieved using api and save in data.txt file
    def get_data(self):
        with open(self.clean_data, 'w') as f:
            with open(self.data, 'w') as wf:
                for draw in range(self.draws):
                    r = requests.get(self.url)
                    files = r.json()
                    result = files['body']
                    head = result.count('1')
                    tail = result.count('0')
                    self.sample_data.append([head, tail])
                    f.write("head: " + str(head) + ", tail: " + str(tail) + '\n')
                    wf.write(result)
                    wf.write('\n')

    # EM alogrithm to learn model parameter
    def em_decoding(self):
        self.thetaa, self.thetab = [], []
        self.thetaa_epoch, self.thetab_epoch = [], []
        for epoch in range(self.epochs):

            # saving model parameter and epoch to plot the graph
            self.thetaa.append(self.theta_a)
            self.thetab.append(self.theta_b)
            self.thetaa_epoch.append(epoch)
            self.thetab_epoch.append(epoch)
            # Expectation(E) step: guessing a probability distribution 
            expected_count = [[0.0, 0.0], [0.0, 0.0]]
            for data in self.sample_data:
                head, tail = data[0], data[1]
                prob_a = (self.theta_a**head)*((1-self.theta_a)**tail) # prob. of the sample coming from coin A
                prob_b = (self.theta_b**head)*((1-self.theta_b)**tail) # prob. of the sample coming from coin B
                prob_a = prob_a/(prob_a+prob_b) # normalization
                prob_a = float("{:.2f}".format(prob_a))
                prob_b = 1 - prob_a
                #expected number of head and tail for each coin - 
                expected_count[0][0] += head*prob_a
                expected_count[0][1] += tail*prob_a
                expected_count[1][0] += head*prob_b
                expected_count[1][1] += tail*prob_b
            
            # Maximization(M) step: re-estimating the model parameters
            self.theta_a = expected_count[0][0] / (expected_count[0][0] + expected_count[0][1])
            self.theta_a = float("{:.2f}".format(self.theta_a))
            self.theta_b = expected_count[1][0] / (expected_count[1][0] + expected_count[1][1]) 
            self.theta_b = float("{:.2f}".format(self.theta_b))

    # save the model parameter
    def save_model(self):
        model = {}
        model['theta_a'] = self.theta_a
        model['theta_b'] = self.theta_b
        with open(self.model, 'w', encoding='utf-8') as wf:
            json.dump(model, wf, ensure_ascii=False, indent=4)
        print(self.theta_a)
        print(self.theta_b)

    def plot_graph(self):

        # plt.subplot(1, 2, 1)
        plt.plot(self.thetaa_epoch, self.thetaa,  'o:b', linestyle = 'dotted')
        plt.title("Model Estimation: Coin A")
        plt.xlabel("Epochs")
        plt.ylabel("θ_a")
        plt.show()
        # plt.subplot(1, 2, 2)
        plt.plot(self.thetab_epoch, self.thetab,  'o:r', linestyle = 'dotted')
        plt.title("Model Estimation: Coin B")
        plt.xlabel("Epochs")
        plt.ylabel("θ_b")
        plt.show()


if __name__ == "__main__":
    eM = EM_Algorithm()
    eM.get_data()
    eM.em_decoding()
    eM.save_model()
    eM.plot_graph()
