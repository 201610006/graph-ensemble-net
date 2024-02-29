from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



class AverageReporter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.targ = []
        self.pred = []

    def update(self, target, predict):
        self.targ.append(target.cpu().tolist())
        predict = predict.cpu()
        _, pred = predict.topk(1, 1, True, True)
        pred = pred.t()[0].tolist()
        self.pred.append(pred)

    def reporter(self):
        #print(self.targ)
        #print(self.pred)
        #target = np.array(self.targ).flatten().tolist()
        #predict = np.array(self.pred).flatten().tolist()
        target = [j for i in self.targ for j in i]
        predict = [j for i in self.pred for j in i]
        print(confusion_matrix(target, predict))
        print(classification_report(target, predict, digits=4))#, target_names=['RUP', 'SP', 'HP', 'FP', 'PP', 'TFP','ot1','ot2','ot3','ot4']))
