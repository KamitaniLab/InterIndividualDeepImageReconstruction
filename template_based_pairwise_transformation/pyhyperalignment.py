import numpy as np
from scipy.linalg import orthogonal_procrustes


class Hyperalignment(object):

    def __init__(self):
        self.hypmaps = []
        self.tm = None
        return 


    def train(self, data_list, Lambda=None):


        target = data_list[0]
        tm = target.shape[1]
        self.tm = tm
        sbj_num = len(data_list)

        # Level 1
        level1_data_list = [data_list[0]]
        for i in range(1,sbj_num):
            source = data_list[i]
            sm = source.shape[1]
            if sm < tm:
                source = np.hstack([source, np.zeros((source.shape[0], tm-sm))])
            if sm > tm:
                target = np.hstack([target, np.zeros((target.shape[0], sm-tm))])           

            M, scale = orthogonal_procrustes(source, target)

            transformed = np.matmul(source, M)[:,:tm]
            level1_data_list.append(transformed)
            target = target[:,:tm]

            target = (target + transformed)/2
            

        # Level 2
        level2_data_list = []
        for i in range(sbj_num):
            source = data_list[i]
            sm = source.shape[1]
            
            others_data = []
            for j in range(sbj_num):
                if j == i:
                    continue
                else:
                    others_data.append(level1_data_list[j])
            target = np.mean(others_data, axis=0)
            
            if sm < tm:
                source = np.hstack([source, np.zeros((source.shape[0], tm-sm))])
            if sm > tm:
                target = np.hstack([target, np.zeros((target.shape[0], sm-tm))])           

            M, scale = orthogonal_procrustes(source, target)
            transformed = np.matmul(source, M)[:,:tm]
            target = target[:,:tm]

            level2_data_list.append(transformed)


        target = np.mean(level2_data_list, axis=0)

        # Level 3
        for i in range(sbj_num):
            source = data_list[i]
            sm = source.shape[1]
            if sm < tm:
                source = np.hstack([source, np.zeros((source.shape[0], tm-sm))])
            if sm > tm:
                target = np.hstack([target, np.zeros((target.shape[0], sm-tm))])           

            M, scale = orthogonal_procrustes(source, target)
            #transformed = np.matmul(source, M)[:,:tm]
            self.hypmaps.append(M)
        return self.hypmaps

    def forward(self, data_list):
        transformed = []
        for i, x in enumerate(data_list):
            sm = x.shape[1]
            if sm < self.tm:
                x = np.hstack([x, np.zeros((x.shape[0], self.tm-sm))])
            transformed.append(np.matmul(x, self.hypmaps[i])[:,:self.tm])
        return transformed