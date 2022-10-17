import torch.nn as nn
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn.functional as F

class PrunedConv(nn.Module):
    def __init__(self, conv, padding, stride, n_clusters, saved_filters):
        super(PrunedConv, self).__init__()
        def clust_one_filter(weights, n_clusters):
            cout, cin, h, w = weights.shape
            kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(weights.reshape(cout,cin*h*w))
            labels = kmeans.labels_
            centroids  = kmeans.cluster_centers_.reshape(n_clusters, cin, h, w)
            return centroids, labels #centroids shape - (n_clusters x cin x 3 x 3)

        def get_new_weights(centroid, label):
            new_labels = []
            saved_filters = []
            for l in labels:
                if l not in new_labels:
                    new_labels.append(l)
                    saved_filters.append(1)
                else:
                    saved_filters.append(0)
            weights = []
            for centroid_class in new_labels:
                weights.append(torch.tensor(centroid[centroid_class]))
            weights = torch.stack(weights)
            return weights, saved_filters
        
        weights = conv.weight.data
        clear_weights = []
        for i in range(len(saved_filters)):
            if saved_filters[i]==1:
                clear_weights.append(weights[:,i].unsqueeze(1))
                
        weights = torch.cat(clear_weights, axis=1)
        centroids, labels = clust_one_filter(weights, n_clusters)
        new_weights, saved_filters = get_new_weights(centroids, labels)
        
        c = nn.Conv2d(new_weights.shape[1], n_clusters, 3, padding=padding, stride=stride, bias=False)
        c.weight.data = new_weights.type(torch.float)
        self.c = c
        self.saved_filters = saved_filters
        self.labels = labels
        
    def forward(self, x):
        return self.c(x)
    
class PrunedBn(nn.Module):
    def __init__(self, bn, saved_filters):
        super(PrunedBn, self).__init__()
        weight = bn.weight.data
        bias = bn.bias.data
        tmp = []
        
        new_weight = []
        new_bias = []
        for i in range(len(saved_filters)):
            if saved_filters[i]==1:
                new_weight.append(weight[i])
                new_bias.append(bias[i])
            
        new_bn = nn.BatchNorm2d(len(new_weight))
        new_bn.weight.data = torch.tensor(new_weight).type(torch.float)
        new_bn.bias.data = torch.tensor(new_bias).type(torch.float)
        
        self.bn = new_bn
        
    def forward(self, x):
        return self.bn(x)



class PruneResNetLayer(nn.Module):
    def __init__(self, layer, n_clusters, skipper=False):
        super(PruneResNetLayer, self).__init__()
        self.skipper = skipper
        
        if skipper:
            saved_filters = np.ones(layer.conv1.weight.data.shape[1])

            self.c1 = PrunedConv(layer.conv1, padding=1, stride=2, n_clusters=n_clusters, saved_filters=saved_filters)
            tmp_saved_filters = self.c1.saved_filters

            self.c2 = PrunedConv(layer.conv2, padding=1, stride=1, n_clusters=layer.conv2.weight.data.shape[0], saved_filters=tmp_saved_filters)
            
            self.skip = layer.skipper[0]
            self.bn3 = layer.skipper[1]
        else:
            saved_filters = np.ones(layer.conv1.weight.data.shape[1])
            self.c1 = PrunedConv(layer.conv1, padding=1, stride=1, n_clusters=n_clusters, saved_filters=saved_filters)
            tmp_saved_filters = self.c1.saved_filters

            self.c2 = PrunedConv(layer.conv2, padding=1, stride=1, n_clusters=layer.conv2.weight.data.shape[0], saved_filters=tmp_saved_filters)
        
        
        self.bn1 = PrunedBn(layer.bn1, self.c1.saved_filters)
        self.bn2 = layer.bn2
        
    def forward(self, x):
        out = F.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))

        if self.skipper:
            out += self.bn3(self.skip(x))
        else:
            out += x

        out = F.relu(out)

        return out


class SecondPruning(nn.Module):
    def __init__(self, model, r):
        super(SecondPruning, self).__init__()
        r = 1-r
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.layer1 =  PruneResNetLayer(model.layer1, int(16*r))
        self.layer2 =  PruneResNetLayer(model.layer2, int(16*r))
        self.layer3 =  PruneResNetLayer(model.layer3, int(16*r))
        self.layer4 =  PruneResNetLayer(model.layer4, int(32*r), True)
        self.layer5 =  PruneResNetLayer(model.layer5, int(32*r))
        self.layer6 =  PruneResNetLayer(model.layer6, int(32*r))
        self.layer7 =  PruneResNetLayer(model.layer7, int(64*r), True)
        self.layer8 =  PruneResNetLayer(model.layer8, int(64*r))
        self.layer9 =  PruneResNetLayer(model.layer9, int(64*r))
        self.avg = model.avg
        self.l = model.l
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.avg(out).view(-1,1,64)
        out = self.l(out)
        return out


