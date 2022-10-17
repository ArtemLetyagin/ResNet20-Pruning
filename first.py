import torch.nn as nn
import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F

class PruneLayer(nn.Module):
    def __init__(self, layer, n_clusters, skipper=False):
        super(PruneLayer, self).__init__()
        self.skipper = skipper
        
        def clust_one_conv(conv, n_clusters):
            cin, h, w = conv.shape
            kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(layer.reshape(cin,h*w))
            labels = kmeans.labels_
            centroids  = kmeans.cluster_centers_.reshape(n_clusters, h, w)
            return centroids, labels
        def clustering(layer, n_clusters):
            cout, cin, h, w = layer.shape
            kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(layer.reshape(cout,cin*h*w))
            labels = kmeans.labels_
            centroids  = kmeans.cluster_centers_.reshape(n_clusters, cin, h, w)
            return centroids, labels
        
        def cluster_block(weights, padding, stride):
            cen_convs = nn.ModuleList()
            num_centroids = weights.shape[0]
            c_in = weights.shape[1]
            kernel_size = weights.shape[2]
            for i in range(num_centroids):
                centroid_conv = nn.Conv2d(c_in, 1, kernel_size, padding=padding, stride=stride)
                centroid_conv.weight.data = torch.tensor([weights[i]])
                cen_convs.append(centroid_conv)
            return cen_convs
        
        resBlock = {}
        for name, param in layer.named_parameters():
            resBlock[name] = param.data.detach().numpy()
            
        conv_center1, labels1 = clustering(resBlock['conv1.weight'], n_clusters)
        conv_center2, labels2 = clustering(resBlock['conv2.weight'], n_clusters)
        
        if skipper:
            conv_center3, labels3 = clustering(resBlock['skipper.0.weight'], n_clusters)
        
        if skipper:
            self.c1 = cluster_block(conv_center1, 1, 2)
            self.c2 = cluster_block(conv_center2, 1, 1)
            self.skipper = cluster_block(conv_center3, 0, 2)
            self.bn3 = layer.skipper[1]
            self.labels1 = labels1
            self.labels2 = labels2
            self.labels3 = labels3
            
        else:
            self.c1 = cluster_block(conv_center1, 1, 1)
            self.c2 = cluster_block(conv_center2, 1, 1)
            
            self.labels1 = labels1
            self.labels2 = labels2
            
        self.bn1 = layer.bn1
        self.bn2 = layer.bn2
        
    def forward(self, x):
        def centroids_apply(x, centroids, labels):
            centroid_work = []
            for conv in centroids:
                centroid_work.append(conv(x))
            out = centroid_work[labels[0]]
            for i in range(1, len(labels)):
                out = torch.cat((out,centroid_work[labels[i]]), axis=1)
            return out
        
        out = F.relu(self.bn1(centroids_apply(x, self.c1, self.labels1)))
        out = self.bn2(centroids_apply(out, self.c2, self.labels2))
        
        if self.skipper:
            out += self.bn3(centroids_apply(x, self.skipper, self.labels3))
        else:
            out += x
            
        out = F.relu(out)
        
        return out



class FirstPruning(nn.Module):
    def __init__(self, model, r):
        super(FirstPruning, self).__init__()
        r = 1-r
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.layer1 =  PruneLayer(model.layer1, int(16*r))
        self.layer2 =  PruneLayer(model.layer2, int(16*r))
        self.layer3 =  PruneLayer(model.layer3, int(16*r))
        self.layer4 =  PruneLayer(model.layer4, int(32*r), skipper=True)
        self.layer5 =  PruneLayer(model.layer5, int(32*r))
        self.layer6 =  PruneLayer(model.layer6, int(32*r))
        self.layer7 =  PruneLayer(model.layer7, int(64*r), skipper=True)
        self.layer8 =  PruneLayer(model.layer8, int(64*r))
        self.layer9 =  PruneLayer(model.layer9, int(64*r))
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