import torch
import torch.nn as nn
import geffnet
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d
import pdb


sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    
class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()


    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)


        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)

        return loss


class Effnet(nn.Module):
    def __init__(self, enet_type, out_dim, pretrained=True, freeze_cnn=False, load_model=False ,pretrain_cnn=False,pretrain_file=''):
        super(Effnet, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()
        if pretrain_cnn:
            pretrained_dict=torch.load(pretrain_file)
            model_dict=self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.myfc = nn.Linear(in_ch, out_dim)
        if freeze_cnn:
            for p in self.enet.parameters():
                p.requires_grad = False
            
            
    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Resnest(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Resnest, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Seresnext(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
