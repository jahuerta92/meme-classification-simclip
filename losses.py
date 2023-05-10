import torch
import torch.nn.functional as F
import pytorch_lightning as L

class CustomLoss(L.LightningModule):
    def __init__(self, num_classes, smoothing=0., class_weights=None, scale=1.):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing+1e-12
        self.weights = class_weights
        self.scale = scale
    
    def generate_labels(self, labels):
        if len(labels.shape) > 1:
            return labels
        else:
             label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
             return torch.where(label_one_hot==1.,
                           1.-self.smoothing,
                           self.smoothing/(self.num_classes-1))
    
    def weight(self, loss, labels):
        if self.weights is not None:
            if len(labels.shape) > 1:
                return (self.weights.to(self.device).unsqueeze(0) * labels).sum(-1) * loss
            else:
                return self.weights.to(self.device)[labels] * loss
        else: 
            return loss


class NormalizedGeneralizedCrossEntropy(CustomLoss):
    def __init__(self, q=0.7, *args, **kwargs):
        super(NormalizedGeneralizedCrossEntropy, self).__init__(*args, **kwargs)
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = self.generate_labels(labels)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        loss = numerators / denominators
        return self.scale * self.weight(loss, labels).mean()
    
class NormalizedCrossEntropy(CustomLoss):
    #FROM https://github.com/HanxunH/Active-Passive-Losses
    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = self.generate_labels(labels)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * self.weight(loss, labels).mean()
    
class ReverseCrossEntropy(CustomLoss):
    #FROM https://github.com/HanxunH/Active-Passive-Losses
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = self.generate_labels(labels)
        loss = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * self.weight(loss, labels).mean()
    
class MeanAbsoluteError(CustomLoss):
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = self.generate_labels(labels)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * self.weight(loss, labels).mean()


class CombinedLoss(L.LightningModule):
    def __init__(self):
        super(L.LightningModule, self).__init__()

    def forward(self, pred, labels):
        return self.loss_a(pred, labels) + self.loss_p(pred, labels)

class NCEandRCE(CombinedLoss):
    #FROM https://github.com/HanxunH/Active-Passive-Losses
    def __init__(self, num_classes, alpha=1., beta=1., smoothing=0., class_weights=None):
        super(NCEandRCE, self).__init__()
        self.loss_a = NormalizedCrossEntropy(scale=alpha, 
                                             num_classes=num_classes, 
                                             smoothing=smoothing, 
                                             class_weights=class_weights)
        self.loss_p = ReverseCrossEntropy(scale=beta,
                                          num_classes=num_classes, 
                                          smoothing=smoothing, 
                                          class_weights=class_weights)

class NGCEandMAE(CombinedLoss):
    #FROM https://github.com/HanxunH/Active-Passive-Losses
    def __init__(self, num_classes, alpha=1., beta=1., q=.7, smoothing=0., class_weights=None):
        super(NGCEandMAE, self).__init__()
        self.loss_a = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, 
                                                        num_classes=num_classes, 
                                                        smoothing=smoothing, 
                                                        class_weights=class_weights)
        self.loss_p = MeanAbsoluteError(scale=beta,
                                        num_classes=num_classes, 
                                        smoothing=smoothing, 
                                        class_weights=class_weights)

