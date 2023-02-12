import torch


def pfbeta(labels, predictions, beta: float = 1):
    """
    Official implementation of the evaluation metrics, pf1 Score,
    cf. https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview/evaluation
    """
    y_true_count = 0
    ctp = 0
    cfp = 0
    for idx in range(len(labels)):

        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    # Add if ever there is no true prediction to avoid divide by 0
    if y_true_count == 0:
        return 0

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


class CustomBCELoss(torch.nn.Module):
    def __init__(self, weight_fn=None):
        super(CustomBCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        if weight_fn is None:
            weight_fn = lambda x: 1
        self.weight_fn = weight_fn

    def forward(self, input, target):
        weight = self.weight_fn(target)
        loss = self.loss_fn(input, target)
        weighted_loss = weight * loss
        return weighted_loss.mean()
