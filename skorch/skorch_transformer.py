"""
This file provides functionality for using a model from the Huggingface transformers package with skorch.
This is based on the following page in the skorch documentation: https://skorch.readthedocs.io/en/latest/user/huggingface.html
And the associated tutorial notebook: https://nbviewer.org/github/skorch-dev/skorch/blob/master/notebooks/Hugging_Face_Finetuning.ipynb
"""

import torch
import transformers
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, LRScheduler


class TransformerClassifierModule(torch.nn.Module):
    """
    torch module wrapper around the model returned by transformers, required to work with skorch.
    """
    def __init__(self, name, num_labels, problem_type):
        super().__init__()
        self.name = name
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.reset_weights()
        
    def reset_weights(self):
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels, problem_type = self.problem_type
        )
        
    def forward(self, **kwargs):
        # kwargs are passed from the wrapping NeuralNetClassifier, these may include sample_weight, which is not compatible with the inner model.
        if "sample_weight" in kwargs:
            del kwargs['sample_weight']
        pred = self.model(**kwargs)
        return pred.logits


class TransformerNeuralNetClassifier(NeuralNetClassifier):
    """
    Custom NeuralNetClassifier that can calculate a sample weighted loss.
    """
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        if "sample_weight" in X:
            sample_weight = X['sample_weight']
            loss_reduced = (sample_weight*loss_unreduced).mean()
        else:
            loss_reduced = loss_unreduced.mean()
        return loss_reduced


class AdaptingSchedulerCallback(Callback):
    """
    wrapper around the LRScheduler callback, allowing us to use it with a function closure.
    This is to make sure our learning rate scheduler adapts to changing model parameters (batch size) in a grid search.
    See skorch documentation on callbacks for more information.
    """
    def on_train_begin(self, net, X, **kwargs):
        n_samples = X["input_ids"].shape[0]
        self.schedule_closure = create_scheduler_lambda(n_samples, net.max_epochs, net.batch_size)
        self.scheduler = LRScheduler(torch.optim.lr_scheduler.LambdaLR, lr_lambda = self.schedule_closure, step_every="batch")
        self.scheduler.initialize()
        self.scheduler.on_train_begin(net, **kwargs)

    def on_batch_end(self, net, **kwargs):
        self.scheduler.on_batch_end(net, **kwargs)


def create_scheduler_lambda(n_samples:int, max_epochs:int, batch_size:int):
    """
    Creates a function closure for a linear scheduler, based on passed model parameters.
    """
    def scheduler_lambda(current_step):
        num_training_steps = max_epochs * (n_samples // batch_size + 1)
        factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
        assert factor >= 0
        return factor
    return scheduler_lambda
