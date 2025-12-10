import numpy as np
import torch

class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''
    def __init__(self, patience=1000, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, model_params):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_params)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                if self.counter % 10 == 0:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_params)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_params):
        '''Saves model when validation loss decreases, including hyperparameters.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        
        # Create the dictionary to save
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_params': model_params,
        }
        torch.save(checkpoint, self.path) # Save the dictionary
        self.val_loss_min = val_loss