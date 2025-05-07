import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.data import Subset, DataLoader
from datasets import IndexedDataset, SubsetGenerator
from utils import GradualWarmupScheduler, Adahessian


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CRESTTrainer:
    def __init__(
        self, 
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        """
        Consolidated CRESTTrainer class
        :param args: arguments
        :param model: model to train
        :param train_dataset: training dataset
        :param val_loader: validation data loader
        :param train_weights: weights for the training data
        """
        self.args = args
        self.model = model

        # if more than one GPU is available, use DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.args.device)

        self.train_dataset = train_dataset
        self.val_loader = val_loader
        
        # Initialize train weights
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = torch.ones(len(self.train_dataset))
        self.train_weights = self.train_weights.to(self.args.device)

        # the default optimizer is SGD
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_milestones,
            last_epoch=-1,
            gamma=args.gamma,
        )

        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1,
            total_epoch=args.warm_start_epochs,
            after_scheduler=lr_scheduler,
        )

        self.train_criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)
        self.val_criterion = nn.CrossEntropyLoss().to(args.device)

        # Performance metrics
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.val_loss = 0
        self.val_acc = 0

        # Timing metrics
        self.batch_data_time = AverageMeter()
        self.batch_forward_time = AverageMeter()
        self.batch_backward_time = AverageMeter()
        
        # CREST specific initialization
        self.train_target = np.array(self.train_dataset.dataset.targets)
        self.train_indices = np.arange(len(self.train_dataset))
        self.subset_generator = SubsetGenerator(greedy=(args.selection_method!="rand"), smtk=args.smtk)
        self.num_selection = 0
        
        self.steps_per_epoch = np.ceil(int(len(self.train_dataset) * self.args.train_frac) / self.args.batch_size).astype(int)
        self.reset_step = self.steps_per_epoch
        self.random_sets = np.array([])
        
        # Initialize subset and subset_weights
        self.subset = np.array([])
        self.subset_weights = np.array([])

        # Initialize train loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        self.num_checking = 0
        self.gradient_approx_optimizer = Adahessian(self.model.parameters())
        self.loss_watch = np.ones((self.args.watch_interval, len(self.train_dataset))) * -1

        self.approx_time = AverageMeter()
        self.compare_time = AverageMeter()
        self.similarity_time = AverageMeter()

        self.deltas = []
        self.w_previous = None
        self.current_epoch = 0
        self.current_step = 0

    def train(self):
        """
        Train the model
        """
        # load checkpoint if resume is True
        if self.args.resume_from_epoch > 0:
            self._load_checkpoint(self.args.resume_from_epoch)

        for epoch in range(self.args.resume_from_epoch, self.args.epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)
            self._log_epoch(epoch)

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "val_loss": self.val_loss,
                    "val_acc": self.val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                
            self.lr_scheduler.step()

            if (epoch+1) % self.args.save_freq == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint()

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch
        :param epoch: current epoch
        """
        self.current_epoch = epoch
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):
            if (training_step > self.reset_step) and ((training_step - self.reset_step) % self.args.check_interval == 0):
                self._check_approx_error(epoch, training_step)

            if training_step == self.reset_step:
                self._select_subset(epoch, training_step)
                self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                self._get_quadratic_approximation(epoch, training_step)
            elif training_step == 0:
                self.train_iter = iter(self.train_loader)

            data_start = time.time()
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, data_idx = batch
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)
            self.current_step += 1
            
            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            data_start = time.time()

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_step": training_step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc})

    def _val_epoch(self, epoch):
        """
        Validate the model on the validation set
        :param epoch: current epoch
        """
        self.model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for _, (data, target, _) in enumerate(self.val_loader):
                data, target = data.cuda(), target.cuda()

                output = self.model(data)

                loss = self.val_criterion(output, target)

                val_loss += loss.item() * data.size(0)
                val_acc += (output.argmax(dim=1) == target).float().sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_acc /= len(self.val_loader.dataset)

        self.val_loss = val_loss
        self.val_acc = val_acc

    def _forward_and_backward(self, data, target, data_idx):
        """
        Forward and backward pass for training
        :param data: input data
        :param target: target data
        :param data_idx: indices of data
        :return: loss and accuracy
        """
        self.optimizer.zero_grad()

        # Save w_{t-1} (previous weights) at every step
        if self.w_previous is None:
            self.w_previous = [p.clone().detach() for p in self.model.parameters()]
        else:
            # Compute delta (w_t - w_{t-1}) and store the change
            delta = []
            for p, prev_p in zip(self.model.parameters(), self.w_previous):
                delta.append(p - prev_p)  # Actual delta (weight change)
            
            self.deltas.append(delta)  # Store the delta for tracking

            if self.args.use_wandb:  # Log only if WandB is enabled
                # Create a dictionary of deltas to log
                delta_values = {}
                for i, d in enumerate(delta):
                    delta_values[f"delta_param_{i}_mean"] = d.mean().item()

                wandb.log({
                    "epoch": self.current_epoch,
                    "step": self.current_step,
                    **delta_values
                })
            
            # Update w_previous to the current weights
            self.w_previous = [p.clone().detach() for p in self.model.parameters()]

        # Train model with the current batch and record forward and backward time
        forward_start = time.time()
        output = self.model(data)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)
        
        # Calculate loss
        loss = self.train_criterion(output, target)
        loss = (loss * self.train_weights[data_idx]).mean()

        # Update delta with gradient
        lr = self.lr_scheduler.get_last_lr()[0]
        if lr > 0:
            # Compute the parameter change delta
            self.model.zero_grad()
            # Approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)                   
            self.delta -= lr * gf_current

        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # Update training loss and accuracy
        train_acc = (output.argmax(dim=1) == target).float().mean().item()
        self.train_loss.update(loss.item(), data.size(0))
        self.train_acc.update(train_acc, data.size(0))

        return loss, train_acc

    def _get_quadratic_approximation(self, epoch: int, training_step: int):
        """
        Compute the quadratic approximation of the loss function
        :param epoch: current epoch
        :param training_step: current training step
        """
        if self.args.approx_with_coreset:
            # Update the second-order approximation with the coreset
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.subset),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )
        else:
            # Update the second-order approximation with random subsets
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.random_sets),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )

        approx_start = time.time()
        curvature_norm = AverageMeter()
        self.start_loss = AverageMeter()
        for approx_batch, (input, target, idx) in enumerate(approx_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target

            # compute output
            output = self.model(input_var)
                
            if self.args.approx_with_coreset:
                loss = self.train_criterion(output, target_var)
                batch_weight = self.train_weights[idx.long()]
                loss = (loss * batch_weight).mean()
            else:
                loss = self.val_criterion(output, target_var)
            self.model.zero_grad()

            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)

            if approx_batch == 0:
                self.gf = gf_tmp * len(idx)
                self.ggf = ggf_tmp * len(idx)
                self.ggf_moment = ggf_tmp_moment * len(idx)
            else:
                self.gf += gf_tmp * len(idx)
                self.ggf += ggf_tmp * len(idx)
                self.ggf_moment += ggf_tmp_moment * len(idx)

            curvature_norm.update(ggf_tmp_moment.norm())
            self.start_loss.update(loss.item(), input.size(0))

        approx_time = time.time() - approx_start
        self.approx_time.update(approx_time)

        self.gf /= len(approx_loader.dataset)
        self.ggf /= len(approx_loader.dataset)
        self.ggf_moment /= len(approx_loader.dataset)
        self.delta = 0

        gff_norm = curvature_norm.avg
        self.start_loss = self.start_loss.avg
        if self.args.approx_moment:
            self.ggf = self.ggf_moment

        if training_step == self.steps_per_epoch:
            self.init_curvature_norm = gff_norm 
        else:
            self.args.check_interval = int(torch.ceil(self.init_curvature_norm / gff_norm * self.args.interval_mul))
            self.args.num_minibatch_coreset = min(self.args.check_interval * self.args.batch_num_mul, self.steps_per_epoch)
        self.args.logger.info(f"Checking interval {self.args.check_interval}. Number of minibatch coresets {self.args.num_minibatch_coreset}")
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'ggf_norm': gff_norm,
                'check_interval': self.args.check_interval,
                'num_minibatch_coreset': self.args.num_minibatch_coreset})

    def _check_approx_error(self, epoch: int, training_step: int):
        """
        Check the approximation error of the current batch
        :param epoch: current epoch
        :param training_step: current training step
        """
        # Implementation can be added here if needed
        pass

    def _get_train_output(self):
        """
        Evaluate the model on the training set and record the output and softmax
        """
        self.model.eval()

        self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
        self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))

        with torch.no_grad():
            for _, (data, _, data_idx) in enumerate(self.train_val_loader):
                data = data.to(self.args.device)

                output = self.model(data)

                self.train_output[data_idx] = output.cpu().numpy()
                self.train_softmax[data_idx] = output.softmax(dim=1).cpu().numpy()

        self.model.train()

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        :param epoch: current epoch
        :param training_step: current training step
        """
        self.num_selection += 1

        if self.args.selection_method == "rand":
            # Random selection
            self.subset = np.random.choice(
                self.train_indices,
                size=int(len(self.train_dataset) * self.args.train_frac),
                replace=False
            )
            self.subset_weights = np.ones(len(self.subset))
        else:
            # Use subset generator for other selection methods
            self._get_train_output()
            self.subset, self.subset_weights, _, _ = self.subset_generator.generate_subset(
                self.train_output,
                epoch,
                int(len(self.train_dataset) * self.args.train_frac),
                self.train_indices,
                self.train_target,
                mode="dense"
            )

        if self.args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "training_step": training_step,
                "num_selection": self.num_selection,
                "subset_size": len(self.subset)
            })

    def _update_train_loader_and_weights(self):
        """
        Update train loader and weights with selected subset
        """
        self.args.logger.info("Updating train loader and weights with subset of size {}".format(len(self.subset)))
        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.subset),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_weights = np.zeros(len(self.train_dataset))
        self.subset_weights = self.subset_weights / np.sum(self.subset_weights) * len(self.subset)
        self.train_weights[self.subset] = self.subset_weights
        self.train_weights = torch.from_numpy(self.train_weights).float().to(self.args.device)

    def _save_checkpoint(self, epoch=None):
        """
        Save model checkpoint
        :param epoch: current epoch
        """
        if epoch is not None:
            save_path = self.args.save_dir + "/model_epoch_{}.pt".format(epoch)
        else:
            save_path = self.args.save_dir + "/model_final.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.train_loss.avg,
                "train_acc": self.train_acc.avg,
                "val_loss": self.val_loss,
                "val_acc": self.val_acc,
                "args": self.args,
            }, 
            save_path)
        
        self.args.logger.info("Checkpoint saved to {}".format(save_path))
        
    def _load_checkpoint(self, epoch):
        """
        Load model checkpoint
        :param epoch: epoch to load from
        """
        save_path = self.args.save_dir + "/model_epoch_{}.pt".format(epoch)
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.train_acc = checkpoint["train_acc"]

    def _log_epoch(self, epoch):
        """
        Log epoch statistics
        :param epoch: current epoch
        """
        self.args.logger.info(
            "Epoch: {}\tTrain Loss: {:.6f}\tTrain Acc: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {:.6f}".format(
                epoch,
                self.train_loss.avg,
                self.train_acc.avg,
                self.val_loss,
                self.val_acc,
            )
        )

    def _reset_metrics(self):
        """Reset metrics for a new epoch"""
        self.train_loss.reset()
        self.train_acc.reset()
        self.batch_data_time.reset()
        self.batch_forward_time.reset()
        self.batch_backward_time.reset()

    def get_model(self):
        """Get the model"""
        return self.model

    def get_train_loss(self):
        """Get training loss"""
        return self.train_loss.avg

    def get_train_acc(self):
        """Get training accuracy"""
        return self.train_acc.avg

    def get_val_loss(self):
        """Get validation loss"""
        return self.val_loss

    def get_val_acc(self):
        """Get validation accuracy"""
        return self.val_acc

    def get_train_time(self):
        """Get training time statistics"""
        return {
            "data_time": self.batch_data_time.avg,
            "forward_time": self.batch_forward_time.avg,
            "backward_time": self.batch_backward_time.avg,
        }