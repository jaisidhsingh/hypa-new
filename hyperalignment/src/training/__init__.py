import torch
from torch.utils.flop_counter import FlopCounterMode
import math
import wandb
import numpy as np
from contextlib import suppress
import warnings
warnings.simplefilter("ignore")


class SeparateTrainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def train_one_epoch(self, model, loader, criterion, optimizer, scheduler, scaler, epoch):
        model.train()

        autocast = torch.amp.autocast if self.device == "cuda" else suppress
        logs = {"avg_loss": 0, "accuracy": 0}
        correct, total = 0, 0

        flop_counter = FlopCounterMode(model, display=True, depth=4)
        total_flops = None

        with flop_counter:
            for idx, (image_features, text_features) in enumerate(loader):
                step = int(epoch * len(loader)) + idx + 1
                batch_size = image_features.shape[0]

                image_features = image_features.float()
                image_features = image_features.view(batch_size, self.args.image_embed_dim)

                text_features = text_features.float().to(self.device)
                text_features = text_features.view(batch_size, self.args.text_embed_dim)

                if scheduler is not None:
                    scheduler(step)

                optimizer.zero_grad()

                with autocast(self.device):
                    mapped_text_features = model(text_features)
                    mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)
                    loss, in_batch_corrects = criterion.compute_loss_and_accuracy(
                        model.logit_scale,
                        image_features,
                        mapped_text_features
                    )
                    logs["avg_loss"] += loss.item()

                correct += in_batch_corrects
                total += batch_size
                accuracy = round(correct/total * 100, 2)
                logs["accuracy"] = accuracy

                if self.args.use_wandb:
                    wandb.log({"loss": loss.item(), "accuracy": accuracy})

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100))
                    
                del image_features
                del text_features
                del mapped_text_features

            logs["avg_loss"] /= idx+1

        total_flops = flop_counter.get_total_flops()
        return logs, total_flops

    @torch.no_grad()
    def val_one_epoch(self, model, loader, criterion):
        model.eval()

        autocast = torch.cuda.amp.autocast if self.device == "cuda" else suppress
        logs = {"avg_loss": 0, "accuracy": 0}
        correct, total = 0, 0

        for idx, (image_features, text_features) in enumerate(loader):
            batch_size = image_features.shape[0]

            image_features = image_features.float()
            image_features = image_features.view(batch_size, self.args.image_embed_dim)

            text_features = text_features.float().to(self.device)
            text_features = text_features.view(batch_size, self.args.text_embed_dim)

            with autocast(self.device):
                mapped_text_features = model(text_features)
                mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)
                loss, in_batch_corrects = criterion.compute_loss_and_accuracy(
                    model.logit_scale,
                    image_features,
                    mapped_text_features
                )
                logs["avg_loss"] += loss.item()

            correct += in_batch_corrects
            total += batch_size
            accuracy = round(correct/total * 100, 2)
            logs["accuracy"] = accuracy

            del image_features
            del text_features
            del mapped_text_features

        logs["avg_loss"] /= idx+1
        return logs


class JointTrainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def train_one_epoch(self, hnet, main_model, loader, criterion, optimizer, scheduler, scaler, epoch):
        hnet.train()
        corrects = {}
        total = 0
        loss = 0
        logit_scale = torch.tensor(np.log(self.args.logit_scale)).to(self.device)
        autocast = torch.cuda.amp.autocast if self.device == "cuda" else suppress

        for idx, (image_features, text_features) in enumerate(loader):
            image_features = image_features.float().to(self.device)
            text_features = text_features.squeeze(1).float().to(self.device)

            batch_size = image_features.shape[0]
            dim = image_features.shape[-1]
            N = image_features.shape[1]

            if scheduler is not None:
                step = epoch * len(loader) + (idx+1)
                scheduler(step)

            optimizer.zero_grad()
            total_loss = 0

            with autocast():
                params = hnet(cond_id=[i for i in range(N)])
                for j in range(N):
                    mapped_text_features = main_model(text_features, weights=params[j]).view(batch_size, dim)
                    mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)
                    per_param_loss, in_batch_corrects = criterion.compute_loss_and_accuracy(
                        logit_scale,
                        image_features[:, j, :].view(batch_size, dim),
                        mapped_text_features
                    )

                    total_loss += per_param_loss
                    if j not in corrects:
                        corrects[j] = 0
                    corrects[j] += in_batch_corrects

            total += batch_size
            loss += total_loss.item() / N
            accuracies = [round(corr/total * 100, 2) for corr in corrects.values()]

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss /= idx+1
        return {"avg_loss": loss, "accuracies": accuracies}

    @torch.no_grad()
    def val_one_epoch(self, hnet, main_model, loader, criterion, params=None):
        if params is None:
            hnet.eval()

        corrects = {}
        total = 0
        loss = 0
        logit_scale = torch.tensor(np.log(self.args.logit_scale)).to(self.device)
        autocast = torch.cuda.amp.autocast if self.device == "cuda" else suppress

        for idx, (image_features, text_features) in enumerate(loader):
            image_features = image_features.float().to(self.device)
            text_features = text_features.squeeze(1).float().to(self.device)

            batch_size = image_features.shape[0]
            N = image_features.shape[1]
            total_loss = 0

            with autocast():
                if params is None:
                    params = hnet(cond_id=[i for i in range(N)])

                for j in range(N):
                    mapped_text_features = main_model(text_features, weights=params[j])
                    mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)
                    per_param_loss, in_batch_corrects = criterion.compute_loss_and_accuracy(
                        logit_scale,
                        image_features[:, j, :],
                        mapped_text_features
                    )

                    total_loss += per_param_loss.item()
                    if j not in corrects:
                        corrects[j] = 0
                    corrects[j] += in_batch_corrects

            total += batch_size
            loss += total_loss / N
            accuracies = [round(corr/total * 100, 2) for corr in corrects.values()]

        loss /= idx+1
        return {"avg_loss": loss, "accuracies": accuracies}

"""
@torch.no_grad()
def val_joint_mapper_over_loader(args, model, params, loader, criterion):
    model.eval()

    autocast = torch.cuda.amp.autocast if args.device == "cuda" else suppress
    logs = {"avg_loss": 0, "accuracy": 0}
    correct, total = 0, 0

    for idx, (image_features, text_features) in enumerate(loader):
        batch_size = image_features.shape[0]

        image_features = image_features.float()
        image_features = image_features.view(batch_size, args.image_embed_dim)

        text_features = text_features.float().to(args.device)
        text_features = text_features.view(batch_size, args.text_embed_dim)

        with autocast():
            mapped_text_features = model(text_features, weights=params)
            loss, in_batch_corrects = criterion.compute_loss_and_accuracy(
                torch.tensor(np.log(100)).to(args.device),
                image_features,
                mapped_text_features
            )
            logs["avg_loss"] += loss.item()

        correct += in_batch_corrects
        total += batch_size
        accuracy = round(correct/total * 100, 2)
        logs["accuracy"] = accuracy

        del image_features
        del text_features
        del mapped_text_features

    logs["avg_loss"] /= idx+1
    return logs
"""
