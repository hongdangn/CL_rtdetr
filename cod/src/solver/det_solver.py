from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

# add clip embed and otod
from .mlp import MLP
from .desc_embed import get_desc_embed
from .uotod.match import BalancedSinkhorn
from .uotod.loss import DetectionLoss
from .uotod.loss import MultipleObjectiveLoss, GIoULoss, NegativeProbLoss
from torch.nn import L1Loss
import torch
#

from termcolor import cprint

class DetSolver(BaseSolver):

    def fit(
        self,
    ):
        self.train()

        args = self.cfg

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        task_idx = self.train_dataloader.dataset.task_idx
        data_ratio = self.train_dataloader.dataset.data_ratio

        # embed text with CLIP
        desc_embed = get_desc_embed(self.device).to(torch.float32)
        # print(desc_embed)
        enc_mlp = MLP(input_dim=512, hidden_dim=1024, output_dim=256, num_layers=2)
        enc_mlp.to(self.device)

        matching_method = BalancedSinkhorn(
            cls_match_module=NegativeProbLoss(reduction="none"),
            loc_match_module=MultipleObjectiveLoss(
                losses=[GIoULoss(reduction="none"), L1Loss(reduction="none")],
                weights=[1., 5.],
            ),
            background_cost=0.,  # Does not influence the matching when using balanced OT
        )
        desc_criterion = DetectionLoss(matching_method = matching_method)
        ######

        cprint(f"Task {task_idx} training...", "red", "on_yellow")

        for epoch in range(self.last_epoch + 1, args.epochs):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_one_epoch(
                self.model,
                enc_mlp, # mlp for embedding description
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                desc_criterion, # description criterion
                desc_embed, # description embedding
                args.clip_max_norm,
                ema=self.ema,
                scaler=self.scaler,
                task_idx=task_idx,
                data_ratio=data_ratio,
                pseudo_label=args.pseudo_label,
                distill_attn=args.distill_attn,
                teacher_path=args.teacher_path,
            )

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model

            ap = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
            )

            if self.output_dir:
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_path = (
                        self.output_dir
                        / f"{data_ratio}_t{task_idx}_{epoch+1}e_ap{round(ap, 2)}.pth"
                    )
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

        # # Generating Buffer with extra epoch
        # last_task = task_idx + 1 == args.total_tasks
        # if last_task == False and args.rehearsal:
        #     print(f"Model update for generating buffer list")
        #     self.rehearsal_classes = construct_replay_extra_epoch(
        #         args=self.args,
        #         Divided_Classes=self.Divided_Classes,
        #         model=self.model,
        #         criterion=self.criterion,
        #         device=self.device,
        #         rehearsal_classes=self.rehearsal_classes,
        #         task_num=task_idx,
        #     )
        #     print(f"complete save and merge replay's buffer process")
        #     print(f"next replay buffer list : {self.rehearsal_classes.keys()}")

    def val(
        self,
    ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model

        evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
        )
