import argparse
import pytorch_lightning as pl
import torch
from .util import yolo_loss
from torch.autograd import Variable

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100
x_sample = []
y_sample = []

class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if not loss in ("ctc", "transformer","yolo_loss"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        elif loss == "yolo_loss":
            self.loss_fn = yolo_loss

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        #For the very first batch, take the first image and store it as reference
        #global variables. This is to know if the weights are indeed changing.
        logits = self(x)
        loss,results,targets = self.loss_fn(logits, y,self.args["cuda"])
        self.log("train_loss", loss)
        sm = torch.nn.Softmax(dim=1)
        self.train_acc(sm(results[:,5:]), targets[:,4].long())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        loss = Variable(loss, requires_grad = True)
        output = {"loss":loss}
        return output

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss,results,targets = self.loss_fn(logits, y,self.args["cuda"])
        self.log("val_loss", loss, prog_bar=True)
        sm = torch.nn.Softmax(dim=1)
        #logits,y = getTensors(logits,y,self.args["cuda"])
        self.val_acc(sm(results[:,5:]), targets[:,4].long())
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss,results,targets = self.loss_fn(logits, y,self.args["cuda"])
        #logits,y = getTensors(logits,y,self.args["cuda"])
        sm = torch.nn.Softmax(dim=1)
        self.test_acc(sm(results[:,5:]), targets[:,4].long())
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
