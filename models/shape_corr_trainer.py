"""Shape correspondence template."""
from argparse import Namespace
import collections
from models.correspondence_utils import square_distance
from utils.tensor_utils import to_numpy
from data.point_cloud_db.surreal import BigRandomSampler

from data.point_cloud_db.point_cloud_dataset import PointCloudDataset, matrix_map_from_corr_map
from models.metrics.metrics import AccuracyAssumeEye

import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from torch import Tensor

from utils import argparse_init, switch_functions


class ShapeCorrTemplate(LightningModule):
    """
    This is a template for future shape correspondence templates using pytorch lightning.

    """

    def __init__(self, hparams, **kwargs):
        """Stub."""
        super(ShapeCorrTemplate, self).__init__()
        load_hparams = vars(hparams) if isinstance(hparams,Namespace) else hparams
        for k,v in load_hparams.items():
            setattr(self.hparams,k,v)

        self.train_accuracy = AccuracyAssumeEye()
        self.val_accuracy = AccuracyAssumeEye()
        self.test_accuracy = AccuracyAssumeEye()
        self.losses = {}
        self.tracks = {}

        
    def setup(self, stage):
        (self.train_dataset, self.val_dataset, self.test_dataset,) = switch_functions.load_dataset(self.hparams)

    def forward(self, data) -> Tensor:
        """Stub."""
        raise NotImplementedError()


    def training_step(self, batch, batch_idx, mode="train"):
        """
        Lightning calls this inside the training loop with the 
        data from the training dataloader passed in as `batch`.
        """
        self.losses = {}
        self.tracks = {}
        self.hparams.batch_idx = batch_idx
        self.hparams.mode = mode
        self.batch = batch

        # forward pass
        # self.log_weights_norm()
        batch = self(batch)


        if len(self.losses) > 0:
            loss = sum(self.losses.values()).mean()
            self.tracks[f"{mode}_tot_loss"] = loss
        else:
            loss = None

        all = {k: to_numpy(v) for k, v in {**self.tracks, **self.losses}.items()}
        getattr(self, f"{mode}_logs", None).append(all)

        if (batch_idx % (self.hparams.log_every_n_steps if self.hparams.mode != 'test' else 1) == 0):
            for k, v in all.items():
                self.logger.experiment.add_scalar(f"{k}/step", v,self.global_step)

        if self.vis_iter():
            self.visualize(batch, mode=mode)

        output = collections.OrderedDict({"loss": loss})
        return output

    def vis_iter(self,):
        return (self.hparams.batch_idx % eval(f"self.hparams.{self.hparams.mode}_vis_interval") == 0) and self.hparams.show_vis

    def visualize(self, batch, mode="train"):
        """
        Here we perform visualizations.
        """

    def validation_step(self, batch, batch_idx, mode="val"):
        """Lightning calls this inside the validation loop with the data from the validation dataloader passed in as `batch`."""
        return self.training_step(batch, batch_idx, mode=mode)

    def log_test_step(self):
        logs_step = {k: to_numpy(v) for k, v in {**self.tracks}.items()}
        getattr(self, f"test_logs", None).append(logs_step)

        self.log_dict({f"test/{k}": v for k, v in self.tracks.items()}, on_step=False, on_epoch=True)


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        """
        self.optimizer = switch_functions.choose_optimizer(self.hparams, self.parameters())
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (1 - epoch / self.hparams.max_epochs))
        return self.optimizer  # , [self.optimizer],[self.scheduler]

    def dataloader(self, dataset, mode):
        """
        Returns the relevant dataloader (called once per training).
        
        Args:
            train_val_test (str, optional): Define which dataset to choose from. Defaults to 'train'.
        
        Returns:
            Dataset
        """
        # init data generators
        if self.hparams.task_name == "shape_corr":
            # sampler = RandomSampler(dataset, replacement=False) if mode == 'train' else None # shape corr means dataset = N**2, replacment=False is too slow (but for a small dataset it is ok)
            sampler = None if self.hparams.dataset_name != 'surreal' else BigRandomSampler(dataset, replacement=True)
            loader = switch_functions.get_dataloader(self.hparams.task_name, self.hparams)(
                dataset=dataset,
                batch_size=getattr(self.hparams, f"{mode}_batch_size", 1),
                num_workers=self.hparams.num_data_workers,
                shuffle=True if mode == 'train' and sampler is None else False,
                drop_last=True if mode == 'train' else False,
                sampler=sampler,
            )
        else:
            raise Exception("No match for task_name in load_dataset")

        return loader

    def prepare_data(self):
        """
        Here we download the data, called once(for all gpus).
        The definition of the datasets happens in the setup()

        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        """

    def train_dataloader(self):
        """Stub."""
        log.info("Training data loader called.")
        return self.dataloader(self.train_dataset,mode='train')

    def val_dataloader(self):
        """Stub."""
        log.info("Validation data loader called.")
        return self.dataloader(self.val_dataset,mode='val')

    def test_dataloader(self):
        """Stub."""
        log.info("Test data loader called.")
        return self.dataloader(self.test_dataset,mode='test')
    
    def predict_dataloader(self):
        """Stub."""
        log.info("Test data loader called.")
        return self.dataloader(self.test_dataset,mode='test')

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx, mode="test")

        return output

    def on_validation_epoch_start(self):
        self.val_logs = []
        self.hparams.mode = 'val'

    def on_train_epoch_start(self):
        self.train_logs = []
        self.hparams.mode = 'train' 
        self.hparams.current_epoch = self.current_epoch

    def on_test_epoch_start(self):
        self.test_logs = []
        self.hparams.mode = 'test'
    
    def on_epoch_end_generic(self):
        logs = getattr(self, f"{self.hparams.mode}_logs", None)
        dict_of_lists = {k: [dic[k] for dic in logs] for k in logs[0]}
        for key, lst in dict_of_lists.items():
            s = 0
            for item in lst:
                s += item.sum()
            name = f"{self.hparams.mode}/{key}/epoch"
            val = s / len(lst)
            self.tracks[name] = val

            self.logger.experiment.add_scalar(name, val, self.current_epoch)


        return dict_of_lists

    def on_train_epoch_end(self, outputs) -> None:
        self.on_epoch_end_generic()

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end_generic()



    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        """
        Static function to add all arguments that are relevant only for this module

        Example: 
        parser = argparse_init.default_arg_parser("some description", parents=[parent_parser])
        parser.set_defaults(learning_rate=1e-3)
        parser.add_argument('--unetappnp_depth', default=3,help="number of encoder steps")
        parser = APPNP.APPNP.add_model_specific_args(parser)

        pool_type = parser.parse_known_args()[0].pool_type
        pool_layer = switch_functions.choose_pool_func(pool_type)
        parser = pool_layer.add_model_specific_args(parser)

        Args:
            parent_parser (ArgparseManager): The general argparser
        
        Returns:
        ArgparseManager : The new argparser
        """
        parser = argparse_init.default_arg_parser(parents=[parent_parser], is_lowest_leaf=False)


        parser.add_argument(
            "--similarity_init",
            default="cosine",
            # choices=["multiplication", "cosine", "difference_inverse", "difference_max_norm",],
            help="define how to mesure the distance between vectors",
        )
        parser.add_argument(
            "--p_normalization",
            default="l2",
            choices=["l1", "l2", "softmax", "no_normalize"],
            help="The way to normalize the P matrix",
        )
        parser.add_argument(
            "--P_pow_factor", default=2, help="The factor to power the P matrix",
        )
        parser.add_argument(
            "--sim_normalization",
            default="softmax",
            choices=["l1", "l2", "softmax", "no_normalize"],
            help="The way to normalize neighbors similarity",
        )

        return parser

    @staticmethod
    def extract_labels_for_test(test_batch):
        data_dict = test_batch
        if('label_flat' in test_batch):
            label = data_dict['label_flat'].squeeze(0)
            pinput1 = data_dict['src_flat']
            input2 = data_dict['tgt_flat']
        else:
            pinput1 = data_dict['source']['pos']
            input2 = data_dict['target']['pos']
            label = matrix_map_from_corr_map(data_dict['gt_map'].squeeze(0),pinput1.squeeze(0),input2.squeeze(0))


        ratio_list, soft_labels = PointCloudDataset.extract_soft_labels_per_pair(
            label,
            input2.squeeze(0)
        )
        
        return label,pinput1,input2,ratio_list,soft_labels

    @staticmethod
    def compute_acc(label, ratio_list, soft_labels, p,input2,track_dict={},hparams=Namespace()):
        corr_tensor = ShapeCorrTemplate._prob_to_corr_test(p)

        hit = label.argmax(-1).squeeze(0)
        pred_hit = p.squeeze(0).argmax(-1)
        target_dist = square_distance(input2.squeeze(0), input2.squeeze(0)) 
        track_dict["acc_mean_dist"] = target_dist[pred_hit,hit].mean().item()
        if(getattr(hparams,'dataset_name','') == 'tosca' or (hparams.mode == 'test' and hparams.test_on_tosca)):
            track_dict["acc_mean_dist"] /= 3 # TOSCA is not scaled to meters as the other datasets. /3 scales the shapes to be coherent with SMAL (animals as well)


        acc_000 = ShapeCorrTemplate._label_ACC_percentage_for_inference(corr_tensor, label.unsqueeze(0))
        track_dict["acc_0.00"] = acc_000
        for idx,ratio in enumerate(ratio_list):
            track_dict["acc_" + str(ratio)] = ShapeCorrTemplate._label_ACC_percentage_for_inference(corr_tensor, soft_labels[f"{ratio}"].unsqueeze(0)).item()
        return track_dict

    @staticmethod
    def _label_ACC_percentage_for_inference(label_in, label_gt):
        assert (label_in.shape == label_gt.shape)
        bsize = label_in.shape[0]
        b_acc = []
        for i in range(bsize):
            element_product = torch.mul(label_in[i], label_gt[i])
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1)  # N1x1

            hit = (sum_row != 0).sum()
            acc = hit.float() / torch.tensor(N1).float()
            b_acc.append(acc * 100.0)
        mean = torch.mean(torch.stack(b_acc))
        return mean

    @staticmethod
    def _prob_to_corr_test(prob_matrix):
        c = torch.zeros_like(prob_matrix)
        idx = torch.argmax(prob_matrix, dim=2, keepdim=True)
        for bsize in range(c.shape[0]):
            for each_row in range(c.shape[1]):
                c[bsize][each_row][idx[bsize][each_row]] = 1.0

        return c

