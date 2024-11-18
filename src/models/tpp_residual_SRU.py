import torch
import torch.nn as nn
from .utils import clamp_preserve_gradients
from .batch import Batch
from models.log_norm_mix import LogNormalMixtureDistribution
from sru import SRU, SRUCell


#torch.set_default_tensor_type(torch.cuda.FloatTensor)

class RecurrentTPPResidual(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
            self,
            mean_log_inter_time: float = 0.0,
            std_log_inter_time: float = 1.0,
            context_size: int = 32,
            rnn_type: str = "GRU",
            num_features: int = 1,
            consider_survival: bool = False,
            num_mix_components: int = 64,
            consider_objid: bool = False,
            consider_objsize: bool = False,
    ):
        super().__init__()
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.num_features = num_features
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN

        # self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)
        self.rnn = SRU(input_size=self.num_features, hidden_size=self.context_size)  # , batch_first=True
        print("Using SRU ...")

        self.consider_survival = consider_survival
        self.num_mix_components = num_mix_components

        self.consider_objid = consider_objid
        self.consider_objsize = consider_objsize
        # to infer tao distribution
        # input is RNN hidden state
        self.mlp_input = self.context_size + 1
        if self.consider_objid:
            # input is [RNN hidden state, object ID]
            self.mlp_input = self.mlp_input + 1
        if self.consider_objsize:
            self.mlp_input = self.mlp_input + 1
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.context_size + 1, 3 * self.num_mix_components),
        #     nn.ReLU(),
        #     nn.Linear(3 * self.num_mix_components, 3 * self.num_mix_components)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input, 3 * self.num_mix_components),
            nn.ReLU(),
            nn.Linear(3 * self.num_mix_components, 3 * self.num_mix_components),
            nn.ReLU(),
            nn.Linear(3 * self.num_mix_components, 3 * self.num_mix_components)
        )

    def get_features(self, batch: Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.
        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).
        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
        """
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        return features  # (batch_size, seq_len, num_features)

    def get_age_features(self, batch: Batch) -> (torch.Tensor, torch.Tensor):
        ages = torch.rand(batch.inter_times.shape)
        ages = batch.inter_times - batch.inter_times * ages
        residual_times = batch.inter_times - ages
        # normalize ages.
        ages = torch.log(ages + 1e-8)
        ages = (ages - self.mean_log_inter_time) / self.std_log_inter_time
        return ages.unsqueeze(-1), residual_times  # (batch_size, seq_len, 1)

    def get_context(self, features: torch.Tensor, remove_last: bool = True) -> torch.Tensor:
        context = self.rnn(features.permute(1, 0, 2))[0].permute(1, 0, 2)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            batch_size, seq_len, context_size = context.shape
            context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
            context = context[:, :-1, :]
            context = torch.cat([context_init, context], dim=1)
        return context


    def get_residual_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        raw_params = self.mlp(context)  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )

    def log_prob(self, batch: Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.
        Args:
            batch:
        Returns:
            log_p: shape (batch_size,)
        """
        features = self.get_features(batch)  # (batch_size, seq_len, num_features)
        context = self.get_context(features)  # (batch_size, seq_len, context_size)
        ages, residual_times = self.get_age_features(batch)  # (batch_size, seq_len, 1)
        context = torch.cat([context, ages], dim=-1)
        if self.consider_objid:
            obj_ids = torch.log(batch.obj_ids + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
            obj_ids = (obj_ids - self.mean_log_inter_time) / self.std_log_inter_time
            context = torch.cat([context, obj_ids], dim=-1)
        if self.consider_objsize:
            obj_sizes = batch.obj_sizes.unsqueeze(-1)
            context = torch.cat([context, obj_sizes], dim=-1)

        residual_time_dist = self.get_residual_time_dist(context)  # (batch_size, seq_len)
        residual_times = residual_times.clamp(1e-10)
        log_p = residual_time_dist.log_prob(residual_times)
        log_p *= batch.mask  # (batch_size, seq_len)


        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        if self.consider_survival:
            last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
            log_surv_all = residual_time_dist.log_survival_function(residual_times)  # (batch_size, seq_len)
            log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)
            return log_p.sum(-1) + log_surv_last  # (batch_size,)

        return log_p.sum(-1)  # (batch_size,)

