import torch
import torch.nn as nn
from dpp.utils import clamp_preserve_gradients
from .utils import clamp_preserve_gradients
from .batch import Batch
from models.log_norm_mix import LogNormalMixtureDistribution

# using LRB features as input to density network to learn the distribution of a LRB label doesn't work.

class FeatureDensityNetwork(nn.Module):
    """
    Density Network to model residual time distribution based on input feature.

    Args:
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}
    """
    def __init__(
        self,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        rnn_type: str = "GRU",
        use_rnn: bool = False,
        num_mix_components: int = 64,
    ):
        super().__init__()
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.use_rnn = use_rnn
        if self.use_rnn:
            self.rnn_type = rnn_type
            self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN
            self.rnn = getattr(nn, rnn_type)(input_size=1, hidden_size=self.context_size, batch_first=True)
        else:
            self.rnn = None
        self.num_mix_components = num_mix_components
        self.mlp = nn.Sequential(
            nn.Linear(self.context_size, 3 * self.num_mix_components),
            nn.ReLU(),
            nn.Linear(3 * self.num_mix_components, 3 * self.num_mix_components)
        )

    def get_train_features(self, batch: Batch) -> (torch.Tensor, torch.Tensor):
        """
        Convert each event in a sequence into a feature vector.
        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).
        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
        """
        # LRB featues: [past 31 inter-arrival times, current age, next residual time]
        features = torch.log(batch.inter_times[:, :-1] + 1e-8)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        residual_times = batch.inter_times[:, -1]
        return features, residual_times  # (batch_size, seq_len, num_features)

    def get_test_features(self, batch: Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.
        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).
        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
        """
        # LRB featues: [past 31 inter-arrival times, current age, next residual time]
        features = torch.log(batch.inter_times + 1e-8)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]
        context = context[:, -1, :]
        return context # (batch_size, context_size)

    def get_residual_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.
        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)
        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)
        """
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
        features, residual_times = self.get_train_features(batch)
        if self.use_rnn:
            features = self.get_context(features)
        residual_time_dist = self.get_residual_time_dist(features)
        residual_times = residual_times.clamp(1e-10)
        log_p = residual_time_dist.log_prob(residual_times)  # (batch_size, seq_len)

        return log_p.sum(-1)  # (batch_size,)
