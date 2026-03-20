from typing import List, Optional, Tuple, Dict
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.layers import LinearEncoder, Initialization
import numpy as np

from mlagents.trainers.torch_entities.encoders import (
    SimpleVisualEncoder,
    ResNetVisualEncoder,
    NatureVisualEncoder,
    SmallVisualEncoder,
    FullyConnectedVisualEncoder,
    VectorInput,
)
from mlagents.trainers.settings import EncoderType, ScheduleType
from mlagents.trainers.torch_entities.attention import EntityEmbedding, ResidualSelfAttention
from mlagents.trainers.exception import UnityTrainerException
from mlagents_envs.base_env import ObservationSpec, DimensionProperty


class ModelUtils:
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.FULLY_CONNECTED: 1,
        EncoderType.MATCH3: 5,
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    VALID_VISUAL_PROP = frozenset(
        [
            (
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
                DimensionProperty.NONE,
            ),
            (
                DimensionProperty.NONE,
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
            ),
            (DimensionProperty.UNSPECIFIED,) * 3,
        ]
    )

    VALID_VECTOR_PROP = frozenset(
        [(DimensionProperty.NONE,), (DimensionProperty.UNSPECIFIED,)]
    )

    VALID_VAR_LEN_PROP = frozenset(
        [(DimensionProperty.VARIABLE_SIZE, DimensionProperty.NONE)]
    )

    @staticmethod
    def update_learning_rate(optim: torch.optim.Optimizer, lr: float) -> None:
        for param_group in optim.param_groups:
            param_group["lr"] = lr

    class DecayedValue:
        def __init__(
            self,
            schedule: ScheduleType,
            initial_value: float,
            min_value: float,
            max_step: int,
        ):
            self.schedule = schedule
            self.initial_value = initial_value
            self.min_value = min_value
            self.max_step = max_step

        def get_value(self, global_step: int) -> float:
            if self.schedule == ScheduleType.CONSTANT:
                return self.initial_value
            elif self.schedule == ScheduleType.LINEAR:
                return ModelUtils.polynomial_decay(
                    self.initial_value, self.min_value, self.max_step, global_step
                )
            else:
                raise UnityTrainerException(f"The schedule {self.schedule} is invalid.")

    @staticmethod
    def polynomial_decay(
        initial_value: float,
        min_value: float,
        max_step: int,
        global_step: int,
        power: float = 1.0,
    ) -> float:
        global_step = min(global_step, max_step)
        decayed_value = (initial_value - min_value) * (
            1 - float(global_step) / max_step
        ) ** (power) + min_value
        return decayed_value

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> nn.Module:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: SimpleVisualEncoder,
            EncoderType.NATURE_CNN: NatureVisualEncoder,
            EncoderType.RESNET: ResNetVisualEncoder,
            EncoderType.MATCH3: SmallVisualEncoder,
            EncoderType.FULLY_CONNECTED: FullyConnectedVisualEncoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(encoder_type)

    @staticmethod
    def _check_resolution_for_encoder(
        height: int, width: int, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    @staticmethod
    def get_encoder_for_obs(
        obs_spec: ObservationSpec,
        normalize: bool,
        h_size: int,
        attention_embedding_size: int,
        vis_encode_type: EncoderType,
    ) -> Tuple[nn.Module, int]:
        shape = obs_spec.shape
        dim_prop = obs_spec.dimension_property

        # VISUAL
        if dim_prop in ModelUtils.VALID_VISUAL_PROP:
            visual_encoder_class = ModelUtils.get_encoder_for_type(vis_encode_type)
            h, w, c = shape[0], shape[1], shape[2]
            ModelUtils._check_resolution_for_encoder(h, w, vis_encode_type)
            return (visual_encoder_class(h, w, c, h_size), h_size)

        # VECTOR
        if dim_prop in ModelUtils.VALID_VECTOR_PROP:
            return (VectorInput(shape[0], normalize), shape[0])

        # VARIABLE LENGTH
        if dim_prop in ModelUtils.VALID_VAR_LEN_PROP:
            return (
                EntityEmbedding(
                    entity_size=shape[1],
                    entity_num_max_elements=shape[0],
                    embedding_size=attention_embedding_size,
                ),
                0,
            )

        # OTHER
        raise UnityTrainerException(f"Unsupported Sensor with specs {obs_spec}")

    @staticmethod
    def create_input_processors(
        observation_specs: List[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        attention_embedding_size: int,
        normalize: bool = False,
    ) -> Tuple[nn.ModuleList, List[int]]:
        encoders: List[nn.Module] = []
        embedding_sizes: List[int] = []
        for obs_spec in observation_specs:
            encoder, embedding_size = ModelUtils.get_encoder_for_obs(
                obs_spec, normalize, h_size, attention_embedding_size, vis_encode_type
            )
            encoders.append(encoder)
            embedding_sizes.append(embedding_size)

        x_self_size = sum(embedding_sizes)
        if x_self_size > 0:
            for enc in encoders:
                if isinstance(enc, EntityEmbedding):
                    enc.add_self_embedding(attention_embedding_size)
        return (nn.ModuleList(encoders), embedding_sizes)

    @staticmethod
    def list_to_tensor(
        ndarray_list: List[np.ndarray], dtype: Optional[torch.dtype] = torch.float32
    ) -> torch.Tensor:
        return torch.as_tensor(np.asanyarray(ndarray_list), dtype=dtype)

    @staticmethod
    def list_to_tensor_list(
        ndarray_list: List[np.ndarray], dtype: Optional[torch.dtype] = torch.float32
    ) -> torch.Tensor:
        return [
            torch.as_tensor(np.asanyarray(_arr), dtype=dtype) for _arr in ndarray_list
        ]

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def break_into_branches(
        concatenated_logits: torch.Tensor, action_size: List[int]
    ) -> List[torch.Tensor]:
        action_idx = [0] + list(np.cumsum(action_size))
        branched_logits = [
            concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        return branched_logits

    @staticmethod
    def actions_to_onehot(
        discrete_actions: torch.Tensor, action_size: List[int]
    ) -> List[torch.Tensor]:
        onehot_branches = [
            torch.nn.functional.one_hot(_act.T, action_size[i]).float()
            for i, _act in enumerate(discrete_actions.long().T)
        ]
        return onehot_branches

    @staticmethod
    def dynamic_partition(
        data: torch.Tensor, partitions: torch.Tensor, num_partitions: int
    ) -> List[torch.Tensor]:
        res: List[torch.Tensor] = []
        for i in range(num_partitions):
            res += [data[(partitions == i).nonzero().squeeze(1)]]
        return res

    @staticmethod
    def masked_mean(tensor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return (tensor.T * masks).sum() / torch.clamp(
            (torch.ones_like(tensor.T) * masks).float().sum(), min=1.0
        )

    @staticmethod
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(
                source.parameters(), target.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                torch.add(
                    target_param.data,
                    source_param.data,
                    alpha=tau,
                    out=target_param.data,
                )

    @staticmethod
    def create_residual_self_attention(
        input_processors: nn.ModuleList, embedding_sizes: List[int], hidden_size: int
    ) -> Tuple[Optional[ResidualSelfAttention], Optional[LinearEncoder]]:
        rsa, x_self_encoder = None, None
        entity_num_max: int = 0
        var_processors = [p for p in input_processors if isinstance(p, EntityEmbedding)]
        for processor in var_processors:
            entity_max: int = processor.entity_num_max_elements
            if entity_max > 0:
                entity_num_max += entity_max
        if len(var_processors) > 0:
            if sum(embedding_sizes):
                x_self_encoder = LinearEncoder(
                    sum(embedding_sizes),
                    1,
                    hidden_size,
                    kernel_init=Initialization.Normal,
                    kernel_gain=(0.125 / hidden_size) ** 0.5,
                )
            rsa = ResidualSelfAttention(hidden_size, entity_num_max)
        return rsa, x_self_encoder

    @staticmethod
    def trust_region_value_loss(
        values: Dict[str, torch.Tensor],
        old_values: Dict[str, torch.Tensor],
        returns: Dict[str, torch.Tensor],
        epsilon: float,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = ModelUtils.masked_mean(torch.max(v_opt_a, v_opt_b), loss_masks)
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    @staticmethod
    def trust_region_policy_loss(
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        loss_masks: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        advantage = advantages.unsqueeze(-1)
        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = torch.clamp(r_theta, 1.0 - epsilon, 1.0 + epsilon) * advantage
        policy_loss = -1 * ModelUtils.masked_mean(
            torch.min(p_opt_a, p_opt_b), loss_masks
        )
        return policy_loss