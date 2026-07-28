


# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause


######### MLP ####

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# from isaaclab.utils import configclass


# @configclass
# class CustomQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 50
#     experiment_name = "stanford_DelayedPDActuator"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=0.5,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.0025,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )


#####     GRU ##################

# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticRecurrentCfg,
#     RslRlPpoAlgorithmCfg,
# )

# from isaaclab.utils import configclass

# @configclass
# class CustomQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 50
#     experiment_name = "stanford_DelayedPDActuator_GRU"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticRecurrentCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#         rnn_type="gru",
#         rnn_hidden_dim=128,
#         rnn_num_layers=1,
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=0.5,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005, #entropy_coef=0.0025 with MLP (only) and entropy_coef=0.01 with GRU
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )



##### LSTM 

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.utils import configclass

@configclass
class CustomQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "stanford_DelayedPDActuator_LSTM"
    empirical_normalization = False
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# Transfomer

# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )
# from isaaclab.utils import configclass

# @configclass
# class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
#     class_name: str = "ActorCriticTransformer"
#     context_length: int = 16
#     embed_dim: int = 128
#     num_heads: int = 4
#     num_layers: int = 1
#     ffn_dim: int = 512

# @configclass
# class CustomQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 40000
#     save_interval = 50
#     experiment_name = "stanford_DelayedPDActuator_Transformer"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticTransformerCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#         context_length=16,
#         embed_dim=128,
#         num_heads=4,
#         num_layers=1,
#         ffn_dim=512,
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=0.5,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005,
#         num_learning_epochs=5,
#         num_mini_batches=16,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )