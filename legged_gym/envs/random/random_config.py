# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class RandomCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        robot_names = ['a1', 'go2', 'max', 'aliengo', 'laikago', 'cheetah']
        # robot_names = ['a1', 'go2']
        use_offset = True
    
    class init_state( LeggedRobotCfg.init_state ):
        # x,y,z [m]
        pos = {
            'a1': [0.0, 0.0, 0.42], 
            'go2': [0.0, 0.0, 0.42],
            'max': [0.0, 0.0, 0.31],
            'aliengo': [0.0, 0.0, 0.50],
            'laikago': [0.0, 0.0, 0.42],
            'cheetah': [0.0, 0.0, 0.32],
        }
        # = target angles [rad] when action = 0.0
        default_joint_angles = {
            'a1': { 
                'FL_hip_joint': 0.1,
                'RL_hip_joint': 0.1,
                'FR_hip_joint': -0.1,
                'RR_hip_joint': -0.1,

                'FL_thigh_joint': 0.8,  
                'RL_thigh_joint': 1.,
                'FR_thigh_joint': 0.8,  
                'RR_thigh_joint': 1.,

                'FL_calf_joint': -1.5,
                'RL_calf_joint': -1.5, 
                'FR_calf_joint': -1.5,
                'RR_calf_joint': -1.5, 
            },
            'go2': {
                'FL_hip_joint': 0.1,
                'RL_hip_joint': 0.1,
                'FR_hip_joint': -0.1,
                'RR_hip_joint': -0.1,

                'FL_thigh_joint': 0.8,  
                'RL_thigh_joint': 1.,
                'FR_thigh_joint': 0.8,  
                'RR_thigh_joint': 1.,

                'FL_calf_joint': -1.5,
                'RL_calf_joint': -1.5, 
                'FR_calf_joint': -1.5,
                'RR_calf_joint': -1.5, 
            },
            'max': { 
                'joint_FL1': 0.2,
                'joint_HL1': 0.2,
                'joint_FR1': -0.2,
                'joint_HR1': -0.2,

                'joint_FL2': -0.8 - 0.2,
                'joint_HL2': -1.0 - 0.2,
                'joint_FR2': -0.8 - 0.2,
                'joint_HR2': -1.0 - 0.2,

                'joint_FL3': 1.5 + 0.2,
                'joint_HL3': 1.5 + 0.2,
                'joint_FR3': 1.5 + 0.2,
                'joint_HR3': 1.5 + 0.2,
            },
            'aliengo': {
                'FL_hip_joint': 0.10,
                'RL_hip_joint': 0.10,
                'FR_hip_joint': -0.10,
                'RR_hip_joint': -0.10,

                'FL_upper_joint': 0.8,
                'RL_upper_joint': 0.8,
                'FR_upper_joint': 0.8,
                'RR_upper_joint': 0.8,

                'FL_lower_joint': -1.2,
                'RL_lower_joint': -1.2,
                'FR_lower_joint': -1.2,
                'RR_lower_joint': -1.2,
            },
            'laikago': {
                'FL_hip_motor_2_chassis_joint': 0.1,
                'RL_hip_motor_2_chassis_joint': 0.1,
                'FR_hip_motor_2_chassis_joint': -0.1,
                'RR_hip_motor_2_chassis_joint': -0.1,

                'FL_upper_leg_2_hip_motor_joint': 0.2,
                'RL_upper_leg_2_hip_motor_joint': 0.2,
                'FR_upper_leg_2_hip_motor_joint': 0.2,
                'RR_upper_leg_2_hip_motor_joint': 0.2,

                'FL_lower_leg_2_upper_leg_joint': -0.75,
                'RL_lower_leg_2_upper_leg_joint': -0.75,
                'FR_lower_leg_2_upper_leg_joint': -0.75,
                'RR_lower_leg_2_upper_leg_joint': -0.75,
            },
            'cheetah': {
                'torso_to_abduct_fr_j': 0.0,
                'torso_to_abduct_fl_j': 0.0, 
                'torso_to_abduct_hr_j': 0.0, 
                'torso_to_abduct_hl_j': 0.0, 

                'abduct_fr_to_thigh_fr_j': -0.67,   
                'abduct_fl_to_thigh_fl_j': -0.67,   
                'abduct_hr_to_thigh_hr_j': -0.67, 
                'abduct_hl_to_thigh_hl_j': -0.67, 

                'thigh_fr_to_knee_fr_j': 1.25,
                'thigh_fl_to_knee_fl_j': 1.25, 
                'thigh_hr_to_knee_hr_j': 1.25,  
                'thigh_hl_to_knee_hl_j': 1.25,  
            }
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20., '_j': 20}  # [N*m/rad]
        damping = {'joint': 0.5, '_j': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = {
            'a1': '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf',
            'go2': '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2_v2/urdf/go2.urdf',
            'max': '{LEGGED_GYM_ROOT_DIR}/resources/robots/max/max.urdf',
            'aliengo': '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/aliengo.urdf',
            'laikago': '{LEGGED_GYM_ROOT_DIR}/resources/robots/laikago/laikago_toes_zup.urdf',
            'cheetah': '{LEGGED_GYM_ROOT_DIR}/resources/robots/cheetah/mini_cheetah.urdf',
        }
        foot_name = {
            'a1': "foot",
            'go2': "foot",
            'max': "3",
            'aliengo': "lower",
            'laikago': "lower",
            'cheetah': "shank",
        }
        penalize_contacts_on = {
            'a1': ["thigh", "calf"],
            'go2': ["thigh", "calf"],
            'max': ["1", "2"],
            'aliengo': ["hip", "upper"],
            'laikago': ["hip", "upper"],
            'cheetah': ["abduct", "thigh"],
        }
        terminate_after_contacts_on = {
            'a1': ["base"],
            'go2': ["base"],
            'max': ["body"],
            'aliengo': ["trunk"],
            'laikago': ["chassis"],
            'cheetah': ["body"],
        }
        # flip_visual_attachments = False
        # fix_base_link = False
        self_collisions = 1
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    # class viewer:
    #     ref_env = 0
    #     pos = [5, 5, 3]  # [m]
    #     lookat = [0., 0., 0.]  # [m]

class RandomCfgPPO( LeggedRobotCfgPPO ):
    class policy ( LeggedRobotCfgPPO.policy ):
        # hyperparameters for the mlp
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # hyperparameters for the transformer
        shared_backbone = False
        embedding_dim = 64
        mlp_embedding = False
        mlp_prediction = True
        num_layers = 3
        num_heads = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 * 0.1
        learning_rate = 1.e-3 * 1.0
        warmup_steps = 10000
        weight_decay = 1e-4
        betas = (0.9, 0.999)
        num_learning_epochs = 10
        schedule = 'decay' # could be adaptive, fixed

    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'MLPAC'
        policy_class_name = 'TransformerAC'
        algorithm_class_name = 'PPOV2'
        run_name = ''
        experiment_name = 'random'
