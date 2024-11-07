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

class CheetahCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'torso_to_abduct_fr_j': 0.0 ,  # [rad]
            'torso_to_abduct_fl_j': 0.0,   # [rad]
            'torso_to_abduct_hr_j': 0.0,   # [rad]
            'torso_to_abduct_hl_j': 0.0,   # [rad]

            'abduct_fr_to_thigh_fr_j': -0.67,     # [rad]
            'abduct_fl_to_thigh_fl_j': -0.67,     # [rad]
            'abduct_hr_to_thigh_hr_j': -0.67,   # [rad]
            'abduct_hl_to_thigh_hl_j': -0.67,   # [rad]

            'thigh_fr_to_knee_fr_j': 1.25,  # [rad]
            'thigh_fl_to_knee_fl_j': 1.25,   # [rad]
            'thigh_hr_to_knee_hr_j': 1.25,    # [rad]
            'thigh_hl_to_knee_hl_j': 1.25,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'_j': 20.}  # [N*m/rad]
        damping = {'_j': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.12
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cheetah/mini_cheetah.urdf'
        name = "cheetah"
        foot_name = "None"
        penalize_contacts_on = ["shank", "thigh"]
        terminate_after_contacts_on = ["body"]
        flip_visual_attachments = False
        fix_base_link = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            # penalize angular velocity in xy axis
            ang_vel_xy = -0.05
            # penalize linear velocity in z axis
            lin_vel_z = -2.0
            # Penalize joint target changes
            action_rate = -0.01
            torques = -0.0002 * 0.5  # 0.5
            feet_air_time = 1.0 * 0.5  # 0.5

            collision = -1.0
            dof_acc = -2.5e-7 * 0.1  # 0.1
            dof_pos_limits = -10.0

            base_height = 0.0
            dof_vel = 0.0
            feet_stumble = 0.0
            orientation = 0.0
            stand_still = 0.0
            termination = 0.0

    class viewer:
        ref_env = 0
        pos = [5, 5, 3]  # [m]
        lookat = [0., 0., 0.]  # [m]

        # pos = [25, 8, 3]  # [m]
        # lookat = [20.0, 3.8, 0.0]  # [m]

class CheetahCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 1.e-3 * 0.1  # 0.1
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'cheetah'

  