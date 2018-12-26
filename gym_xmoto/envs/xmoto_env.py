#!/usr/bin/env python
# -*- coding: utf-8 -*-


# core modules
import logging.config
import math
import pkg_resources
import random
import pyautogui

from gym import spaces
import gym
from gym.utils import seeding
from gym_xmoto.envs.utils import capture_screen

import numpy as np
import subprocess
import signal
import os
import time
import cv2
from score_recognition import recognize_score



class XmotoEnv(gym.Env):

  ACTION = ["w", "a", "s", "d", " ", "NA"]
  TOTAL_WINS = 0
  LEVELS = open('gym_xmoto/envs/levels.csv', 'r').readlines()

  # DRIVE -------------------------
  """
  Start is used to define if we should keep the key down or not
  """
  def _take_action(self, key, start):
      if start:
        pyautogui.keyDown(str(key))
      else:
        pyautogui.keyUp(str(key))

  #  -------------------------


  def _get_state(self):
      return capture_screen()


  def _action_tostring(self, action):
      return str(self.ACTION[action])

  def __init__(self):

    pyautogui.FAILSAFE = False
    self.previous_score = 0
    self.levels = open('gym_xmoto/envs/levels.csv', 'r').readlines()
    self.viewer = False
    self.state = None
    self.frameskip = (1,8)
    self.seed()
    # WASD SPACE ENTER
    self.action_space = spaces.Discrete(len(self.ACTION))
    self.observation_space = spaces.Box(low=0,
               high=255,
               shape=(150, 200, 4),
               dtype=np.uint8)

  def seed(self, seed=None):
    self.np_random, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    return [seed1, seed2]


  def step(self, action):
    """
    The agent takes a step in the environment.
    Parameters
    ----------
    action : int
    Returns
    -------
    ob, reward, episode_over, info : tuple
        ob (object) :
            an environment-specific object representing your observation of
            the environment.
        reward (float) :
            amount of reward achieved by the previous action. The scale
            varies between environments, but the goal is always to increase
            your total reward.
        episode_over (bool) :
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
        info (dict) :
             diagnostic information useful for debugging. It can sometimes
             be useful for learning (for example, it might contain the raw
             probabilities behind the environment's last state change).
             However, official evaluations of your agent are not allowed to
             use this for learning.
    """

    reward = 0.5 # speed up ?
    # Frameskip stuff
    if isinstance(self.frameskip, int):
      num_steps = self.frameskip
    else:
      # Shouldn't be random but determined by the neural network
      num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
    for _ in range(num_steps):
      self._take_action(self.ACTION[action], True)
    self._take_action(self.ACTION[action], False) # Stop this action

    tmpState = self._get_state()

    if self.template_matching('skip_this_report', tmpState):
      self.render() # esc exit level, no focus on the skip button
      # Maybe can do 6 tabs to get focus on the button then enter ?

    dead = self.template_matching('dead', tmpState)

    win = self.template_matching('win', tmpState)

    #score = recognize_score(tmpState[1][0:0+30,100:100+30])

    #print("score ", score, "previous ", self.previous_score)

    episode_over = dead | win

    #if score != '':
    #  if int(score) < self.previous_score:
    #    reward += 10
    #  self.previous_score = int(score)
      

    if dead:
        reward += -10
        self.previous_score = 0
    if win:
        reward += 10
        self.TOTAL_WINS += 1
        self.previous_score = 0
        print("Total wins " + str(self.TOTAL_WINS))
        print("Reward " + str(reward))

        """
        if self.TOTAL_WINS % 100 == 0: # Next level every 100 wins
            self.next_level()
            """
        


    print("reward" + str(reward))
    return tmpState[0], reward, episode_over, {dead}

  def reset(self):
    self._take_action("enter", True)
    self._take_action("enter", False)
    return self._get_state()[0]

  def render(self):
    self.process = subprocess.Popen(["faketime", "-f", "+0d x10", "xmoto", "-l", "tut1"], preexec_fn=os.setsid)
    time.sleep(2)
    pyautogui.click(x=200, y=200)

  def close(self):
    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

  def next_level(self):
    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
    self.process = subprocess.Popen(["faketime", "-f", "+0d x10", "xmoto", "-l", self.levels[random.randint(0, len(self.levels)-1)][0:-1]],
    preexec_fn=os.setsid)
    time.sleep(2)
    pyautogui.click(x=200, y=200)

  def template_matching(self, img_name, state):
    template = cv2.imread('screenshots/' + img_name + '.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(cv2.cvtColor(np.array(state[1]), cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    return len(np.where( res >= threshold)[0]) > 0
