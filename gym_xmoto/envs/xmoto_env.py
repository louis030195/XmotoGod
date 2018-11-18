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
from gym_xmoto.envs.capturedata import capturedata

import numpy as np
import subprocess
import signal
import os
import time
import cv2



class XmotoEnv(gym.Env):

  ACTION = ["w", "a", "s", "d", " ", "NA"]
  SCREEN_HEIGHT, SCREEN_WIDTH  = 720, 480
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
      return capturedata((80, 90, self.SCREEN_HEIGHT, self.SCREEN_WIDTH))


  def _action_tostring(self, action):
      return str(self.ACTION[action])

  def __init__(self):

    self.current_level = 0
    self.viewer = False
    self.state = None
    self.frameskip = (1,8)
    self.seed()
    # WASD SPACE ENTER
    self.action_space = spaces.Discrete(len(self.ACTION))
    self.observation_space = spaces.Box(low=0,
               high=255,
               shape=(int(self.SCREEN_WIDTH / 4), int(self.SCREEN_HEIGHT / 4), 4),
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

    reward = -0.1 # speed up ?

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

    template = cv2.imread('screenshots/dead.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(cv2.cvtColor(tmpState[1], cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    dead = len(loc[0]) > 0

    template = cv2.imread('screenshots/win.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(cv2.cvtColor(tmpState[1], cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    win = len(loc[0]) > 0
    #dead = pyautogui.locateOnScreen('/home/louis/Documents/xmoto-gym/screenshots/dead.png', grayscale=True) != None
    #win = pyautogui.locateOnScreen('/home/louis/Documents/xmoto-gym/screenshots/win.png', grayscale=True) != None

    episode_over = dead | win

    if dead:
        reward += -50
    if win:
        reward += 50 # TODO : hit next level key ?
        self.TOTAL_WINS += 1
        print("Total wins " + str(self.TOTAL_WINS))
        if self.TOTAL_WINS > 100:
            self.next_level()


    return tmpState[0], reward, episode_over, {dead}


  def reset(self):
    self._take_action("enter", True)
    self._take_action("enter", False)
    return self._get_state()[0]

  def render(self):
      self.process = subprocess.Popen(["faketime", "-f", "+0d x100", "xmoto", "-l", "_iL00_"])
      time.sleep(2)
      pyautogui.click(x=200, y=200)

  def next_level(self):
      if self.current_level < 44:
        self.current_level += 1
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        self.process = subprocess.Popen(["faketime", "-f", "+0d x100", "xmoto", "-l", "_iL0" + str(self.current_level) + "_"])
        pyautogui.click(x=200, y=200)

