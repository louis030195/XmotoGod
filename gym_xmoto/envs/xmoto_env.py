#!/usr/bin/env python
# -*- coding: utf-8 -*-


# core modules
import logging.config
import math
import pkg_resources
import random
import pyautogui
# 3rd party modules
from gym import spaces
import gym
import numpy as np
import subprocess
import time
from gym_xmoto.envs.capturedata2 import capturedata



class XmotoEnv(gym.Env):

  ACTION = ["w", "a", "s", "d", " ", "enter"]


  # DRIVE -------------------------
  def _take_action(self, key):
      print("key : " + str(key))
      pyautogui.keyDown(str(key))
      if str(key) == "w": # find better ?? : store prev action and act until next
          time.sleep(1)
      pyautogui.keyUp(str(key))
      return 0#.01 if str(key) == "w" else 0

  #  -------------------------


  def _get_state(self):
      screen, distance  = capturedata((80,550,120,100)) # distance to apple
      return screen, distance

  def __init__(self):
    #SCREEN_WIDTH, SCREEN_HEIGHT = 480, 720

    self.viewer = None
    self.state = None
    # WASD SPACE ENTER
    self.action_space = spaces.Discrete(6)
    self.observation_space = spaces.Box(0, 255, [120, 100, 3])
    self._prev_dist_apple = 0




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
    reward = -0.01 # speed up ?
    #if isinstance(action, int):
    reward += self._take_action(self.ACTION[action])

    ob, dist_apple = self._get_state()
    #print("DIST APPLE : "+str(dist_apple))
    if(dist_apple < self._prev_dist_apple):
        reward += 0.1
    self._prev_dist_apple = dist_apple


    dead = pyautogui.locateOnScreen('screenshots/dead.png') != None
    win = pyautogui.locateOnScreen('screenshots/win.png') != None

    episode_over = dead | win

    if dead:
        reward += -1.0

    if win:
        reward += 1.0

    # TODO : Detect when end screen appear and return as episode_over
    return ob, reward, episode_over, {dead}


  def reset(self):
    return self._take_action(self.ACTION[5])

  def render(self):
      pass