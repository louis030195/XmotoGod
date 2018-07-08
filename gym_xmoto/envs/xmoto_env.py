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

  ACTION = ["w", "a", "s", "d", " "]
  #ACTION = ["w", " "]
  SCREEN_HEIGHT, SCREEN_WIDTH  = 720, 480
  TOTAL_WINS = 0

  # DRIVE -------------------------
  def _take_action(self, key):
      if self.prevw == 1 :
          pyautogui.keyUp("w")
      pyautogui.keyDown(str(key))
      if str(key) == "w": # find better ?? : store prev action and act until next
          self.prevw = 1
      else :
        pyautogui.keyUp(str(key))
        self.prevw = 0
      return 0#.01 if str(key) == "w" else 0

  #  -------------------------


  def _get_state(self):
      #screen, distance  = capturedata((80,550,120,100), onlyscreen=False) # minimap
      #return screen, distance
      return capturedata((80,120,self.SCREEN_HEIGHT, self.SCREEN_WIDTH))


  def _action_tostring(self, action):
      return str(self.ACTION[action])

  def __init__(self):

    self.viewer = None
    self.state = None
    self.prevw = 1
    # WASD SPACE ENTER
    self.action_space = spaces.Discrete(len(self.ACTION))
    #self.observation_space = spaces.Box(0, 255, [120, 100, 3])
    #self.observation_space = spaces.Box(0, 255,
    #[int(self.SCREEN_HEIGHT / 4), int(self.SCREEN_WIDTH / 4), 4])
    self.observation_space = spaces.Box(low=0,
               high=255,
               shape=(int(self.SCREEN_HEIGHT / 4), int(self.SCREEN_WIDTH / 4), 3),
               dtype=np.uint8)
    #self.observation_space = spaces.Box(low=0, high=255, shape=(120, 100, 3), dtype=np.uint8)
    #self.observation_space = spaces.Discrete(1)
    #self._prev_dist_apple = 0




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
    reward = 0
    #if isinstance(action, int):
    self._take_action(self.ACTION[action])
    #if self._action_tostring(action) == "w":
    #    reward += 1
    #else :
    #    reward -= 1
    #self.state, dist_apple = self._get_state()
    self.state = self._get_state()
    #print("DIST APPLE : "+str(dist_apple))
    #if(dist_apple < self._prev_dist_apple):
    #    reward += 0.1
    #self._prev_dist_apple = dist_apple

    dead = pyautogui.locateOnScreen('screenshots/dead.png',grayscale=True) != None
    win = pyautogui.locateOnScreen('screenshots/win.png',grayscale=True) != None

    episode_over = dead | win

    print("ep over : "+str(episode_over))
    print("win : "+str(win))
    print("dead : "+str(dead))
    if win:
        reward += 1 # TODO : hit next level key ?
        self.TOTAL_WINS += 1
        print("Total wins " + str(self.TOTAL_WINS))
    if dead:
        reward -= 1


    return self.state, reward, episode_over, {dead}


  def reset(self):
    self._take_action("enter")
    #self.state = np.random.rand(1,)
    #return np.array(self.state)
    #self.state, dist_apple = self._get_state()
    #return self.state
    return self._get_state()

  def render(self):
      pass
