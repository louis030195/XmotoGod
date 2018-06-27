# Xmoto AI
Using part of [sentdex's self driving car in gta](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v) code
to draw the ground lines.

The idea would be to use reinforcement learning with OpenAi's gym instead of supervised learning so we won't have to play and collect data manually.

https://github.com/openai/gym

# Actions
 - W = ADVANCE
 - A = LIFT FRONT WHEEL
 - S = BRAKE
 - D = LIFT BACK WHEEL
 - SPACE = change direction
 - ENTER = reset

  # States
  - Dead = Detect when dead screen
  - Win = Detect win screen

  # Observations
   - Get the map lines slope

   <img src="screenshots/maplines.png" width="300">

  # After updating pip package
  python3 setup.py sdist bdist_wheel
  then pip install -e .

  # Usage

  ```

  import gym
  import gym_xmoto

  env = gym.make('Xmoto-v0')

  ```



# TODO
 - Get velocity, position ? => observation
 - Get angle between ground and bike to know if the wheels are lifted or not => observation
 - Get direction
 - find a way to positive reward (reward when going toward apples / objective, reward when moving ..)
 - Implement rendering
