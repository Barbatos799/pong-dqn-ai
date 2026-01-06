from game_env import PongEnv
from dqn_agent import Agent
import pygame
import time

env = PongEnv()
agent = Agent()
agent.load("pong_dqn.pth")  # epsilon = 0 here

state = env.reset()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action = agent.act(state)
    state, _, done = env.step(action)
    env.render()
    time.sleep(0.01)

    if done:
        state = env.reset()
