from game_env import PongEnv
from dqn_agent import Agent
import pygame
import os
print("📁 CURRENT WORKING DIRECTORY:", os.getcwd())


env = PongEnv()
agent = Agent()

episodes = 500

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if ep % 20 == 0:
            env.render()
            env.clock.tick(60)

        if done:
            print(f"Episode {ep}, Reward: {total_reward}")
            break


    if ep > 0 and ep % 50 == 0:
        agent.save()
        print("✅ Final model saved")





