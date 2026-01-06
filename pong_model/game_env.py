import pygame
import numpy as np
import random

WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH = WIDTH // 6
PADDLE_HEIGHT = 10
BALL_RADIUS = 7
PADDLE_SPEED = 10

class PongEnv:
    def __init__(self):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.ball_x = random.randint(100, self.width - 100)
        self.ball_y = random.randint(100, self.height // 2)

        self.ball_dx = random.choice([-4, -3, 3, 4])
        self.ball_dy = random.choice([3, 4])

        self.paddle_x = self.width // 2 - PADDLE_WIDTH // 2
        self.done = False

        return self.get_state()

    def get_state(self):
        return np.array([
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / 5,
            self.ball_dy / 5,
            (self.paddle_x - self.ball_x) / self.width
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.1  # survival reward

        # Actions: 0 = stay, 1 = left, 2 = right
        if action == 1:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED

        self.paddle_x = max(
            0,
            min(self.width - PADDLE_WIDTH, self.paddle_x)
        )

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Wall collisions
        if self.ball_x <= BALL_RADIUS or self.ball_x >= self.width - BALL_RADIUS:
            self.ball_dx *= -1
        if self.ball_y <= BALL_RADIUS:
            self.ball_dy *= -1

        # Paddle collision
        if (
            self.height - PADDLE_HEIGHT - BALL_RADIUS <= self.ball_y <= self.height - BALL_RADIUS
            and self.paddle_x <= self.ball_x <= self.paddle_x + PADDLE_WIDTH
        ):
            self.ball_dy *= -1
            reward = 1

        # Ball falls
        done = False
        if self.ball_y > self.height:
            reward = -10
            done = True

        return self.get_state(), reward, done

    def render(self):
        self.screen.fill((0, 0, 0))

        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            (self.paddle_x, self.height - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT)
        )

        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(self.ball_x), int(self.ball_y)),
            BALL_RADIUS
        )

        pygame.display.flip()
        self.clock.tick(60)
