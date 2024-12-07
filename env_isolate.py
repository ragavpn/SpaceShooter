import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

class MultiAgentSpaceShooterEnv(gymnasium.Env):
    def __init__(self, num_agents=3, num_enemies=5, max_fps=30, reward_params=None, hyperparams=None):
        super(MultiAgentSpaceShooterEnv, self).__init__()
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.max_fps = max_fps
        self.reward_params = reward_params if reward_params else {
            'hit_reward': 10,
            'miss_penalty': -1,
            'survival_bonus': 2,
            'enemy_pass_penalty': -2,
            'health_loss': 0.01
        }
        self.hyperparams = hyperparams if hyperparams else {
            'agent_speed': 0.1,
            'enemy_speed': 0.01,
            'end_on_enemy_pass': False
        }

        # Calculate the correct observation space shape
        obs_shape = (self.num_agents + self.num_enemies + self.num_enemies + self.num_agents + self.num_agents,)
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)  # 0: stay, 1: left, 2: right, 3: shoot
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        self.bot_positions = np.full(self.num_agents, 0.5)
        self.enemy_positions = np.random.uniform(0, 1, self.num_enemies)
        self.enemy_y_positions = np.zeros(self.num_enemies)  # Y positions of enemies
        self.healths = np.ones(self.num_agents)
        self.scores = np.zeros(self.num_agents)
        self.enemy_respawn_times = np.zeros(self.num_enemies)
        self.bullets = []  # List to track bullets

        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.agent_size = 30
        self.enemy_size = 30
        self.bullet_size = 5
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

    def reset(self, seed=0):
        self.bot_positions = np.full(self.num_agents, 0.5)
        self.enemy_positions = np.random.uniform(0, 1, self.num_enemies)
        self.enemy_y_positions = np.zeros(self.num_enemies)
        self.healths = np.ones(self.num_agents)
        self.scores = np.zeros(self.num_agents)
        self.enemy_respawn_times = np.zeros(self.num_enemies)
        self.bullets = []
        return self._get_obs()

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        done = False

        agent_speed = self.hyperparams['agent_speed']
        enemy_speed = self.hyperparams['enemy_speed']
        end_on_enemy_pass = self.hyperparams['end_on_enemy_pass']

        for i, action in enumerate(actions):
            if action == 1:  # Move left
                self.bot_positions[i] = max(0, self.bot_positions[i] - agent_speed)
            elif action == 2:  # Move right
                self.bot_positions[i] = min(1, self.bot_positions[i] + agent_speed)
            elif action == 3:  # Shoot
                self.bullets.append([self.bot_positions[i], 1.0, i])  # Add bullet at agent's position with agent index

        new_bullets = []
        for bullet in self.bullets:
            bullet[1] -= 0.1  # Move bullet up
            if bullet[1] > 0:
                new_bullets.append(bullet)
        self.bullets = new_bullets

        # Check for bullet collisions with enemies
        hit_bullets = []
        for bullet in self.bullets:
            for j, (enemy_x, enemy_y) in enumerate(zip(self.enemy_positions, self.enemy_y_positions)):
                if abs(bullet[0] - enemy_x) < 0.05 and abs(bullet[1] - enemy_y) < 0.05:
                    rewards[bullet[2]] += self.reward_params['hit_reward']
                    self.scores[bullet[2]] += 1
                    self.enemy_respawn_times[j] = 50  # Set respawn time
                    self.enemy_positions[j] = np.random.uniform(0, 1)  # Respawn enemy
                    self.enemy_y_positions[j] = 0
                    hit_bullets.append(bullet)
                    break

        # Remove hit bullets from the list
        self.bullets = [bullet for bullet in self.bullets if bullet not in hit_bullets]

        # Apply miss penalty for bullets that missed
        missed_bullets = []
        for bullet in self.bullets:
            if bullet[1] <= 0:  # Bullet has moved off the screen
                rewards[bullet[2]] += self.reward_params['miss_penalty']
                missed_bullets.append(bullet)  # Mark bullet as missed

        # Remove missed bullets from the list
        self.bullets = [bullet for bullet in self.bullets if bullet not in missed_bullets]

        # Enemy moves toward the bots
        self.enemy_y_positions += enemy_speed  # Move enemies downwards based on hyperparameter
        for j, (enemy_x, enemy_y) in enumerate(zip(self.enemy_positions, self.enemy_y_positions)):
            if enemy_y >= 1:
                for i in range(self.num_agents):
                    self.healths[i] -= self.reward_params['health_loss']  # Lose health if enemy reaches the player
                    rewards[i] += self.reward_params['enemy_pass_penalty']  # Penalize when enemy passes
                self.enemy_positions[j] = np.random.uniform(0, 1)  # Respawn enemy
                self.enemy_y_positions[j] = 0  # Reset Y position
                if end_on_enemy_pass:
                    done = True  # End episode if enemy passes and hyperparameter is set

        # Handle enemy respawn
        for j in range(self.num_enemies):
            if self.enemy_respawn_times[j] > 0:
                self.enemy_respawn_times[j] -= 1
                if self.enemy_respawn_times[j] == 0:
                    self.enemy_positions[j] = np.random.uniform(0, 1)
                    self.enemy_y_positions[j] = 0

        rewards += self.reward_params['survival_bonus']  # Bonus for surviving

        if np.any(self.healths <= 0):
            done = True

        total_reward = np.sum(rewards)

        return self._get_obs(), total_reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.bot_positions, self.enemy_positions, self.enemy_y_positions, self.healths, self.scores])

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))

        # Render the bots (players) as rectangles
        for bot_pos in self.bot_positions:
            bot_x = int(bot_pos * self.screen_width)
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(bot_x, self.screen_height - self.agent_size, self.agent_size, self.agent_size))

        # Render the enemies as rectangles
        for enemy_x, enemy_y in zip(self.enemy_positions, self.enemy_y_positions):
            enemy_x = int(enemy_x * self.screen_width)
            enemy_y = int(enemy_y * self.screen_height)
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(enemy_x, enemy_y, self.enemy_size, self.enemy_size))

        # Render the bullets as rectangles
        for bullet in self.bullets:
            bullet_x = int(bullet[0] * self.screen_width)
            bullet_y = int(bullet[1] * self.screen_height)
            pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(bullet_x, bullet_y, self.bullet_size, self.bullet_size))

        # Display health and score for each agent
        for i in range(self.num_agents):
            health_text = self.font.render(f'Agent {i+1} Health: {int(self.healths[i] * 100)}%', True, (255, 255, 255))
            score_text = self.font.render(f'Agent {i+1} Score: {int(self.scores[i])}', True, (255, 255, 255))
            self.screen.blit(health_text, (10, 10 + i * 40))
            self.screen.blit(score_text, (10, 30 + i * 40))

        pygame.display.flip()
        self.clock.tick(self.max_fps)  # Limit to max FPS

    def close(self):
        pygame.quit()
