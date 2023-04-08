# import pygame

# pygame.init()

# # Set up the window
# screen_width, screen_height = 800, 600
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("House Environment")

# # Set up the house
# house_width, house_height = 300, 300
# house_color = (0, 0, 255)
# house_rect = pygame.Rect(screen_width/2 - house_width/2, screen_height/2 - house_height/2, house_width, house_height)

# # Set up the temperature
# initial_temperature = 20.0  # in Celsius
# temperature_change_per_second = 1.0  # in Celsius
# current_temperature = initial_temperature

# # Set up the clock
# clock = pygame.time.Clock()

# # Start the game loop
# running = True
# while running:
#     # Handle events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Update the temperature
#     current_temperature += temperature_change_per_second * clock.get_time() / 1000.0

#     # Update the house color based on the temperature
#     temperature_color = (0, 0, 0)
#     if current_temperature > initial_temperature:
#         # If the temperature is hotter than the initial temperature, make the house more red
#         temperature_color = (min(255, int((current_temperature - initial_temperature) * 10)), 0, 0)
#     elif current_temperature < initial_temperature:
#         # If the temperature is colder than the initial temperature, make the house more blue
#         temperature_color = (0, 0, min(255, int((initial_temperature - current_temperature) * 10)))
#     house_color = temperature_color

#     # Clear the screen
#     screen.fill((255, 255, 255))

#     # Draw the house
#     pygame.draw.rect(screen, house_color, house_rect)

#     # Update the screen
#     pygame.display.flip()

#     # Wait for the next frame
#     clock.tick(60)

# # Clean up
# pygame.quit()


import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment

# Start the Unity environment
env = UnityEnvironment(file_name="HouseEnvironment")

# Set up the brain for the environment
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
agent_id = spec.create_agent()
obs_size = spec.observation_shapes[0][0]
action_size = spec.action_shape

# Reset the environment and get the initial state
env.reset()
decision_steps, _ = env.get_steps(behavior_name)
state = decision_steps.obs[0]

# Run the simulation for a few steps
for i in range(100):
    # Choose a random action
    action = np.random.uniform(-1.0, 1.0, size=action_size)

    # Take a step in the environment
    env.set_actions(behavior_name, np.array([action]))
    env.step()
    decision_steps, _ = env.get_steps(behavior_name)

    # Get the new state and reward
    state = decision_steps.obs[0]
    reward = decision_steps.reward[0]

    # Print the state and reward
    print(state, reward)