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


import pygame

pygame.init()

# Set up the window
size = (1200, 1200)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Simulation")

# Load the background image
background = pygame.image.load("Visualization/plant.png")
background = pygame.transform.scale(background, (size))

# Load the fan image
fan_image = pygame.image.load('Visualization/fan.png')
fan_image = pygame.transform.scale(fan_image, (100,100))

# Set up the fan position and speed
fan_pos = [100, 300]
fan_speed = 5

# Set up the clock
clock = pygame.time.Clock()

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Update the fan speed based on the simulation time
    fan_speed = pygame.time.get_ticks() / 1000.0

    # Clear the screen
    # screen.fill((255, 255, 255))

    screen.blit(background, (100, 200))

    # Draw the fan
    
    fan_rotated = pygame.transform.rotate(fan_image, fan_speed * 10)
    fan_rect = fan_rotated.get_rect(center=fan_pos)
    screen.blit(fan_rotated, fan_rect)

    # Update the screen
    pygame.display.update()

    # Limit the frame rate
    clock.tick(60)