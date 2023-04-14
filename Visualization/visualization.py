
import pygame
import random
import math
import numpy as np

def visualization(indoor_temperature, outdoor_temperature, co2, fan_speed, heating, cooling  ):

    pygame.init()

    # Set up the window
    size = (1600, 1000)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Simulation")
    clock = pygame.time.Clock()

    # Load the background image
    background = pygame.image.load("Visualization/background_officie.png")
    background = pygame.transform.scale(background, (size))

    # FAN

    fan_image1 = pygame.image.load('Visualization/fan.png')
    fan_image1 = pygame.transform.scale(fan_image1, (70, 70))
    fan_speed1 = fan_speed[0]
    fan_positions = [[849, 631], [390, 240], [240, 700], [890, 280]]

    # PEOPLE

    start_pos = [ [1100, 830], [1030,187], [900,660], [380,640], [380,700],[305,187],[305,295],[490,295], [886,697],[828,697],[600,500],[1225,328], [500, 730],[200,450]]
    end_pos = [[1030,368],[1030,187], [920,660], [380,640], [380,700],[305,187],[305,295],[490,295], [886,697],[828,697],[787,600],[1225,328],  [787,660],[730,368]]
    num_people=14
    point_pos = start_pos

    # HEATING

    square_width = 80
    square_height = 20
    square_positions = [[386, 360], [860, 100], [210, 880], [830, 490]]
    square_rects = [pygame.Rect(pos[0], pos[1], square_width, square_height) for pos in square_positions]
    current_temperature = indoor_temperature
    state_temperature_heating =  heating
    state_temperature_cooling =  cooling

    # Set up the clock
    count1 = 0
    count2=0
    next_state = 0
    timer2 = 120
    count3=0
    count4=0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(background, (0, 0))

        count4 = count4 + 1
        print(count4)
        if count4 == timer2:
            indoor_temperature = indoor_temperature + temperature_change_per_second
            outdoor_temperature = outdoor_temperature + temperature_change_per_second
            co2 = co2 + 10.0
            count4 = 0
        # Draw the square
        pygame.draw.rect(screen, (255, 255, 255), (20, 20, 200, 80), 2)

        # Display the values of the variables
        font = pygame.font.Font(None, 24)
        text_surface = font.render('Indoor Temperature: {:.1f} C'.format(indoor_temperature), True, (0, 0, 0))
        screen.blit(text_surface, (30, 30))
        text_surface = font.render('Outdoor Temperature: {:.1f} C'.format(outdoor_temperature), True, (0, 0, 0))
        screen.blit(text_surface, (30, 50))
        text_surface = font.render('CO2: {:.1f} ppm'.format(co2), True, (0, 0, 0))
        screen.blit(text_surface, (30, 70))

        #PEOPLE

        for i in range(num_people):
            if point_pos[i]!= end_pos[i]:
            # Check if the point needs to move in the x-axis or y-axis
                if point_pos[i][0] != end_pos[i][0]:
                    x_direction = 1 if end_pos[i][0] > point_pos[i][0] else -1
                    point_pos[i] = (point_pos[i][0] + x_direction, point_pos[i][1])
                else:
                    y_direction = 1 if end_pos[i][1] > point_pos[i][1] else -1
                    point_pos[i] = (point_pos[i][0], point_pos[i][1] + y_direction)

            # Check if the point has reached the end position

            if abs(point_pos[i][0] - end_pos[i][0]) + abs(point_pos[i][1] - end_pos[i][1]) <= 10:
                point_pos[i] = end_pos[i]
                
            pygame.draw.circle(screen, (255, 150, 0), point_pos[i], 10)

        #FAN
        count3 = count3 +1 
        if count3 == timer2:

            next_state = next_state + 1
            fan_speed1 = fan_speed[next_state]
            count3=0

        fan_speed2 = pygame.time.get_ticks()*fan_speed1/ 1000.0

        for i in range(len(fan_positions)):
            
            fan_rotated = pygame.transform.rotate(fan_image1, fan_speed2 * 50)
            fan_rect = fan_rotated.get_rect(center=fan_positions[i])
            screen.blit(fan_rotated, fan_rect)

        count2 = count2 + 1
        if count2 == timer2:
        
            current_temperature = state_temperature[next_state] 
            count2=0
    
        # Heating

        for i in range(len(square_rects)):
            # Update the color based on the temperature
            temperature_color = (0, 0, 0)
            if current_temperature >= 25:
                temperature_color = (255, 51, 51)
            elif current_temperature >= 22:
                temperature_color = (139, 0, 0)
            elif current_temperature <= 14:
                temperature_color = (0,90, 230) 
            elif current_temperature <= 18:
                temperature_color = (0, 0, 230)

            pygame.draw.rect(screen, temperature_color, square_rects[i])

        
        # Update the screen
        pygame.display.update()

        # Limit the frame rate
        clock.tick(60)

        return


