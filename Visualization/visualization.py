
import pygame
import random
import math
import numpy as np
import matplotlib as plt
def thermal_bar(indoor_temperature, screen1):


    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    gray = (128, 128, 128)

    font = pygame.font.SysFont('Arial', 20)

    thermal_width, thermal_height = 20, 200
    thermal_x, thermal_y = 10, 10
    thermal_rect = pygame.Rect(thermal_x, thermal_y, thermal_width, thermal_height)

    arrow_width, arrow_height = 30, 5
    arrow_x, arrow_y = thermal_x - arrow_width - 5, thermal_y + thermal_height // 2 - arrow_height // 2

    arrow_rect = pygame.Rect(arrow_x, arrow_y, arrow_width, arrow_height)
    temp_min, temp_max = 0, 35

    # Update the thermal arrow position and color

    thermal_range = temp_max - temp_min
    arrow_position = int((indoor_temperature - temp_min) / thermal_range * thermal_height)
    arrow_rect.y = thermal_y + thermal_height - arrow_position - arrow_height
    
    if indoor_temperature >= 17:
        arrow_color = red
    else:
        arrow_color = blue
    
    # Draw the thermal bar
    pygame.draw.rect(screen1, gray, thermal_rect)
    thermal_value = int((indoor_temperature - temp_min) / thermal_range * thermal_height)
    
    if indoor_temperature >= 17:
        thermal_color = red
    else:
        thermal_color = blue
    pygame.draw.rect(screen1, thermal_color, (thermal_x, thermal_y + thermal_height - thermal_value, thermal_width, thermal_value))
    
    # Draw the thermal arrow
    pygame.draw.rect(screen1, arrow_color, arrow_rect)
    
    # Draw the temperature label
    label_text = f"Indoor Temperature: {indoor_temperature} °C"
    label_surface = font.render(label_text, True, black)
    screen1.blit(label_surface, (thermal_x + thermal_width + 10, thermal_y + thermal_height - label_surface.get_height() // 2))

    return

def plot_energy(our_energy, real_energy, next_state):

    plt.plot(our_energy, next_state)
    plt.plot(real_energy, next_state)
    plt.show()

    return

def visualization(indoor_temperature, outdoor_temperature, co2, fan_speed, heating, cooling, thermal_comfort  ):

    pygame.init()
    
    # Set up the window
    size = (1600, 1000)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Simulation")
    # screen = pygame.display.set_mode((300, 300)) 
    # pygame.display.set_caption("Window 2")  
    clock = pygame.time.Clock()
    
    # Load the background image
    background = pygame.image.load("Visualization/background_officie.png")
    background = pygame.transform.scale(background, (size))

    #Face
    smiley_image = pygame.image.load("Visualization/smileface.png")
    sad_face_image = pygame.image.load("Visualization/sadface.png")

    # FAN

    fan_image1 = pygame.image.load('Visualization/fan.png')
    fan_image1 = pygame.transform.scale(fan_image1, (70, 70))
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

    # Set up the clock

    count1 = 0
    next_state = 0
    timer2 = 60

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(background, (0, 0))

        count1 = count1 + 1
        if count1 == timer2:
            next_state = next_state + 1
            count1 = 0
        
        # draw_thermal_bar(indoor_temperature[next_state], screen)
        #thermal_bar(indoor_temperature[next_state], screen )

        if thermal_comfort[next_state] >= 0.5:
            smiley_image = pygame.transform.scale(smiley_image, (30, 30))
            image_face = smiley_image
        else:
            sad_face_image = pygame.transform.scale(sad_face_image, (33, 23))
            image_face = sad_face_image
            
        # Draw the square

        # screen.fill((255, 255, 255))

        pygame.draw.rect(screen, (255, 255, 255), (20, 20, 200, 80), 2)

        # Display the values of the variables
        font = pygame.font.Font(None, 24)
        screen.blit(image_face, (172, 0))
        text = font.render("Thermal Comfort: ", True,  (0, 0, 0))
        screen.blit(text, (30, 10))
        text_surface = font.render('Indoor Temperature: {:.1f} C'.format(indoor_temperature[next_state]), True, (0, 0, 0))
        screen.blit(text_surface, (30, 30))
        text_surface = font.render('Outdoor Temperature: {:.1f} C'.format(outdoor_temperature[next_state]), True, (0, 0, 0))
        screen.blit(text_surface, (30, 50))
        text_surface = font.render('CO2: {:.1f} ppm'.format(co2[next_state]), True, (0, 0, 0))
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

        fan_speed1 = pygame.time.get_ticks()* fan_speed[next_state]/ 1000.0

        for i in range(len(fan_positions)):
            
            fan_rotated = pygame.transform.rotate(fan_image1, fan_speed1 * 10)
            fan_rect = fan_rotated.get_rect(center=fan_positions[i])
            screen.blit(fan_rotated, fan_rect)
    
        current_temperature = indoor_temperature[next_state]


        for i in range(len(square_rects)):
            # Update the color based on the temperature
            temperature_color = (0, 0, 0)
            # if current_temperature >= 25:
            #     temperature_color = (255, 51, 51)
            # elif current_temperature >= 22:
            #     temperature_color = (139, 0, 0)
            # elif current_temperature <= 14:
            #     temperature_color = (0,90, 230) 
            # elif current_temperature <= 18:
            #     temperature_color = (0, 0, 230)
            if current_temperature > cooling[next_state]:
                temperature_color = (0, 0, 230)
            elif current_temperature < heating[next_state]:
                temperature_color = (139, 0, 0)
            pygame.draw.rect(screen, temperature_color, square_rects[i])

        # plot_energy(our_energy, real_energy, next_state)

        # Update the screen1
        # pygame.display.update()
        pygame.display.update()

        # Limit the frame ratescreen2
        clock.tick(60)

    return



if __name__ == '__main__':

    indoor_temperature = [15, 16, 17, 18 , 19 , 20 , 21, 21, 21, 21]
    outdoor_temperature = [25, 26, 27, 28 , 29 , 20 , 22, 22, 23, 24]
    co2 = [25, 26, 27, 28 , 29 , 20 , 22, 22, 23, 24] 
    fan_speed = [2, 10 ,2 , 7 ,1 ,5, 7, 3 , 0, 2]
    heating = [25, 25, 25,14, 14 ,14, 14, 14, 14]
    cooling = [25, 25, 25,13, 13 ,13, 13, 13, 13]
    thermal_comfort = [0.3, 0.3, 0.3 , 0.6 , 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    # real_energy = [90, 100, 125, 105, 110, 90]
    # our_energy = [70, 79, 100, 85, 89, 60]
    visualization( indoor_temperature, outdoor_temperature, co2, fan_speed, heating, cooling,thermal_comfort)

    # Se a temperatura interior for menor que o heating, vai aquecer 
    # Se a temperatura interior for maior que o cooling vai arrefecer 