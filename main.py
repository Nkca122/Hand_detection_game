# init
import pygame
import pygame.camera
from pygame.locals import *
import math
import cv2
import mediapipe as mp
import numpy as np

pygame.init()
pygame.camera.init()
# player
import player

# Screen
screen = pygame.display.set_mode((1280, 720))
## Title and Icon
pygame.display.set_caption(title="Space Invaders Clone")
pygame.display.set_icon(pygame.image.load("./assets/ufo.png"))
## clock
clock = pygame.time.Clock()

# Player
img = pygame.image.load("./assets/spaceship.png")
p = player.Player(img, screen=screen)

# Game Loop
running = True
key = None
while running:
    screen.fill((0, 0, 0))
    for e in pygame.event.get():
        # close button
        if e.type == pygame.QUIT:
            running = False
    p.update(screen=screen)
    # update display after all the game functions
    pygame.display.update()
    clock.tick(32)

pygame.quit()
quit()
