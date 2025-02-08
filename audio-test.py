import pygame

pygame.mixer.init()
pygame.mixer.music.load("dog.wav")  # Replace with your audio file
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pass