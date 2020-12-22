import sys
import numpy as np
import pygame
from pygame.locals import *

from mnist_test import MnistModel

class Visualizer():

    def __init__(self):
        pygame.init()

        self.canvas = pygame.display.set_mode((300, 300))
        pygame.display.set_caption('Tensorflow Visualizer')
        icon_surface = pygame.image.load('tfLogo.png')
        pygame.display.set_icon(icon_surface)

        self.clock = pygame.time.Clock()

        self.new_grid = lambda: np.zeros((28, 28), float)

        self.grid = self.new_grid()

        mnistModel = MnistModel()
        self.model = mnistModel.model

        self.last_pos = None

        while True:
            self.update()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('...Exiting')
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                pos = event.pos
                if pygame.mouse.get_pressed(3)[0] and all(10 <= val < 290 for val in pos) and self.last_pos != pos:
                    new_pos = [(val-10) // 10 for val in pos]
                    self.user_clicked(new_pos)
            elif event.type == KEYUP:
                if event.key == 13:
                    self.evaluate_image()

                    
        self.canvas.fill((200, 200,200))
        pygame.draw.rect(self.canvas, (55, 55, 55), pygame.Rect(5, 5, 290, 290))
        self.canvas.blit(self.get_image_surface(), (10, 10))

        pygame.display.update()
        self.clock.tick()

    def get_image_surface(self):
        im_surface = pygame.Surface((280, 280))
        im_surface.fill((255,)*3)
        for y in range(len(self.grid)):
            for x in range(len(self.grid)):
                val = self.grid[y][x]
                if val != 0:
                    col = ((1.0-val)*255,)*3
                    pygame.draw.rect(im_surface, col, Rect(10*x, 10*y, 10, 10))
    
        return im_surface

    def user_clicked(self, pos, opacity=0.9):
        if opacity < 0.5: return
        [x, y] = pos
        self.grid[y][x] += opacity
        if self.grid[y][x] > 1: self.grid[y][x] = 1
        for i in [-1, 1]:
            if 0 <= x+i < 28: self.user_clicked([x+i, y], opacity*0.8)
            if 0 <= y+i < 28: self.user_clicked([x, y+i], opacity*0.8)

    def evaluate_image(self):
        predictions = self.model.predict(np.array([self.grid]))
        pred = np.argmax(predictions)
        print(f'Prediction: {pred}   Confidence: {round(predictions[0][pred]*100, 4)}%')

        self.grid = self.new_grid()

if __name__ == "__main__":
    vis = Visualizer()