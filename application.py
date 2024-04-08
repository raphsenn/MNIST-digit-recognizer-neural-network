import pygame
import numpy as np
from neuralnetwork import NeuralNetwork


class Application: 
    def __init__(self, width:int=1224, height:int=840) -> None:
        # Window settings. 
        self.WIDTH, self.HEIGHT = width, height
        self.IMG_WIDTH, self.IMG_HEIGHT = height, height
        self.PXLSIZE = height // 28

        self.SCREEN = pygame.display.set_mode((width, height))

        self.MOUSEPOS = pygame.mouse.get_pos()

        # 28 x 28 pixel.
        self.IMAGE = [[0 for i in range(28)] for j in range(28)]

        self.neuralnet = NeuralNetwork()
        self.neuralnet.load_weights()

    def draw(self):
        self.SCREEN.fill((0, 0, 0))
        for x in range(0, 28):
            for y in range(0, 28):
                grayscale_value = self.IMAGE[x][y]
                rect = pygame.Rect(x * self.PXLSIZE, y * self.PXLSIZE, self.PXLSIZE, self.PXLSIZE)
                color = (grayscale_value, grayscale_value, grayscale_value)
                pygame.draw.rect(self.SCREEN, color, rect)
        
        for line in range(0, self.IMG_WIDTH + 28, self.PXLSIZE):
            pygame.draw.line(self.SCREEN, (255, 255, 255), (0, line), (self.IMG_WIDTH, line), width=1)
            pygame.draw.line(self.SCREEN, (255, 255, 255), (line, 0), (line, self.IMG_HEIGHT), width=1)

    def draw_number(self, x, y):
        # Draw a number (digit) onto the matrix at position (x, y)
        size = 2  # Adjust this to control the size of the drawn digit
        for i in range(max(0, x - size), min(28, x + size + 1)):
            for j in range(max(0, y - size), min(28, y + size + 1)):
                distance = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
                if distance <= size:
                    intensity = max(0, min(255, (1 - distance / size) * 255))  # Adjusts intensity based on distance
                    self.IMAGE[i][j] = max(int(intensity), self.IMAGE[i][j])

    def clear_image(self):
        self.IMAGE = [[0 for i in range(28)] for j in range(28)]

    def update(self):
        self.draw()
        pygame.display.flip()

    def evaluate(self):
        img = np.array(self.IMAGE)
        img = img.T
        img = img.reshape(-1, 1)
        self.prediction = self.neuralnet.predict(img)
        print(self.prediction)

    def run(self):
        background_colour = (0,0,0)
        pressed = False
        pygame.display.set_caption('Tutorial 1')
        self.SCREEN.fill(background_colour)
        pygame.display.flip()
        running = True
        while running:
            mouse_position = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        self.clear_image()
                    elif event.key == pygame.K_RETURN:
                        self.evaluate()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pressed = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    pressed = False
                if pressed and mouse_position[0] <= self.IMG_WIDTH:
                    current_pos = pygame.mouse.get_pos() 
                    i = current_pos[0] // self.PXLSIZE
                    j = current_pos[1] // self.PXLSIZE
                    self.draw_number(i, j)        
            self.update()


app = Application()
print(app.IMAGE)
print()
app.run()
print(app.IMAGE)

