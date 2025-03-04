import random
import SeleniumAdblock
import torch
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from PIL import Image
import numpy as np
from io import BytesIO
import time

from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


class Gym:
    def __init__(self):
        # Setup WebDriver
        options = SeleniumAdblock.SeleniumAdblock()._startAdBlock()
        # options.add_argument("--headless")  # Run in headless mode
        self.driver = webdriver.Chrome(options=options)

        # Navigate to the game
        self.driver.get("https://playsuikagame.com")
        time.sleep(3)  # Wait for the game to load
        consent = self.driver.find_element(By.CSS_SELECTOR, 'body > div.fc-consent-root > div.fc-dialog-container > div.fc-dialog.fc-choice-dialog > div.fc-footer-buttons-container > div.fc-footer-buttons > button.fc-button.fc-cta-consent.fc-primary-button > p')
        if consent is not None:
            consent.click()

    def take_screenshot(self):
        screenshot = self.driver.find_element(By.ID, 'game-container').find_element(By.TAG_NAME, 'canvas').screenshot_as_png
        image = Image.open(BytesIO(screenshot))
        image = image.convert('L')  # Convert to greyscale
        image = image.resize((100, 100))  # Rescale to 100x100 pixels
        if random.randint(1, 10) == 5:
            image.save(f'screenshot{random.randint(1, 100)}.png')
        image = torch.FloatTensor(np.array(image))
        return image

    def extract_score(self):
        return int(self.driver.find_element(By.ID, 'currentScore').text)

    def game_over(self):
        return self.driver.execute_script("return isOver;")

    def click_next(self):
        button = self.driver.find_element(By.ID, 'restart-game')
        wait = WebDriverWait(self.driver, 10)
        wait.until(expected_conditions.element_to_be_clickable(button))
        button.click()

    def click_on_canvas(self, x, y):
        # Locate the game container and the sub-canvas within it
        game_container = self.driver.find_element(By.ID, 'game-container')
        sub_canvas = game_container.find_element(By.TAG_NAME, 'canvas')

        # Perform the click at the calculated position
        actions = ActionChains(self.driver)
        actions.move_to_element_with_offset(sub_canvas, x, y).click().perform()