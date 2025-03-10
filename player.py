import pygame
import pygame.camera as camera
from pygame.locals import *
import cv2
import mediapipe as mp
import numpy as np
import math



class Detector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    def detect_hand_orientation(self, hand_landmarks, screen, frame):
        index_tip = hand_landmarks.landmark[8] # index finger tip location in the frame
        centerX = frame.shape[0]/2
        centerY = frame.shape[1]/2

        index_tipX = index_tip.x * frame.shape[0] - centerX
        index_tipY = index_tip.y * frame.shape[1] - centerY

        return [screen.get_width() - centerX, centerY, screen.get_width() - centerX + index_tipX, centerY + index_tipY]

    def count_fingers(self, hand_landmarks):
        finger_tip_ids = [8, 12, 16, 20] #id - 1 = mid section of the fingers
        raised_fingers = []
        wrist = 0

        if hand_landmarks:
            for id in finger_tip_ids:
                if (hand_landmarks.landmark[id].y - hand_landmarks.landmark[id-1].y)*(hand_landmarks.landmark[wrist].y - hand_landmarks.landmark[id-1].y) <= 0:
                    raised_fingers.append(1)
                else:
                    raised_fingers.append(0)
        return raised_fingers.count(1), raised_fingers, finger_tip_ids
    



class Controller:
    def __init__(self, screen):
            self.size = (368, 368)
            self.display = screen
            self.cam_rect = pygame.Rect((screen.get_width() - self.size[0], 0), self.size)
            self.clist = pygame.camera.list_cameras()
            if not self.clist:
                raise ValueError("No cameras detected.")
            self.cam = pygame.camera.Camera(self.clist[0], self.size)
            self.cam.start()
            self.cam.set_controls(hflip=True, vflip=False)
            self.snapshot = pygame.surface.Surface(self.size, 0, self.display)
            self.detector = Detector() 
            self.count = 0
            self.orientation = 0

    def update(self, angle, screen):
        finger_rects = []
        self.orientation = []
        if self.cam.query_image():
            self.snapshot = self.cam.get_image(self.snapshot)
            
            frame = pygame.surfarray.array3d(self.snapshot)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = self.detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fingers = None
            finger_tip_ids = None
            if results.multi_hand_landmarks:
                for hand_marks in  results.multi_hand_landmarks:
                    self.orientation = self.detector.detect_hand_orientation(hand_marks, screen, frame)
                    self.count, fingers, finger_tip_ids = self.detector.count_fingers(hand_marks)
                    for mark in hand_marks.landmark:
                         finger_rects.append(pygame.Rect((self.cam_rect.x + mark.x * self.size[0], self.cam_rect.y + mark.y * self.size[1]), (12, 12)))
        self.display.blit(self.snapshot, self.cam_rect)
        if self.orientation:
            pygame.draw.line(self.display, (225, 0, 0), start_pos=(self.orientation[0], self.orientation[1]), end_pos=(self.display.get_width(), self.orientation[1]))
            pygame.draw.line(self.display, (225, 0, 0), start_pos=(self.orientation[0], self.orientation[1]), end_pos=(self.orientation[2], self.orientation[3]))
        if finger_rects:
            for idx, rect in enumerate(finger_rects):
                if idx in finger_tip_ids and fingers[finger_tip_ids.index(idx)]:
                    pygame.draw.rect(self.display, (0, 225, 0), rect, 3)
                else:
                    pygame.draw.rect(self.display, (225, 0, 0), rect, 1)
             
       

class Player:
    def __init__(self, img, screen, width=64, height=64):
        self.img = pygame.transform.scale(img, (width, height))
        self.original_img = self.img
        self.x = (screen.get_width() - width) / 2
        self.y = (screen.get_height() - height) / 2
        self.rect = self.img.get_rect(center=(self.x + width / 2, self.y + height / 2))
        self.speed = 5
        self.rotate = 1
        self.angle = 0
        self.controls = Controller(screen)

    def update(self, screen):
        self.__move(screen)
        self.controls.update(self.angle, screen)

        # angle calculation for image roatation

        angle = 0
        if self.angle >= 0 and self.angle < 90:
            angle = -(90 - self.angle)
        else:
            angle = self.angle - 90

        rotated_img = pygame.transform.rotate(self.original_img, angle)
        rotated_rect = rotated_img.get_rect(center=self.rect.center)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 5)

        screen.blit(pygame.font.SysFont('timesnewroman',  30).render(f"{self.angle}", False, (255, 255, 255)), (16, 16))
        screen.blit(pygame.font.SysFont('timesnewroman',  30).render(f"{angle}", False, (255, 255, 255)), (16, 60))
        screen.blit(pygame.font.SysFont('timesnewroman',  30).render(f"{self.controls.count}", False, (255, 255, 255)), (16, 90))

        screen.blit(rotated_img, rotated_rect)

        self.img = rotated_img
        self.rect = rotated_rect

    def __move(self, screen):
        if self.controls.count > 0:
            self.__translate(screen)
            self.__rotate(screen, self.controls.orientation)

    def __rotate(self, screen, dir):
        if dir:
            A = (dir[2] - dir[0], dir[3] - dir[1])
            B = (screen.get_width() - dir[0], dir[3] - dir[1])

            # θ = acos(A.B/|A||B|)

            dot_product = A[0] * B[0] + A[1] * B[1]
            magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
            magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

            angle_deg = None

            if magnitude_A * magnitude_B != 0:
                angle_rad = math.acos(dot_product / (magnitude_A * magnitude_B))
                angle_deg = math.degrees(angle_rad)

                # for reflexive angles
                if dir[3] > dir[1]:
                    angle_deg = 360-angle_deg
            else:
                angle_deg = 0
            
            self.angle = angle_deg

    def __translate(self, screen):
        x_delta = self.speed * math.cos(math.radians(self.angle))
        y_delta = self.speed * math.sin(math.radians(self.angle))

        self.rect.x += x_delta
        self.rect.y -= y_delta

        self.rect.clamp_ip(screen.get_rect())
