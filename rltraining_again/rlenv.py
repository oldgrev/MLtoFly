import gym
from gym import spaces
import pandas as pd
import numpy as np
from global_hotkeys import *

from torch import t
#from learnv1.learninglogger import UDPServerSocket
import pyvjoy
import socket
import io
import os
import datetime
import time
import mouse
import math
#import pyautogui

from pynput.keyboard import Key, Controller
from collections import deque

ACTION_HISTORY = 100
ACTION_AVERAGING = 2

class flyEnv(gym.Env):
    def supercrashlistener(self):
        self.supercrash = True
    def successlistener(self):
        self.success = True

    def __init__(self):
        self.supercrash = False
        bindings = [
            [["backspace"], None, self.supercrashlistener],
            [["insert"], None, self.successlistener],
        ]
        register_hotkeys(bindings)
        self.success = False
        # Finally, start listening for keypresses
        start_checking_hotkeys()
        self.cycles = 0
        self.lifetimesteps = 500
        self.maxRadalt = 0
        self.steps = 0
        self.localIP     = "127.0.0.1"
        self.localPort   = 54585
        self.bufferSize  = 1024
        self.j = pyvjoy.VJoyDevice(1)
        self.currentDT = datetime.datetime.now()
        print("File .\\telemetry\\" + self.currentDT.strftime('%Y%m%d%H%M%S') + ".csv")
        #self.telemetryfile = None
        self.UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((self.localIP, self.localPort))
        self.flushcounter = 0
        self.reward = float(0.0)
        super(flyEnv, self).__init__()
        low = np.array(
            [
                -1.0,  # collective
                -1.0,  # bank
                -1.0,  # pitch
                -1.0,  # rudder
            ]
        ).astype(np.float32)
        high = np.array(
            [
                1.0,  # collective
                1.0,  # bank
                1.0,  # pitch
                1.0,  # rudder
            ]
        ).astype(np.float32)
        self.action_space = spaces.Box(low, high)
        low = np.array(
            [
                -1.0,  # t
                -10.0,  # yaw
                -10.0,  # pitch
                -10.0,  # roll
                -10.0,  # speed.x
                -10.0,  # speed.y
                -10.0,  # speed.z
                -10.0,  # altBar
                -10.0,  # altRad
                -1.0,  # egt
                -20.0,  # yawdelta
                -20.0,  # pitchdelta
                -20.0,  # rolldelta
                -20.0,  # speed.xdelta
                -20.0,  # speed.ydelta
                -20.0,  # speed.zdelta
                -20.0,  # altBardelta
                -20.0,  # altRaddelta
                -20.0,  # egtdelta
                -41,  # myloclat
                -44,  # myloclon
            ]
        ).astype(np.float32)
        high = np.array(
            [
                500.0,
                10.0,  # yaw
                10.0,  # pitch
                10.0,  # roll
                10.0,  # speed.x
                10.0,  # speed.y
                10.0,  # speed.z
                1000.0,  # altBar
                1000.0,  # altRad
                1.0,  # egt
                20.0,  # yawdelta
                20.0,  # pitchdelta
                20.0,  # rolldelta
                20.0,  # speed.xdelta
                20.0,  # speed.ydelta
                20.0,  # speed.zdelta
                20.0,  # altBardelta
                20.0,  # altRaddelta
                20.0,  # egtdelta
                42,  # myloclat
                45,  # myloclon
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32, shape=(21,))        
        self.t = float(0.0)
        self.yaw = float(0.0)
        self.pitch = float(0.0)
        self.roll = float(0.0)
        self.speedx = float(0.0)
        self.speedy = float(0.0)
        self.speedz = float(0.0)
        self.altBar = float(0.0)
        self.altRad = float(0.0)
        self.egt = float(0.0)
        self.myloclat = float(0.0)
        self.myloclon = float(0.0) 
        #print(f'Required shape: {self.observation_space.shape}')

    def step(self, action):
        self.thisstepmaxRadalt = False
        self.prev_collective.append(((action[0] * 32767) + 32767) / 2)
        self.prev_bank.append(((action[1] * 32767) + 32767) / 2)
        self.prev_pitch.append(((action[2] * 32767) + 32767) / 2)
        self.prev_rudder.append(((action[3] * 32767) + 32767) / 2)
        info = {}
        self.j.data.wAxisZ = int(sum(self.prev_collective) / len(self.prev_collective))
        self.j.data.wAxisX = int(sum(self.prev_bank) / len(self.prev_bank))
        self.j.data.wAxisY = int(sum(self.prev_pitch) / len(self.prev_pitch))
        self.j.data.wAxisXRot = int(sum(self.prev_rudder) / len(self.prev_rudder))
        self.j.update()
        
        self.steps += 1
        if(self.steps > self.lifetimesteps):
            self.done = True
            self.success = True
            info["TimeLimit.truncated"] = True
        self.UDPServerSocket.setblocking(0)
        start = time.time()
        message = None
        bytesAddressPair = None
        while not bytesAddressPair:
            end = time.time()
            if(end - start > 20):
                print("Timeout")
                needdata = False
                keyboard = Controller()
                key = Key.esc
                keyboard.press(key)
                keyboard.release(key)
                time.sleep(1)
                mouse.move(969, 697, absolute=True, duration=0.2)
                mouse.click('left')
                time.sleep(2)
                mouse.move(1524, 931, absolute=True, duration=0.2)
                mouse.click('left')
                time.sleep(10)
                mouse.move(1514, 930, absolute=True, duration=0.2)
                mouse.click('left')

                #time.sleep(1)
                #mouse.move(1793, 845, absolute=True, duration=0.2)
                #mouse.click('left')
                #time.sleep(2)
                #mouse.move(2270, 1077, absolute=True, duration=0.2)
                #mouse.click('left')
                #time.sleep(10)
                #mouse.move(2314, 1078, absolute=True, duration=0.2)
                #mouse.click('left')
            try:
                bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
            except:
                pass
        self.UDPServerSocket.setblocking(1)
        print("Step", self.steps)      
        message = bytesAddressPair[0]
        logline = (message.decode() + "," + "{:f}".format(action[0]) + "," + "{:f}".format(action[1]) + "," + "{:f}".format(action[2]) + "," + "{:f}".format(action[3]) + "\n")
        self.telemetryfile.write(logline)
        self.flushcounter =+ 1
        if(self.flushcounter > 1000):
            self.telemetryfile.flush()
            self.flushcounter = 0        
        self.prev_actions.append(action)
        self.messagearray = message.decode().split(",")
        if(not self.lastmessagearray):
            self.lastmessagearray = self.messagearray
            self.t = float(self.messagearray[0])
            self.yaw = float(self.messagearray[1])
            self.pitch = float(self.messagearray[2])
            self.roll = float(self.messagearray[3])
            self.speedx = float(self.messagearray[4])
            self.speedy = float(self.messagearray[5])
            self.speedz = float(self.messagearray[6])
            self.altBar = float(self.messagearray[7])
            self.altRad = float(self.messagearray[8])
            self.egt = float(self.messagearray[9])
            self.myloclat = float(self.messagearray[10])
            self.myloclon = float(self.messagearray[11])   
            self.lastt, self.lastyaw, self.lastpitch, self.lastroll, self.lastspeedx, self.lastspeedy, self.lastspeedz, self.lastaltBar, self.lastaltRad, self.lastegt, self.lastmyloclat, self.lastmyloclon = self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad, self.egt, self.myloclat, self.myloclon
            observation = [self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad,
                           self.egt, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.myloclat, self.myloclon]
            observation = np.array(observation).astype(np.float32)
            done = False
            self.reward = float(1)
            return observation, self.reward, done, info
        self.lastmessagearray = self.messagearray
        self.lastt, self.lastyaw, self.lastpitch, self.lastroll, self.lastspeedx, self.lastspeedy, self.lastspeedz, self.lastaltBar, self.lastaltRad, self.lastegt, self.lastmyloclat, self.lastmyloclon = self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad, self.egt, self.myloclat, self.myloclon
        self.t = float(self.messagearray[0])
        self.yaw = float(self.messagearray[1])
        self.pitch = float(self.messagearray[2])
        self.roll = float(self.messagearray[3])
        self.speedx = float(self.messagearray[4])
        self.speedy = float(self.messagearray[5])
        self.speedz = float(self.messagearray[6])
        self.altBar = float(self.messagearray[7])
        self.altRad = float(self.messagearray[8])
        self.egt = float(self.messagearray[9])
        self.myloclat = float(self.messagearray[10])
        self.myloclon = float(self.messagearray[11])       

        if(self.maxRadalt < self.altRad):
            self.maxRadalt = self.altRad
            self.thisstepmaxRadalt = True
        self.tdelta = self.t - self.lastt
        if(self.tdelta == 0):
            observation = [self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad,
                self.egt, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.myloclat, self.myloclon]
            observation = np.array(observation).astype(np.float32)
            done = False
            self.reward = float(1)
            return observation, self.reward, done, info
        self.yawdelta = (self.yaw - self.lastyaw) / self.tdelta
        self.pitchdelta = (self.pitch - self.lastpitch) / self.tdelta
        self.rolldelta = (self.roll - self.lastroll) / self.tdelta
        self.speedxdelta = (self.speedx - self.lastspeedx) / self.tdelta
        self.speedydelta = (self.speedy - self.lastspeedy) / self.tdelta
        self.speedzdelta = (self.speedz - self.lastspeedz) / self.tdelta
        self.altBardelta = (self.altBar - self.lastaltBar) / self.tdelta
        self.altRaddelta = (self.altRad - self.lastaltRad) / self.tdelta
        self.egtdelta = (self.egt - self.lastegt) / self.tdelta
        observation = [self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad, self.egt, self.yawdelta, self.pitchdelta, self.rolldelta, self.speedxdelta, self.speedydelta, self.speedzdelta, self.altBardelta, self.altRaddelta, self.egtdelta, self.myloclat, self.myloclon]
        observation = np.array(observation).astype(np.float32)
        if(self.altRad > 5):
            self.resettime = time.time()
            print("resetting groundedtime")
        if(self.thisstepmaxRadalt):
            self.resettime = time.time()
            print("resetting groundedtime")
        grounded = False
        groundedtime = time.time() - self.resettime
        print("groundedtime: " + str(groundedtime))
        if(groundedtime > 20):
            grounded = True
        if(grounded or self.supercrash):
            self.done = True
            done = True
            print("grounded for too long, end of run")
            self.reward = -1000 + self.t
            time.sleep(1)
            keyboard = Controller()
            key = Key.esc
            keyboard.press(key)
            keyboard.release(key)
            self.UDPServerSocket.setblocking(0)
            try:
                while self.UDPServerSocket.recvfrom(self.bufferSize): pass
            except:
                pass
            self.UDPServerSocket.setblocking(1)
            mouse.move(969, 697, absolute=True, duration=0.2)
            mouse.click('left')
            time.sleep(2)
            mouse.move(1524, 931, absolute=True, duration=0.2)
            mouse.click('left')
            time.sleep(10)
            mouse.move(1514, 930, absolute=True, duration=0.2)
            mouse.click('left')
            self.reward = self.altRad
            if(abs(self.pitch) > 2 or abs(self.roll) > 2):
                self.reward = -1
            self.telemetryfile.close()
            currentDT = datetime.datetime.now()
            os.rename(".\\telemetry\\current.csv",".\\telemetry\\" + currentDT.strftime('%Y%m%d%H%M%S') + ".csv")
            print(self.reward,message.decode(),"{:f}".format(((action[0] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2), "{:f}".format(((action[3] * 32767) + 32767) / 2))
            return observation, self.reward, self.done, info
        if(self.success):
            self.done = True
            done = True
            keyboard = Controller()
            key = Key.esc
            keyboard.press(key)
            keyboard.release(key)
            time.sleep(1)
            self.UDPServerSocket.setblocking(0)
            try:
                while self.UDPServerSocket.recvfrom(self.bufferSize): pass
            except:
                pass
            self.UDPServerSocket.setblocking(1)
            mouse.move(969, 697, absolute=True, duration=0.2)
            mouse.click('left')
            time.sleep(2)
            mouse.move(1524, 931, absolute=True, duration=0.2)
            mouse.click('left')
            time.sleep(10)
            mouse.move(1514, 930, absolute=True, duration=0.2)
            mouse.click('left')            
            self.reward = self.altRad
            if(abs(self.pitch) > 2 or abs(self.roll) > 2):
                self.reward = -1
            self.telemetryfile.close()
            currentDT = datetime.datetime.now()
            os.rename(".\\telemetry\\current.csv",".\\telemetry\\" + currentDT.strftime('%Y%m%d%H%M%S') + ".csv")
            print(self.reward,message.decode(),"{:f}".format(((action[0] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2), "{:f}".format(((action[3] * 32767) + 32767) / 2))
            return observation, self.reward, self.done, info
        self.cycles += 1            
        self.UDPServerSocket.setblocking(0)
        try:
            while self.UDPServerSocket.recvfrom(self.bufferSize): pass
        except:
            pass
        self.UDPServerSocket.setblocking(1)
        self.reward = self.altRad
        reward = self.reward
        if(abs(self.pitch) > 2 or abs(self.roll) > 2):
            self.reward = -1
        print(self.reward,message.decode(),"{:f}".format(((action[0] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2) , "{:f}".format(((action[2] * 32767) + 32767) / 2), "{:f}".format(((action[3] * 32767) + 32767) / 2))
        return observation, self.reward, self.done, info

    def reset(self):
        self.telemetryfile = io.open(".\\telemetry\\current.csv", 'a')
        self.telemetryfile.write("t,yaw,pitch,roll,speed.x,speed.y,speed.z,altBar,altRad,egt,myloclat,myloclon,weightlb,collective,bank,pitch,rudder\n")
        self.telemetryfile.flush()
        self.prev_collective = deque(maxlen = ACTION_AVERAGING)  # however long we aspire the snake to befor i in range(SNAKE_LEN_GOAL):
        self.prev_bank = deque(maxlen = ACTION_AVERAGING)  # however long we aspire the snake to befor i in range(SNAKE_LEN_GOAL):
        self.prev_pitch = deque(maxlen = ACTION_AVERAGING)  # however long we aspire the snake to befor i in range(SNAKE_LEN_GOAL):
        self.prev_rudder = deque(maxlen = ACTION_AVERAGING)  # however long we aspire the snake to befor i in range(SNAKE_LEN_GOAL):
        for i in range(ACTION_AVERAGING):
            self.prev_collective.append(-1) # to create history
            self.prev_bank.append(-1) # to create history
            self.prev_pitch.append(-1) # to create history
            self.prev_rudder.append(-1) # to create history
        
        ######      
        #self.time_step = 1
        self.resettime = time.time()
        self.supercrash = False
        print("resetting")
        self.reward = 0.0
        self.steps = 0
        reward = float(0.0)
        reward = float(self.reward)
        bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
        message = bytesAddressPair[0]
        self.messagearray = message.decode().split(",")            
        self.lastmessagearray = self.messagearray
        self.t = float(self.messagearray[0])
        self.yaw = float(self.messagearray[1])
        self.pitch = float(self.messagearray[2])
        self.roll = float(self.messagearray[3])
        self.speedx = float(self.messagearray[4])
        self.speedy = float(self.messagearray[5])
        self.speedz = float(self.messagearray[6])
        self.altBar = float(self.messagearray[7])
        self.altRad = float(self.messagearray[8])
        self.egt = float(self.messagearray[9])
        self.myloclat = float(self.messagearray[10])
        self.myloclon = float(self.messagearray[11]) 
        self.cycles = 0
        self.success = False

        self.done = False
        self.prev_actions = deque(maxlen = ACTION_AVERAGING)
        for i in range(ACTION_HISTORY):
            self.prev_actions.append(-1) 
        
        self.maxRadalt = float(0)
        self.t = float(0.0)
        self.yaw = float(0.0)
        self.pitch = float(0.0)
        self.roll = float(0.0)
        self.speedx = float(0.0)
        self.speedy = float(0.0)
        self.speedz = float(0.0)
        self.altBar = float(0.0)
        self.altRad = float(0.0)
        self.egt = float(0.0)
        self.myloclat = float(0.0)
        self.myloclon = float(0.0) 
        
        keyboard = Controller()
        key = Key.f3
        keyboard.press(key)
        keyboard.release(key)
        observation = [self.t, self.yaw, self.pitch, self.roll, self.speedx, self.speedy, self.speedz, self.altBar, self.altRad,
                self.egt, float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0), self.myloclat, self.myloclon]
        observation = np.array(observation).astype(np.float32)
        #observation

        return observation
