#!/usr/bin/env python3

from RPi import GPIO
import time

output_pin = 23
buttons = [16, 18]

state = False
prev_time = 0
double_time = 0


def button_pressed(pin):
    global prev_time, double_time
    now = time.time()
    if (now - prev_time) * 1000 > 300:
        if pin == 18: # require double press of this button to prevent accidental switches
            if now - double_time <= 2:
                toggle(pin, now)
                double_time = 0
            else: # single press is fine
                double_time = now
        else:
            toggle(pin, now)
    prev_time = now


def toggle(pin, now):
    global state
    state = not state
    GPIO.output(output_pin, state)
    line = "[%s]\t%d\t%s\t%f ms" % (time.asctime(), pin, "off" if state else "on", (now - prev_time) * 1000)
    print(line)
    with open("history.txt", "a") as f:
        f.write(line + "\n")


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(output_pin, GPIO.OUT)
    for b in buttons:
        GPIO.setup(b, GPIO.IN, GPIO.PUD_UP)
        GPIO.add_event_detect(b, GPIO.FALLING, button_pressed, 200)

    input("Press Enter to exit\n")
    GPIO.output(output_pin, False)
    GPIO.cleanup()


if __name__ == "__main__":
    main()
