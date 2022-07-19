#!/usr/bin/env python3

import time
import datetime
import random


def main():
    start = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    events = [(start, False)]

    for i in range(90):
        day = start + datetime.timedelta(days=i)

        if day.weekday() >= 5: # weekend
            on_mu    = 8.5
            on_sigma = 60/60
        else: # weekday
            on_mu    = 6.5
            on_sigma = 10/60

        if day.weekday() == 4 or day.weekday() == 5: # weekendnight
            off_mu    = 23.75
            off_sigma = 30/60
        else: # weeknight
            off_mu    = 22.5
            off_sigma = 30/60

        on = day + datetime.timedelta(hours=random.gauss(on_mu, on_sigma))
        off = day + datetime.timedelta(hours=random.gauss(off_mu, off_sigma))

        events.append((on,  True))
        events.append((off, False))

    with open("example_history.txt", "w") as f:
        for e in events:
            t = time.asctime(datetime.datetime.timetuple(e[0]))
            pin = 1
            state = "on" if e[1] else "off"
            d = "0.000000 ms"

            line = f"[{t}]\t{pin}\t{state}\t{d}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
